// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "quantization/scalar_quantization/scalar_quantization_utils.h"
#include "simd/kernels/compute_l2.h"
#include "simd/kernels/sq8_compute.h"
#include "simd/traits/simd_traits_generic.h"

namespace {

using Clock = std::chrono::steady_clock;

template <typename T>
struct Records {
    std::vector<T> values;
    uint64_t count = 0;
};

template <typename T>
Records<T>
read_records(const std::filesystem::path& path, uint64_t expected_dim) {
    std::ifstream input(path, std::ios::binary);
    if (not input) {
        throw std::runtime_error("cannot open " + path.string());
    }
    Records<T> result;
    std::vector<T> row(expected_dim);
    while (true) {
        int32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (input.gcount() == 0 and input.eof()) {
            break;
        }
        if (input.gcount() != sizeof(dim) or dim != static_cast<int32_t>(expected_dim)) {
            throw std::runtime_error("invalid record dimension in " + path.string());
        }
        input.read(reinterpret_cast<char*>(row.data()), sizeof(T) * expected_dim);
        if (not input) {
            throw std::runtime_error("truncated record in " + path.string());
        }
        if constexpr (std::is_same_v<T, float>) {
            if (std::any_of(
                    row.begin(), row.end(), [](float value) { return not std::isfinite(value); })) {
                throw std::runtime_error("non-finite vector in " + path.string());
            }
        }
        result.values.insert(result.values.end(), row.begin(), row.end());
        ++result.count;
    }
    if (result.count == 0) {
        throw std::runtime_error("empty records in " + path.string());
    }
    return result;
}

struct SQ8Codes {
    std::vector<float> lower;
    std::vector<float> diff;
    std::vector<uint8_t> codes;
    double rmse = 0;
};

SQ8Codes
train_and_encode(const std::vector<float>& base, uint64_t count, uint64_t dim) {
    if (count == 0 or dim == 0 or count > base.max_size() / dim or base.size() != count * dim) {
        throw std::invalid_argument("invalid training matrix");
    }
    SQ8Codes model;
    model.lower.assign(dim, std::numeric_limits<float>::max());
    std::vector<float> upper(dim, std::numeric_limits<float>::lowest());
    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t d = 0; d < dim; ++d) {
            const float value = base[i * dim + d];
            if (not std::isfinite(value)) {
                throw std::invalid_argument("non-finite training vector");
            }
            model.lower[d] = std::min(model.lower[d], value);
            upper[d] = std::max(upper[d], value);
        }
    }
    model.diff.resize(dim);
    for (uint64_t d = 0; d < dim; ++d) {
        model.diff[d] = upper[d] - model.lower[d];
        if (not std::isfinite(model.diff[d])) {
            throw std::invalid_argument("training range overflow");
        }
    }
    model.codes.resize(count * dim);
    long double squared_error = 0;
    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t d = 0; d < dim; ++d) {
            const uint64_t offset = i * dim + d;
            const float range = model.diff[d];
            const float delta = range < std::numeric_limits<float>::epsilon()
                                    ? 1.0F
                                    : (base[offset] - model.lower[d]) / range;
            const float clamped = vsag::ClampScalarQuantizationDelta(delta);
            const auto code = static_cast<uint8_t>(255.0F * clamped);
            model.codes[offset] = code;
            const float decoded = (static_cast<float>(code) / 255.0F) * range + model.lower[d];
            const long double error = static_cast<long double>(base[offset]) - decoded;
            squared_error += error * error;
        }
    }
    model.rmse = std::sqrt(static_cast<double>(squared_error / (count * dim)));
    return model;
}

struct Candidate {
    uint64_t id;
    float distance;
};

bool
better(const Candidate& left, const Candidate& right) {
    return left.distance < right.distance or
           (left.distance == right.distance and left.id < right.id);
}

template <typename Distance>
std::vector<Candidate>
top_k(uint64_t count, uint64_t k, Distance distance) {
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> heap(&better);
    for (uint64_t id = 0; id < count; ++id) {
        Candidate next{id, distance(id)};
        if (heap.size() < k) {
            heap.push(next);
        } else if (better(next, heap.top())) {
            heap.pop();
            heap.push(next);
        }
    }
    std::vector<Candidate> result(heap.size());
    for (uint64_t i = result.size(); i > 0; --i) {
        result[i - 1] = heap.top();
        heap.pop();
    }
    return result;
}

double
percentile(std::vector<double> samples, double fraction) {
    std::sort(samples.begin(), samples.end());
    const auto offset = static_cast<uint64_t>(fraction * static_cast<double>(samples.size() - 1));
    return samples.at(offset);
}

uint64_t
hits(const std::vector<Candidate>& neighbors, const int32_t* truth, uint64_t k) {
    std::unordered_set<int32_t> expected(truth, truth + k);
    if (expected.size() != k) {
        throw std::runtime_error("duplicate ground-truth ID");
    }
    uint64_t total = 0;
    for (const auto& neighbor : neighbors) {
        total += expected.count(static_cast<int32_t>(neighbor.id));
    }
    return total;
}

void
self_test() {
    const std::vector<float> base{0, 5, 1, 5, 2, 5, 3, 5};
    const auto model = train_and_encode(base, 4, 2);
    if (model.lower != std::vector<float>{0, 5} or model.diff != std::vector<float>{3, 0} or
        model.codes.size() != 8 or model.codes[0] != 0 or model.codes[6] != 255) {
        throw std::runtime_error("SQ8 boundary encoding failed");
    }
    const float query[2]{0, 5};
    for (uint64_t id = 0; id < 4; ++id) {
        const float distance =
            vsag::simd::SQ8ComputeL2SqrImpl<vsag::simd::SQ8Traits<vsag::simd::GenericSQ8Tag>>(
                query, model.codes.data() + id * 2, model.lower.data(), model.diff.data(), 2);
        if (not std::isfinite(distance) or (id == 0 and distance != 0)) {
            throw std::runtime_error("SQ8 distance failed");
        }
    }
    try {
        train_and_encode({0, std::numeric_limits<float>::quiet_NaN()}, 1, 2);
        throw std::runtime_error("non-finite training data was accepted");
    } catch (const std::invalid_argument&) {
    }
}

void
run(const std::filesystem::path& root) {
    constexpr uint64_t dim = 128;
    constexpr uint64_t k = 10;
    const auto base = read_records<float>(root / "base.fvecs", dim);
    const auto queries = read_records<float>(root / "queries.fvecs", dim);
    const auto truth = read_records<int32_t>(root / "groundtruth.ivecs", k);
    if (queries.count != truth.count or base.count < k or
        base.count > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("inconsistent SIFT dataset");
    }
    const auto start = Clock::now();
    const auto model = train_and_encode(base.values, base.count, dim);
    const double train_encode_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::vector<double> exact_us;
    std::vector<double> sq8_us;
    uint64_t exact_hits = 0;
    uint64_t sq8_hits = 0;
    for (uint64_t q = 0; q < queries.count; ++q) {
        const float* query = queries.values.data() + q * dim;
        auto exact_scan = [&] {
            return top_k(base.count, k, [&](uint64_t id) {
                return vsag::simd::ComputeL2SqrImpl<vsag::simd::SimdTraits<vsag::simd::GenericTag>>(
                    query, base.values.data() + id * dim, dim);
            });
        };
        auto sq8_scan = [&] {
            return top_k(base.count, k, [&](uint64_t id) {
                return vsag::simd::SQ8ComputeL2SqrImpl<
                    vsag::simd::SQ8Traits<vsag::simd::GenericSQ8Tag>>(query,
                                                                      model.codes.data() + id * dim,
                                                                      model.lower.data(),
                                                                      model.diff.data(),
                                                                      dim);
            });
        };
        auto measure = [](const auto& scan, std::vector<double>& latencies) {
            const auto begin = Clock::now();
            auto results = scan();
            latencies.push_back(
                std::chrono::duration<double, std::micro>(Clock::now() - begin).count());
            return results;
        };
        std::vector<Candidate> exact;
        std::vector<Candidate> approximate;
        if (q % 2 == 0) {
            exact = measure(exact_scan, exact_us);
            approximate = measure(sq8_scan, sq8_us);
        } else {
            approximate = measure(sq8_scan, sq8_us);
            exact = measure(exact_scan, exact_us);
        }
        const int32_t* expected = truth.values.data() + q * k;
        for (uint64_t i = 0; i < k; ++i) {
            if (expected[i] < 0 or static_cast<uint64_t>(expected[i]) >= base.count) {
                throw std::runtime_error("ground-truth ID outside base");
            }
        }
        exact_hits += hits(exact, expected, k);
        sq8_hits += hits(approximate, expected, k);
    }
    const uint64_t opportunities = queries.count * k;
    if (exact_hits != opportunities) {
        throw std::runtime_error("provided ground truth differs from FP32 exhaustive scan");
    }
    std::cout << "base_count,query_count,dim,train_encode_ms,exact_recall_at_10,"
                 "sq8_recall_at_10,exact_p50_us,exact_p99_us,sq8_p50_us,sq8_p99_us,"
                 "rmse,fp32_vector_bytes,sq8_code_bytes,sq8_model_bytes\n";
    std::cout << std::fixed << std::setprecision(6) << base.count << ',' << queries.count << ','
              << dim << ',' << train_encode_ms << ','
              << static_cast<double>(exact_hits) / static_cast<double>(opportunities) << ','
              << static_cast<double>(sq8_hits) / static_cast<double>(opportunities) << ','
              << percentile(exact_us, 0.50) << ',' << percentile(exact_us, 0.99) << ','
              << percentile(sq8_us, 0.50) << ',' << percentile(sq8_us, 0.99) << ',' << model.rmse
              << ',' << base.values.size() * sizeof(float) << ',' << model.codes.size() << ','
              << (model.lower.size() + model.diff.size()) * sizeof(float) << '\n';
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        if (argc == 2 and std::string(argv[1]) == "--self-test") {
            self_test();
            return 0;
        }
        if (argc != 2) {
            std::cerr << "usage: lite_sq8_probe SIFT_DIR | --self-test\n";
            return 2;
        }
        run(argv[1]);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
