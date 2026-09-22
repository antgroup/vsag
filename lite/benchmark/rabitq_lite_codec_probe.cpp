// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "simd/kernels/rabitq_pack.h"

namespace {
constexpr uint32_t K_TOTAL_BITS = 8;
constexpr uint32_t K_FILTER_BITS = 3;
constexpr uint32_t K_SUPPLEMENT_BITS = 5;
constexpr uint32_t K_ROUNDS = 4;
constexpr uint32_t K_ENCODE_ROUNDS = 6;
constexpr float K_ERROR_RATE = 1.9F;

using Clock = std::chrono::steady_clock;

void
require(bool value, const char* message) {
    if (not value) {
        throw std::runtime_error(message);
    }
}

void
fht(std::vector<float>& values) {
    for (uint64_t step = 1; step < values.size(); step *= 2) {
        for (uint64_t block = 0; block < values.size(); block += step * 2) {
            for (uint64_t lane = 0; lane < step; ++lane) {
                const float left = values[block + lane];
                const float right = values[block + step + lane];
                values[block + lane] = left + right;
                values[block + step + lane] = left - right;
            }
        }
    }
}

struct Model {
    uint64_t dim{};
    std::vector<float> centroid;
    std::vector<uint8_t> flips;

    void
    Transform(std::vector<float>& values) const {
        const uint64_t bytes = (dim + 7) / 8;
        const float scale = 1.0F / std::sqrt(static_cast<float>(dim));
        for (uint32_t round = 0; round < K_ROUNDS; ++round) {
            for (uint64_t d = 0; d < dim; ++d) {
                if ((flips[round * bytes + d / 8] & (1U << (d % 8))) != 0U) {
                    values[d] = -values[d];
                }
            }
            fht(values);
            for (float& value : values) {
                value *= scale;
            }
        }
    }
};

Model
train(const std::vector<float>& base, uint64_t count, uint64_t dim, uint32_t seed) {
    require(count > 0 and dim > 0 and base.size() == count * dim, "invalid training matrix");
    require((dim & (dim - 1)) == 0, "probe FHT dimension must be a power of two");
    Model model{
        dim, std::vector<float>(dim, 0.0F), std::vector<uint8_t>(K_ROUNDS * ((dim + 7) / 8))};
    for (float value : base) {
        require(std::isfinite(value), "non-finite training value");
    }
    for (uint64_t i = 0; i < count; ++i) {
        for (uint64_t d = 0; d < dim; ++d) {
            model.centroid[d] += base[i * dim + d];
        }
    }
    for (float& value : model.centroid) {
        value /= static_cast<float>(count);
    }
    std::mt19937 generator(seed);
    std::uniform_int_distribution<uint32_t> bytes(0, 255);
    for (uint8_t& value : model.flips) {
        value = static_cast<uint8_t>(bytes(generator));
    }
    model.Transform(model.centroid);
    return model;
}

std::vector<uint8_t>
fast_encode(const std::vector<float>& normalized, float& code_norm) {
    constexpr uint32_t code_max = 255;
    constexpr double center = 127.5;
    double max_abs = 0.0;
    for (float value : normalized) {
        max_abs = std::max(max_abs, std::fabs(static_cast<double>(value)));
    }
    std::vector<uint8_t> codes(normalized.size(), 127);
    if (max_abs == 0.0) {
        code_norm = 1.0F;
        return codes;
    }
    const double inverse_delta = 128.0 / max_abs;
    double ip = 0.0;
    double norm_sqr = 0.0;
    for (uint64_t d = 0; d < normalized.size(); ++d) {
        auto code = static_cast<int64_t>(std::floor((normalized[d] + max_abs) * inverse_delta));
        code = std::clamp<int64_t>(code, 0, code_max);
        codes[d] = static_cast<uint8_t>(code);
        const double centered = static_cast<double>(code) - center;
        ip += normalized[d] * centered;
        norm_sqr += centered * centered;
    }
    const auto score = [](double inner, double norm) {
        return norm > 0.0 ? inner * inner / norm : 0.0;
    };
    for (uint32_t round = 0; round < K_ENCODE_ROUNDS; ++round) {
        bool changed = false;
        for (uint64_t d = 0; d < normalized.size(); ++d) {
            const int32_t current = codes[d];
            const double current_value = current - center;
            int32_t best = current;
            double best_ip = ip;
            double best_norm = norm_sqr;
            double best_score = score(ip, norm_sqr);
            for (int32_t direction : {-1, 1}) {
                const int32_t candidate = current + direction;
                if (candidate < 0 or candidate > static_cast<int32_t>(code_max)) {
                    continue;
                }
                const double next_ip =
                    ip + static_cast<double>(direction) * static_cast<double>(normalized[d]);
                if (next_ip < 0.0) {
                    continue;
                }
                const double next_norm = norm_sqr + 2.0 * current_value * direction + 1.0;
                const double next_score = score(next_ip, next_norm);
                if (next_score > best_score + 1e-8 * std::max(1.0, std::fabs(best_score))) {
                    best = candidate;
                    best_ip = next_ip;
                    best_norm = next_norm;
                    best_score = next_score;
                }
            }
            if (best != current) {
                codes[d] = static_cast<uint8_t>(best);
                ip = best_ip;
                norm_sqr = best_norm;
                changed = true;
            }
        }
        if (not changed) {
            break;
        }
    }
    norm_sqr = 0.0;
    for (uint8_t code : codes) {
        const double value = code - center;
        norm_sqr += value * value;
    }
    code_norm = static_cast<float>(std::sqrt(norm_sqr));
    if (not std::isfinite(code_norm) or code_norm <= 0.0F) {
        code_norm = 1.0F;
    }
    return codes;
}

struct Encoded {
    std::vector<uint8_t> filter, supplement, scalar;
    float norm{}, code_norm{}, error{}, filter_norm{}, filter_error{}, lower_bound_error{};
};

std::vector<float>
normalize(const Model& model, const float* input, float& norm) {
    std::vector<float> values(input, input + model.dim);
    for (float value : values) {
        require(std::isfinite(value), "non-finite vector");
    }
    model.Transform(values);
    double squared = 0.0;
    for (uint64_t d = 0; d < model.dim; ++d) {
        values[d] -= model.centroid[d];
        squared += values[d] * values[d];
    }
    norm = squared < 1e-5 ? 1.0F : static_cast<float>(std::sqrt(squared));
    for (float& value : values) {
        value /= norm;
    }
    return values;
}

Encoded
encode(const Model& model, const float* input) {
    Encoded result;
    auto normalized = normalize(model, input, result.norm);
    result.scalar = fast_encode(normalized, result.code_norm);
    const uint64_t plane_bytes = (model.dim + 7) / 8;
    result.filter.assign(plane_bytes * K_FILTER_BITS, 0);
    result.supplement.assign(plane_bytes * K_SUPPLEMENT_BITS, 0);
    vsag::simd::RaBitQPackScalarToSplitPlanesTail(result.scalar.data(),
                                                  result.filter.data(),
                                                  result.supplement.data(),
                                                  model.dim,
                                                  K_TOTAL_BITS,
                                                  K_FILTER_BITS,
                                                  0);
    double full_ip = 0.0;
    double full_norm_sqr = 0.0;
    double filter_ip = 0.0;
    double filter_norm_sqr = 0.0;
    double query_sum = 0.0;
    for (uint64_t d = 0; d < model.dim; ++d) {
        const float query = normalized[d];
        const float code = result.scalar[d];
        const auto filter_code = static_cast<float>(result.scalar[d] >> K_SUPPLEMENT_BITS);
        full_ip += query * code;
        full_norm_sqr += (code - 127.5F) * (code - 127.5F);
        filter_ip += query * filter_code;
        filter_norm_sqr += (filter_code - 3.5F) * (filter_code - 3.5F);
        query_sum += query;
    }
    result.code_norm = static_cast<float>(std::sqrt(full_norm_sqr));
    result.filter_norm = static_cast<float>(std::sqrt(filter_norm_sqr));
    result.error = static_cast<float>((full_ip - 127.5 * query_sum) / result.code_norm);
    result.filter_error =
        std::fabs(static_cast<float>((filter_ip - 3.5 * query_sum) / result.filter_norm));
    result.filter_error = std::clamp(result.filter_error, 1e-5F, 1.0F);
    result.lower_bound_error =
        std::sqrt(std::max(0.0F, 1.0F - result.filter_error * result.filter_error) /
                  std::max(1.0F, static_cast<float>(model.dim - 1)));
    require(std::isfinite(result.error) and std::isfinite(result.filter_error) and
                std::isfinite(result.lower_bound_error),
            "non-finite encoding metadata");
    return result;
}

uint32_t
read_plane_code(const uint8_t* planes,
                uint64_t plane_bytes,
                uint64_t d,
                uint32_t bits,
                bool most_significant_first) {
    const auto mask = static_cast<uint8_t>(1U << (d & 7U));
    const uint64_t byte = d >> 3U;
    uint32_t code = 0;
    for (uint32_t bit = 0; bit < bits; ++bit) {
        if ((planes[bit * plane_bytes + byte] & mask) != 0U) {
            code += most_significant_first ? 1U << (bits - bit - 1U) : 1U << bit;
        }
    }
    return code;
}

float
filter_centered_ip(const std::vector<float>& query, const uint8_t* filter) {
    const uint64_t plane_bytes = (query.size() + 7) / 8;
    float result = 0.0F;
    for (uint64_t d = 0; d < query.size(); ++d) {
        const auto code = read_plane_code(filter, plane_bytes, d, K_FILTER_BITS, true);
        result += query[d] * (static_cast<float>(code) - 3.5F);
    }
    return result;
}

float
supplement_ip(const std::vector<float>& query, const uint8_t* supplement) {
    const uint64_t plane_bytes = (query.size() + 7) / 8;
    float result = 0.0F;
    for (uint64_t d = 0; d < query.size(); ++d) {
        const auto code = read_plane_code(supplement, plane_bytes, d, K_SUPPLEMENT_BITS, false);
        result += query[d] * static_cast<float>(code);
    }
    return result;
}

float
l2_distance(float base_norm, float query_norm, float normalized_ip) {
    return base_norm * base_norm + query_norm * query_norm -
           2.0F * base_norm * query_norm * normalized_ip;
}

struct FilterEstimate {
    float distance;
    float lower_bound;
    float centered_ip;
};

FilterEstimate
filter_estimate(const std::vector<float>& query, float query_norm, const Encoded& code) {
    const float centered_ip = filter_centered_ip(query, code.filter.data());
    const float normalized_ip = centered_ip / code.filter_norm / code.filter_error;
    const float distance = l2_distance(code.norm, query_norm, normalized_ip);
    const float error =
        2.0F * code.norm * query_norm * K_ERROR_RATE * code.lower_bound_error / code.filter_error;
    const float estimate = distance - error;
    const float lower_bound = estimate - 1e-5F * std::max(1.0F, std::fabs(estimate));
    return {distance, lower_bound, centered_ip};
}

float
full_distance(const std::vector<float>& query,
              float query_norm,
              const Encoded& code,
              float centered_filter_ip) {
    const float query_sum = std::accumulate(query.begin(), query.end(), 0.0F);
    const float filter_ip = centered_filter_ip + 3.5F * query_sum;
    const float code_ip = filter_ip * static_cast<float>(1U << K_SUPPLEMENT_BITS) +
                          supplement_ip(query, code.supplement.data());
    const float base_error = std::fabs(code.error) < 1e-5F ? 1.0F : code.error;
    const float normalized_ip = (code_ip - 127.5F * query_sum) / code.code_norm / base_error;
    return l2_distance(code.norm, query_norm, normalized_ip);
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
        const Candidate next{id, distance(id)};
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

struct SearchResult {
    std::vector<Candidate> neighbors;
    uint64_t reordered{};
};

SearchResult
filtered_search(const std::vector<float>& query,
                float query_norm,
                const std::vector<Encoded>& codes,
                uint64_t k) {
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> heap(&better);
    uint64_t reordered = 0;
    for (uint64_t id = 0; id < codes.size(); ++id) {
        const auto coarse = filter_estimate(query, query_norm, codes[id]);
        if (heap.size() == k and coarse.lower_bound >= heap.top().distance) {
            continue;
        }
        ++reordered;
        const Candidate next{id, full_distance(query, query_norm, codes[id], coarse.centered_ip)};
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
    return {std::move(result), reordered};
}

template <typename T>
struct Records {
    std::vector<T> values;
    uint64_t count{};
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
        require(input.gcount() == sizeof(dim) and dim == static_cast<int32_t>(expected_dim),
                "invalid record dimension");
        input.read(reinterpret_cast<char*>(row.data()), sizeof(T) * expected_dim);
        require(static_cast<bool>(input), "truncated record");
        if constexpr (std::is_same_v<T, float>) {
            require(std::all_of(
                        row.begin(), row.end(), [](float value) { return std::isfinite(value); }),
                    "non-finite vector");
        }
        result.values.insert(result.values.end(), row.begin(), row.end());
        ++result.count;
    }
    require(result.count > 0, "empty records");
    return result;
}

uint64_t
hits(const std::vector<Candidate>& neighbors, const int32_t* truth, uint64_t k) {
    const std::unordered_set<int32_t> expected(truth, truth + k);
    require(expected.size() == k, "duplicate ground-truth ID");
    uint64_t total = 0;
    for (const auto& neighbor : neighbors) {
        total += expected.count(static_cast<int32_t>(neighbor.id));
    }
    return total;
}

void
write_u64(std::ostream& output, uint64_t value, uint64_t bytes = 8) {
    for (uint64_t i = 0; i < bytes; ++i) {
        output.put(static_cast<char>((value >> (8U * i)) & 0xffU));
    }
    require(static_cast<bool>(output), "snapshot write failed");
}

uint64_t
read_u64(std::istream& input, uint64_t bytes = 8) {
    uint64_t value = 0;
    for (uint64_t i = 0; i < bytes; ++i) {
        const int byte = input.get();
        require(byte != std::char_traits<char>::eof(), "truncated snapshot");
        value |= static_cast<uint64_t>(static_cast<uint8_t>(byte)) << (8U * i);
    }
    return value;
}

void
write_float(std::ostream& output, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u64(output, bits, sizeof(bits));
}

float
read_float(std::istream& input) {
    const auto bits = static_cast<uint32_t>(read_u64(input, sizeof(uint32_t)));
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    require(std::isfinite(value), "non-finite snapshot metadata");
    return value;
}

void
write_bytes(std::ostream& output, const std::vector<uint8_t>& bytes) {
    if (not bytes.empty()) {
        output.write(reinterpret_cast<const char*>(bytes.data()),
                     static_cast<std::streamsize>(bytes.size()));
    }
    require(static_cast<bool>(output), "snapshot write failed");
}

void
read_bytes(std::istream& input, std::vector<uint8_t>& bytes) {
    if (not bytes.empty()) {
        input.read(reinterpret_cast<char*>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
    }
    require(static_cast<bool>(input), "truncated snapshot");
}

void
save_snapshot(std::ostream& output, const Model& model, const std::vector<Encoded>& codes) {
    constexpr char magic[] = "VSLRBQ01";
    output.write(magic, 8);
    write_u64(output, 1);
    write_u64(output, model.dim);
    write_u64(output, codes.size());
    write_u64(output, model.centroid.size());
    write_u64(output, model.flips.size());
    for (float value : model.centroid) {
        write_float(output, value);
    }
    write_bytes(output, model.flips);
    const uint64_t plane_bytes = (model.dim + 7) / 8;
    for (const auto& code : codes) {
        require(code.filter.size() == plane_bytes * K_FILTER_BITS and
                    code.supplement.size() == plane_bytes * K_SUPPLEMENT_BITS,
                "invalid encoded record");
        write_float(output, code.norm);
        write_float(output, code.code_norm);
        write_float(output, code.error);
        write_float(output, code.filter_norm);
        write_float(output, code.filter_error);
        write_float(output, code.lower_bound_error);
        write_bytes(output, code.filter);
        write_bytes(output, code.supplement);
    }
}

std::pair<Model, std::vector<Encoded>>
load_snapshot(std::istream& input) {
    char magic[8]{};
    input.read(magic, sizeof(magic));
    require(input and std::memcmp(magic, "VSLRBQ01", 8) == 0, "invalid snapshot magic");
    require(read_u64(input) == 1, "unsupported snapshot version");
    const uint64_t dim = read_u64(input);
    const uint64_t count = read_u64(input);
    const uint64_t centroid_size = read_u64(input);
    const uint64_t flips_size = read_u64(input);
    require(dim > 0 and dim <= (1U << 20U) and (dim & (dim - 1)) == 0 and centroid_size == dim,
            "invalid snapshot model");
    const uint64_t plane_bytes = (dim + 7) / 8;
    require(flips_size == K_ROUNDS * plane_bytes and count <= 1000000, "invalid snapshot layout");
    Model model{dim, std::vector<float>(dim), std::vector<uint8_t>(flips_size)};
    for (float& value : model.centroid) {
        value = read_float(input);
    }
    read_bytes(input, model.flips);
    std::vector<Encoded> codes(count);
    for (auto& code : codes) {
        code.norm = read_float(input);
        code.code_norm = read_float(input);
        code.error = read_float(input);
        code.filter_norm = read_float(input);
        code.filter_error = read_float(input);
        code.lower_bound_error = read_float(input);
        require(code.norm > 0.0F and code.code_norm > 0.0F and code.filter_norm > 0.0F and
                    code.filter_error >= 1e-5F and code.filter_error <= 1.0F and
                    code.lower_bound_error >= 0.0F,
                "invalid snapshot metadata");
        code.filter.resize(plane_bytes * K_FILTER_BITS);
        code.supplement.resize(plane_bytes * K_SUPPLEMENT_BITS);
        read_bytes(input, code.filter);
        read_bytes(input, code.supplement);
    }
    require(input.peek() == std::char_traits<char>::eof(), "snapshot trailing bytes");
    return {std::move(model), std::move(codes)};
}

void
self_test() {
    constexpr uint64_t dim = 128;
    constexpr uint64_t count = 64;
    std::mt19937 generator(91);
    std::normal_distribution<float> distribution;
    std::vector<float> base(count * dim);
    for (float& value : base) {
        value = distribution(generator);
    }
    const auto first = train(base, count, dim, 47);
    const auto second = train(base, count, dim, 47);
    require(first.centroid == second.centroid and first.flips == second.flips,
            "model is not deterministic");
    const auto encoded = encode(first, base.data());
    const auto repeated = encode(first, base.data());
    require(encoded.scalar == repeated.scalar and encoded.filter == repeated.filter and
                encoded.supplement == repeated.supplement,
            "encoding is not deterministic");
    require(encoded.filter.size() == 48 and encoded.supplement.size() == 80,
            "unexpected 128D layout");
    float direct = 0.0F;
    float split = 0.0F;
    auto query = normalize(first, base.data() + dim, direct);
    direct = 0.0F;
    for (uint64_t d = 0; d < dim; ++d) {
        direct += query[d] * static_cast<float>(encoded.scalar[d]);
    }
    const uint64_t plane_bytes = 16;
    for (uint64_t d = 0; d < dim; ++d) {
        const auto mask = static_cast<uint8_t>(1U << (d & 7U));
        const uint64_t byte = d >> 3U;
        uint32_t code = 0;
        for (uint32_t bit = 0; bit < K_FILTER_BITS; ++bit) {
            if ((encoded.filter[bit * plane_bytes + byte] & mask) != 0U) {
                code += 1U << (K_SUPPLEMENT_BITS + K_FILTER_BITS - bit - 1U);
            }
        }
        for (uint32_t bit = 0; bit < K_SUPPLEMENT_BITS; ++bit) {
            if ((encoded.supplement[bit * plane_bytes + byte] & mask) != 0U) {
                code += 1U << bit;
            }
        }
        split += query[d] * static_cast<float>(code);
    }
    require(std::fabs(direct - split) <= 1e-4F * std::max(1.0F, std::fabs(direct)),
            "codec split mismatch");
    std::vector<Encoded> codes;
    codes.reserve(count);
    for (uint64_t id = 0; id < count; ++id) {
        codes.push_back(encode(first, base.data() + id * dim));
    }
    float query_norm = 0.0F;
    const auto normalized_query = normalize(first, base.data() + dim, query_norm);
    const auto full = top_k(count, 10, [&](uint64_t id) {
        const auto coarse = filter_estimate(normalized_query, query_norm, codes[id]);
        return full_distance(normalized_query, query_norm, codes[id], coarse.centered_ip);
    });
    const auto filtered = filtered_search(normalized_query, query_norm, codes, 10);
    require(filtered.reordered > 0 and filtered.reordered <= count,
            "invalid filtered search count");
    require(std::equal(full.begin(),
                       full.end(),
                       filtered.neighbors.begin(),
                       [](const auto& left, const auto& right) {
                           return left.id == right.id and
                                  std::fabs(left.distance - right.distance) <=
                                      1e-4F * std::max(1.0F, std::fabs(left.distance));
                       }),
            "lower-bound filtering changed full-code top-k");
    std::stringstream snapshot(std::ios::in | std::ios::out | std::ios::binary);
    save_snapshot(snapshot, first, codes);
    const std::string bytes = snapshot.str();
    require(not bytes.empty(), "empty snapshot");
    snapshot.seekg(0);
    auto [loaded_model, loaded_codes] = load_snapshot(snapshot);
    require(loaded_model.dim == first.dim and loaded_model.centroid == first.centroid and
                loaded_model.flips == first.flips and loaded_codes.size() == codes.size(),
            "snapshot model mismatch");
    float loaded_query_norm = 0.0F;
    const auto loaded_query = normalize(loaded_model, base.data() + dim, loaded_query_norm);
    const auto loaded = filtered_search(loaded_query, loaded_query_norm, loaded_codes, 10);
    require(std::equal(filtered.neighbors.begin(),
                       filtered.neighbors.end(),
                       loaded.neighbors.begin(),
                       [](const auto& left, const auto& right) {
                           return left.id == right.id and left.distance == right.distance;
                       }),
            "snapshot search mismatch");
    for (const std::string& invalid : {bytes.substr(0, bytes.size() - 1),
                                       std::string("BADMAGIC") + bytes.substr(8),
                                       bytes + std::string(1, '\0')}) {
        std::stringstream damaged(invalid, std::ios::in | std::ios::binary);
        try {
            static_cast<void>(load_snapshot(damaged));
            throw std::runtime_error("damaged snapshot was accepted");
        } catch (const std::runtime_error& error) {
            require(std::string(error.what()) != "damaged snapshot was accepted",
                    "damaged snapshot was accepted");
        }
    }
}

void
run(const std::filesystem::path& root) {
    constexpr uint64_t dim = 128;
    constexpr uint64_t k = 10;
    const auto base = read_records<float>(root / "base.fvecs", dim);
    const auto queries = read_records<float>(root / "queries.fvecs", dim);
    const auto truth = read_records<int32_t>(root / "groundtruth.ivecs", k);
    require(queries.count == truth.count and base.count >= k and
                base.count <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
            "inconsistent SIFT dataset");
    const auto build_start = Clock::now();
    const auto model = train(base.values, base.count, dim, 47);
    std::vector<Encoded> codes;
    codes.reserve(base.count);
    for (uint64_t id = 0; id < base.count; ++id) {
        codes.push_back(encode(model, base.values.data() + id * dim));
        codes.back().scalar.clear();
        codes.back().scalar.shrink_to_fit();
    }
    const double build_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - build_start).count();
    uint64_t full_hits = 0;
    uint64_t filtered_hits = 0;
    uint64_t agreement = 0;
    uint64_t reordered = 0;
    std::vector<double> search_us;
    for (uint64_t q = 0; q < queries.count; ++q) {
        float query_norm = 0.0F;
        const auto query = normalize(model, queries.values.data() + q * dim, query_norm);
        const auto full = top_k(base.count, k, [&](uint64_t id) {
            const auto coarse = filter_estimate(query, query_norm, codes[id]);
            return full_distance(query, query_norm, codes[id], coarse.centered_ip);
        });
        const auto start = Clock::now();
        const auto filtered = filtered_search(query, query_norm, codes, k);
        search_us.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        const int32_t* expected = truth.values.data() + q * k;
        for (uint64_t i = 0; i < k; ++i) {
            require(expected[i] >= 0 and static_cast<uint64_t>(expected[i]) < base.count,
                    "ground-truth ID outside base");
        }
        full_hits += hits(full, expected, k);
        filtered_hits += hits(filtered.neighbors, expected, k);
        reordered += filtered.reordered;
        for (uint64_t i = 0; i < k; ++i) {
            agreement += static_cast<uint64_t>(full[i].id == filtered.neighbors[i].id);
        }
    }
    std::sort(search_us.begin(), search_us.end());
    const uint64_t opportunities = queries.count * k;
    const uint64_t plane_bytes = (dim + 7) / 8;
    std::cout << "base_count,query_count,dim,build_encode_ms,full_recall_at_10,"
                 "filtered_recall_at_10,filtered_full_agreement,mean_reordered,reorder_ratio,"
                 "search_p50_us,filter_bytes,supplement_bytes,metadata_bytes\n";
    std::cout << std::fixed << std::setprecision(6) << base.count << ',' << queries.count << ','
              << dim << ',' << build_ms << ','
              << static_cast<double>(full_hits) / static_cast<double>(opportunities) << ','
              << static_cast<double>(filtered_hits) / static_cast<double>(opportunities) << ','
              << static_cast<double>(agreement) / static_cast<double>(opportunities) << ','
              << static_cast<double>(reordered) / static_cast<double>(queries.count) << ','
              << static_cast<double>(reordered) / static_cast<double>(queries.count * base.count)
              << ',' << search_us[search_us.size() / 2] << ','
              << base.count * plane_bytes * K_FILTER_BITS << ','
              << base.count * plane_bytes * K_SUPPLEMENT_BITS << ','
              << base.count * 6 * sizeof(float) << '\n';
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        if (argc == 1 or (argc == 2 and std::string(argv[1]) == "--self-test")) {
            self_test();
            std::cout << "rabitq_lite_codec_probe: PASS\n";
            return 0;
        }
        if (argc == 2) {
            run(argv[1]);
            return 0;
        }
        std::cerr << "usage: lite_rabitq_codec_probe [SIFT_DIR | --self-test]\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
