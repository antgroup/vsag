// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include <sys/resource.h>

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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "lite/backend.h"
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

double
microseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::micro>(end - start).count();
}

double
milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double
percentile(const std::vector<double>& sorted, double fraction) {
    require(not sorted.empty(), "cannot compute an empty percentile");
    const auto position =
        static_cast<uint64_t>(std::ceil(fraction * static_cast<double>(sorted.size())));
    return sorted[std::max<uint64_t>(1, position) - 1];
}

uint64_t
peak_rss_kib() {
    rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
#if defined(__APPLE__)
    return static_cast<uint64_t>(usage.ru_maxrss) / 1024;
#else
    return static_cast<uint64_t>(usage.ru_maxrss);
#endif
}

uint64_t
current_rss_kib() {
#if defined(__linux__)
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream fields(line);
            std::string key;
            uint64_t value = 0;
            std::string unit;
            fields >> key >> value >> unit;
            require(unit == "kB", "unexpected VmRSS unit");
            return value;
        }
    }
    throw std::runtime_error("VmRSS is unavailable");
#else
    return peak_rss_kib();
#endif
}

void
fht(float* values, uint64_t dim) {
    for (uint64_t step = 1; step < dim; step *= 2) {
        for (uint64_t block = 0; block < dim; block += step * 2) {
            for (uint64_t lane = 0; lane < step; ++lane) {
                const float left = values[block + lane];
                const float right = values[block + step + lane];
                values[block + lane] = left + right;
                values[block + step + lane] = left - right;
            }
        }
    }
}

void
kacs_walk(std::vector<float>& values) {
    const uint64_t half = values.size() / 2;
    const uint64_t base = values.size() % 2;
    const uint64_t offset = base + half;
    for (uint64_t i = 0; i < half; ++i) {
        const float left = values[i];
        const float right = values[i + offset];
        values[i] = left + right;
        values[i + offset] = left - right;
    }
    if (base != 0) {
        values[half] *= std::sqrt(2.0F);
    }
}

uint64_t
floor_power_of_two(uint64_t value) {
    uint64_t result = 1;
    while (result <= value / 2) {
        result *= 2;
    }
    return result;
}

struct Model {
    uint64_t dim{};
    std::vector<float> centroid;
    std::vector<uint8_t> flips;

    void
    Transform(std::vector<float>& values) const {
        require(values.size() == dim, "transform dimension mismatch");
        const uint64_t bytes = (dim + 7) / 8;
        const uint64_t truncated_dim = floor_power_of_two(dim);
        const float scale = 1.0F / std::sqrt(static_cast<float>(truncated_dim));
        for (uint32_t round = 0; round < K_ROUNDS; ++round) {
            for (uint64_t d = 0; d < dim; ++d) {
                if ((flips[round * bytes + d / 8] & (1U << (d % 8))) != 0U) {
                    values[d] = -values[d];
                }
            }
            float* block = round % 2 == 0 ? values.data() : values.data() + dim - truncated_dim;
            fht(block, truncated_dim);
            for (uint64_t d = 0; d < truncated_dim; ++d) {
                block[d] *= scale;
            }
            if (truncated_dim != dim) {
                kacs_walk(values);
            }
        }
        if (truncated_dim != dim) {
            for (float& value : values) {
                value *= 0.25F;
            }
        }
    }
};

Model
train(const std::vector<float>& base, uint64_t count, uint64_t dim, uint32_t seed) {
    require(count > 0 and dim > 0 and base.size() == count * dim, "invalid training matrix");
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

struct EncodedMetadata {
    float norm{}, code_norm{}, error{}, filter_norm{}, filter_error{}, lower_bound_error{};
};

struct EncodedView {
    const uint8_t* filter;
    const uint8_t* supplement;
    EncodedMetadata metadata;
};

struct EncodedRecords {
    explicit EncodedRecords(uint64_t input_dim) : dim(input_dim) {
    }

    [[nodiscard]] uint64_t
    FilterBytes() const {
        return ((dim + 7) / 8) * K_FILTER_BITS;
    }

    [[nodiscard]] uint64_t
    SupplementBytes() const {
        return ((dim + 7) / 8) * K_SUPPLEMENT_BITS;
    }

    [[nodiscard]] uint64_t
    Size() const {
        return metadata.size();
    }

    void
    Reserve(uint64_t count) {
        metadata.reserve(count);
        filters.reserve(count * FilterBytes());
        supplements.reserve(count * SupplementBytes());
    }

    void
    Resize(uint64_t count) {
        metadata.resize(count);
        filters.resize(count * FilterBytes());
        supplements.resize(count * SupplementBytes());
    }

    void
    Append(Encoded code) {
        require(code.filter.size() == FilterBytes() and code.supplement.size() == SupplementBytes(),
                "invalid encoded record");
        metadata.push_back({code.norm,
                            code.code_norm,
                            code.error,
                            code.filter_norm,
                            code.filter_error,
                            code.lower_bound_error});
        filters.insert(filters.end(), code.filter.begin(), code.filter.end());
        supplements.insert(supplements.end(), code.supplement.begin(), code.supplement.end());
    }

    void
    Replace(uint64_t slot, Encoded code) {
        require(slot < Size() and code.filter.size() == FilterBytes() and
                    code.supplement.size() == SupplementBytes(),
                "invalid encoded replacement");
        metadata[slot] = {code.norm,
                          code.code_norm,
                          code.error,
                          code.filter_norm,
                          code.filter_error,
                          code.lower_bound_error};
        std::copy(code.filter.begin(), code.filter.end(), filters.data() + slot * FilterBytes());
        std::copy(code.supplement.begin(),
                  code.supplement.end(),
                  supplements.data() + slot * SupplementBytes());
    }

    void
    RemoveSwap(uint64_t slot) {
        require(slot < Size(), "encoded removal outside storage");
        const uint64_t last = Size() - 1;
        if (slot != last) {
            metadata[slot] = metadata[last];
            std::copy_n(filters.data() + last * FilterBytes(),
                        FilterBytes(),
                        filters.data() + slot * FilterBytes());
            std::copy_n(supplements.data() + last * SupplementBytes(),
                        SupplementBytes(),
                        supplements.data() + slot * SupplementBytes());
        }
        metadata.pop_back();
        filters.resize(last * FilterBytes());
        supplements.resize(last * SupplementBytes());
    }

    [[nodiscard]] EncodedView
    At(uint64_t id) const {
        require(id < Size(), "encoded record outside storage");
        return {filters.data() + id * FilterBytes(),
                supplements.data() + id * SupplementBytes(),
                metadata[id]};
    }

    uint64_t dim;
    std::vector<EncodedMetadata> metadata;
    std::vector<uint8_t> filters;
    std::vector<uint8_t> supplements;
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
filter_estimate(const std::vector<float>& query, float query_norm, const EncodedView& code) {
    const float centered_ip = filter_centered_ip(query, code.filter);
    const float normalized_ip =
        centered_ip / code.metadata.filter_norm / code.metadata.filter_error;
    const float distance = l2_distance(code.metadata.norm, query_norm, normalized_ip);
    const float error = 2.0F * code.metadata.norm * query_norm * K_ERROR_RATE *
                        code.metadata.lower_bound_error / code.metadata.filter_error;
    const float estimate = distance - error;
    const float lower_bound = estimate - 1e-5F * std::max(1.0F, std::fabs(estimate));
    return {distance, lower_bound, centered_ip};
}

float
full_distance(const std::vector<float>& query,
              float query_norm,
              const EncodedView& code,
              float centered_filter_ip) {
    const float query_sum = std::accumulate(query.begin(), query.end(), 0.0F);
    const float filter_ip = centered_filter_ip + 3.5F * query_sum;
    const float code_ip = filter_ip * static_cast<float>(1U << K_SUPPLEMENT_BITS) +
                          supplement_ip(query, code.supplement);
    const float base_error = std::fabs(code.metadata.error) < 1e-5F ? 1.0F : code.metadata.error;
    const float normalized_ip =
        (code_ip - 127.5F * query_sum) / code.metadata.code_norm / base_error;
    return l2_distance(code.metadata.norm, query_norm, normalized_ip);
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

bool
farther(const Candidate& left, const Candidate& right) {
    return left.distance > right.distance or
           (left.distance == right.distance and left.id > right.id);
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
                const EncodedRecords& codes,
                uint64_t k) {
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> heap(&better);
    uint64_t reordered = 0;
    for (uint64_t id = 0; id < codes.Size(); ++id) {
        const auto code = codes.At(id);
        const auto coarse = filter_estimate(query, query_norm, code);
        if (heap.size() == k and coarse.lower_bound >= heap.top().distance) {
            continue;
        }
        ++reordered;
        const Candidate next{id, full_distance(query, query_norm, code, coarse.centered_ip)};
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

struct GraphTopology {
    [[nodiscard]] uint64_t
    Size() const {
        return offsets.empty() ? 0 : offsets.size() - 1;
    }

    std::vector<uint64_t> offsets;
    std::vector<uint64_t> neighbors;
};

GraphTopology
build_graph_topology(const std::vector<float>& base,
                     uint64_t count,
                     uint64_t dim,
                     uint64_t max_degree,
                     uint64_t ef_search) {
    auto created = vsag::lite::detail::make_brute_force_backend(dim);
    require(static_cast<bool>(created), "cannot create Lite build source");
    auto source = std::move(*created);
    for (uint64_t slot = 0; slot < count; ++slot) {
        auto added = source->Add(static_cast<int64_t>(slot), base.data() + slot * dim, dim);
        require(static_cast<bool>(added), "cannot populate Lite build source");
    }
    auto built = vsag::lite::detail::make_graph_backend(*source, max_degree, ef_search);
    require(static_cast<bool>(built), "cannot build Lite graph topology");
    auto graph = std::move(*built);
    GraphTopology result;
    result.offsets.reserve(count + 1);
    result.neighbors.reserve(count * max_degree);
    result.offsets.push_back(0);
    for (uint64_t slot = 0; slot < count; ++slot) {
        require(graph->IdAt(slot) == static_cast<int64_t>(slot), "graph slot order changed");
        const uint64_t degree = graph->LinkCountAt(slot);
        require(degree <= max_degree, "graph degree exceeds limit");
        for (uint64_t edge = 0; edge < degree; ++edge) {
            const uint64_t neighbor = graph->LinkAt(slot, edge);
            require(neighbor < count and neighbor != slot, "invalid graph edge");
            result.neighbors.push_back(neighbor);
        }
        result.offsets.push_back(result.neighbors.size());
    }
    return result;
}

struct GraphSearchResult {
    std::vector<Candidate> neighbors;
    uint64_t visited{};
    uint64_t reordered{};
};

GraphSearchResult
graph_search(const std::vector<float>& query,
             float query_norm,
             const EncodedRecords& codes,
             const GraphTopology& graph,
             uint64_t k,
             uint64_t ef_search) {
    require(graph.Size() == codes.Size(), "graph and encoded records disagree");
    k = std::min(k, codes.Size());
    if (k == 0) {
        return {};
    }
    const uint64_t ef = std::min(codes.Size(), std::max(k, ef_search));
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> best(&better);
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&farther)> candidates(&farther);
    std::vector<uint8_t> visited(codes.Size(), 0);
    uint64_t visited_count = 0;

    auto visit = [&](uint64_t slot) {
        if (visited[slot] != 0) {
            return;
        }
        visited[slot] = 1;
        ++visited_count;
        const auto estimate = filter_estimate(query, query_norm, codes.At(slot));
        const Candidate next{slot, estimate.distance};
        candidates.push(next);
        best.push(next);
        if (best.size() > ef) {
            best.pop();
        }
    };

    visit(0);
    const uint64_t last = codes.Size() - 1;
    if (last != 0) {
        visit(last);
    }
    constexpr uint64_t k_extra_entry_points = 6;
    for (uint64_t i = 1; i <= k_extra_entry_points; ++i) {
        const uint64_t entry = i * last / (k_extra_entry_points + 1);
        if (entry != 0 and entry != last) {
            visit(entry);
        }
    }

    while (not candidates.empty()) {
        const Candidate current = candidates.top();
        candidates.pop();
        if (best.size() == ef and better(best.top(), current)) {
            break;
        }
        if (codes.Size() > 1) {
            visit((current.id + codes.Size() - 1) % codes.Size());
            visit((current.id + 1) % codes.Size());
        }
        for (uint64_t edge = graph.offsets[current.id]; edge < graph.offsets[current.id + 1];
             ++edge) {
            visit(graph.neighbors[edge]);
        }
    }

    const uint64_t reorder_count = best.size();
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> reordered(&better);
    while (not best.empty()) {
        const uint64_t slot = best.top().id;
        best.pop();
        const auto code = codes.At(slot);
        const auto estimate = filter_estimate(query, query_norm, code);
        const Candidate next{slot, full_distance(query, query_norm, code, estimate.centered_ip)};
        if (reordered.size() < k) {
            reordered.push(next);
        } else if (better(next, reordered.top())) {
            reordered.pop();
            reordered.push(next);
        }
    }
    std::vector<Candidate> result(reordered.size());
    for (uint64_t i = result.size(); i > 0; --i) {
        result[i - 1] = reordered.top();
        reordered.pop();
    }
    return {std::move(result), visited_count, reorder_count};
}

std::vector<std::vector<uint64_t>>
expand_graph(const GraphTopology& graph) {
    std::vector<std::vector<uint64_t>> result(graph.Size());
    for (uint64_t slot = 0; slot < graph.Size(); ++slot) {
        result[slot].assign(
            graph.neighbors.begin() + static_cast<int64_t>(graph.offsets[slot]),
            graph.neighbors.begin() + static_cast<int64_t>(graph.offsets[slot + 1]));
    }
    return result;
}

GraphTopology
compact_graph(const std::vector<std::vector<uint64_t>>& adjacency) {
    GraphTopology result;
    result.offsets.reserve(adjacency.size() + 1);
    result.offsets.push_back(0);
    for (const auto& neighbors : adjacency) {
        result.neighbors.insert(result.neighbors.end(), neighbors.begin(), neighbors.end());
        result.offsets.push_back(result.neighbors.size());
    }
    return result;
}

class MutableGraphState {
public:
    MutableGraphState(Model input_model,
                      EncodedRecords input_codes,
                      const GraphTopology& input_graph,
                      std::vector<int64_t> input_ids,
                      uint64_t input_max_degree,
                      uint64_t input_ef_search)
        : model_(std::move(input_model)),
          codes_(std::move(input_codes)),
          ids_(std::move(input_ids)),
          adjacency_(expand_graph(input_graph)),
          max_degree_(input_max_degree),
          ef_search_(input_ef_search) {
        require(max_degree_ >= 2 and max_degree_ <= 64 and ef_search_ >= max_degree_,
                "invalid mutable graph options");
        require(ids_.size() == codes_.Size() and adjacency_.size() == codes_.Size(),
                "mutable graph storage disagrees");
        for (uint64_t slot = 0; slot < ids_.size(); ++slot) {
            require(slots_.emplace(ids_[slot], slot).second, "duplicate mutable graph ID");
        }
        Validate();
    }

    [[nodiscard]] uint64_t
    Size() const {
        return codes_.Size();
    }

    [[nodiscard]] bool
    Add(int64_t id, const float* vector) {
        require(vector != nullptr, "null mutable graph vector");
        if (slots_.count(id) != 0) {
            return false;
        }
        float query_norm = 0.0F;
        const auto query = normalize(model_, vector, query_norm);
        const auto neighbors =
            nearest(query, query_norm, std::numeric_limits<uint64_t>::max(), max_degree_);
        const uint64_t slot = Size();
        codes_.Append(encode(model_, vector));
        ids_.push_back(id);
        adjacency_.push_back(neighbors);
        slots_.emplace(id, slot);
        for (uint64_t neighbor : neighbors) {
            link(neighbor, slot);
        }
        return true;
    }

    [[nodiscard]] bool
    Update(int64_t id, const float* vector) {
        require(vector != nullptr, "null mutable graph vector");
        const auto found = slots_.find(id);
        if (found == slots_.end()) {
            return false;
        }
        const uint64_t slot = found->second;
        float query_norm = 0.0F;
        const auto query = normalize(model_, vector, query_norm);
        const auto neighbors = nearest(query, query_norm, slot, max_degree_);
        for (auto& reverse : adjacency_) {
            reverse.erase(std::remove(reverse.begin(), reverse.end(), slot), reverse.end());
        }
        codes_.Replace(slot, encode(model_, vector));
        adjacency_[slot] = neighbors;
        for (uint64_t neighbor : neighbors) {
            link(neighbor, slot);
        }
        return true;
    }

    [[nodiscard]] bool
    Remove(int64_t id) {
        const auto found = slots_.find(id);
        if (found == slots_.end()) {
            return false;
        }
        const uint64_t slot = found->second;
        const uint64_t last = Size() - 1;
        for (auto& neighbors : adjacency_) {
            neighbors.erase(std::remove(neighbors.begin(), neighbors.end(), slot), neighbors.end());
            for (uint64_t& neighbor : neighbors) {
                if (neighbor == last) {
                    neighbor = slot;
                }
            }
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
        if (slot != last) {
            ids_[slot] = ids_[last];
            adjacency_[slot] = std::move(adjacency_[last]);
            adjacency_[slot].erase(
                std::remove(adjacency_[slot].begin(), adjacency_[slot].end(), slot),
                adjacency_[slot].end());
            slots_.at(ids_[slot]) = slot;
        }
        slots_.erase(found);
        ids_.pop_back();
        adjacency_.pop_back();
        codes_.RemoveSwap(slot);
        return true;
    }

    [[nodiscard]] GraphSearchResult
    Search(const float* query, uint64_t k) const {
        require(query != nullptr, "null mutable graph query");
        float query_norm = 0.0F;
        const auto normalized = normalize(model_, query, query_norm);
        return graph_search(
            normalized, query_norm, codes_, compact_graph(adjacency_), k, ef_search_);
    }

    void
    Validate() const {
        require(ids_.size() == Size() and adjacency_.size() == Size() and slots_.size() == Size(),
                "mutable graph size mismatch");
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            const auto found = slots_.find(ids_[slot]);
            require(found != slots_.end() and found->second == slot,
                    "mutable graph ID map mismatch");
            require(adjacency_[slot].size() <= max_degree_, "mutable graph degree exceeds limit");
            for (uint64_t i = 0; i < adjacency_[slot].size(); ++i) {
                const uint64_t neighbor = adjacency_[slot][i];
                require(neighbor < Size() and neighbor != slot, "invalid mutable graph neighbor");
                require(std::find(adjacency_[slot].begin(),
                                  adjacency_[slot].begin() + static_cast<int64_t>(i),
                                  neighbor) == adjacency_[slot].begin() + static_cast<int64_t>(i),
                        "duplicate mutable graph neighbor");
            }
        }
    }

    [[nodiscard]] const Model&
    GetModel() const {
        return model_;
    }

    [[nodiscard]] const EncodedRecords&
    GetCodes() const {
        return codes_;
    }

    [[nodiscard]] GraphTopology
    GetGraph() const {
        return compact_graph(adjacency_);
    }

    [[nodiscard]] const std::vector<int64_t>&
    GetIds() const {
        return ids_;
    }

    [[nodiscard]] int64_t
    IdAt(uint64_t slot) const {
        require(slot < ids_.size(), "mutable graph ID outside storage");
        return ids_[slot];
    }

    [[nodiscard]] uint64_t
    GetMaxDegree() const {
        return max_degree_;
    }

    [[nodiscard]] uint64_t
    GetEfSearch() const {
        return ef_search_;
    }

private:
    [[nodiscard]] std::vector<float>
    decode_query(uint64_t slot) const {
        const auto code = codes_.At(slot);
        const uint64_t plane_bytes = (model_.dim + 7) / 8;
        std::vector<float> result(model_.dim);
        for (uint64_t d = 0; d < model_.dim; ++d) {
            const uint32_t high = read_plane_code(code.filter, plane_bytes, d, K_FILTER_BITS, true);
            const uint32_t low =
                read_plane_code(code.supplement, plane_bytes, d, K_SUPPLEMENT_BITS, false);
            result[d] = (static_cast<float>((high << K_SUPPLEMENT_BITS) + low) - 127.5F) /
                        code.metadata.code_norm;
        }
        return result;
    }

    [[nodiscard]] float
    pair_distance(uint64_t source, uint64_t target) const {
        const auto query = decode_query(source);
        const auto code = codes_.At(target);
        const auto coarse = filter_estimate(query, codes_.At(source).metadata.norm, code);
        return full_distance(query, codes_.At(source).metadata.norm, code, coarse.centered_ip);
    }

    [[nodiscard]] std::vector<uint64_t>
    nearest(const std::vector<float>& query,
            float query_norm,
            uint64_t excluded,
            uint64_t count) const {
        std::priority_queue<Candidate, std::vector<Candidate>, decltype(&better)> heap(&better);
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            if (slot == excluded) {
                continue;
            }
            const auto code = codes_.At(slot);
            const auto coarse = filter_estimate(query, query_norm, code);
            const Candidate candidate{slot,
                                      full_distance(query, query_norm, code, coarse.centered_ip)};
            if (heap.size() < count) {
                heap.push(candidate);
            } else if (better(candidate, heap.top())) {
                heap.pop();
                heap.push(candidate);
            }
        }
        std::vector<uint64_t> result(heap.size());
        for (uint64_t i = result.size(); i > 0; --i) {
            result[i - 1] = heap.top().id;
            heap.pop();
        }
        return result;
    }

    void
    link(uint64_t source, uint64_t target) {
        auto& neighbors = adjacency_[source];
        if (source == target or
            std::find(neighbors.begin(), neighbors.end(), target) != neighbors.end()) {
            return;
        }
        neighbors.push_back(target);
        if (neighbors.size() > max_degree_) {
            std::vector<Candidate> ranked;
            ranked.reserve(neighbors.size());
            for (uint64_t neighbor : neighbors) {
                ranked.push_back({neighbor, pair_distance(source, neighbor)});
            }
            std::sort(ranked.begin(), ranked.end(), better);
            neighbors.resize(max_degree_);
            for (uint64_t i = 0; i < max_degree_; ++i) {
                neighbors[i] = ranked[i].id;
            }
        }
    }

    Model model_;
    EncodedRecords codes_;
    std::vector<int64_t> ids_;
    std::unordered_map<int64_t, uint64_t> slots_;
    std::vector<std::vector<uint64_t>> adjacency_;
    uint64_t max_degree_;
    uint64_t ef_search_;
};

uint64_t
result_checksum(const std::vector<Candidate>& neighbors,
                const MutableGraphState& state,
                uint64_t checksum) {
    constexpr uint64_t prime = 1099511628211ULL;
    for (const auto& neighbor : neighbors) {
        checksum ^= static_cast<uint64_t>(state.IdAt(neighbor.id));
        checksum *= prime;
        uint32_t distance_bits = 0;
        std::memcpy(&distance_bits, &neighbor.distance, sizeof(distance_bits));
        checksum ^= distance_bits;
        checksum *= prime;
    }
    return checksum;
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
write_i64(std::ostream& output, int64_t value) {
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u64(output, bits);
}

int64_t
read_i64(std::istream& input) {
    const uint64_t bits = read_u64(input);
    int64_t value = 0;
    std::memcpy(&value, &bits, sizeof(value));
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
write_bytes(std::ostream& output, const uint8_t* bytes, uint64_t size) {
    if (size > 0) {
        output.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    }
    require(static_cast<bool>(output), "snapshot write failed");
}

void
write_bytes(std::ostream& output, const std::vector<uint8_t>& bytes) {
    write_bytes(output, bytes.data(), bytes.size());
}

void
read_bytes(std::istream& input, uint8_t* bytes, uint64_t size) {
    if (size > 0) {
        input.read(reinterpret_cast<char*>(bytes), static_cast<std::streamsize>(size));
    }
    require(static_cast<bool>(input), "truncated snapshot");
}

void
read_bytes(std::istream& input, std::vector<uint8_t>& bytes) {
    read_bytes(input, bytes.data(), bytes.size());
}

void
write_snapshot_payload(std::ostream& output,
                       uint64_t version,
                       const Model& model,
                       const EncodedRecords& codes) {
    constexpr char magic[] = "VSLRBQ01";
    require((version == 1 or version == 2 or version == 3) and model.dim == codes.dim,
            "snapshot model and records disagree");
    output.write(magic, 8);
    write_u64(output, version);
    write_u64(output, model.dim);
    write_u64(output, codes.Size());
    write_u64(output, model.centroid.size());
    write_u64(output, model.flips.size());
    for (float value : model.centroid) {
        write_float(output, value);
    }
    write_bytes(output, model.flips);
    for (uint64_t id = 0; id < codes.Size(); ++id) {
        const auto code = codes.At(id);
        write_float(output, code.metadata.norm);
        write_float(output, code.metadata.code_norm);
        write_float(output, code.metadata.error);
        write_float(output, code.metadata.filter_norm);
        write_float(output, code.metadata.filter_error);
        write_float(output, code.metadata.lower_bound_error);
        write_bytes(output, code.filter, codes.FilterBytes());
        write_bytes(output, code.supplement, codes.SupplementBytes());
    }
}

void
save_snapshot(std::ostream& output, const Model& model, const EncodedRecords& codes) {
    write_snapshot_payload(output, 1, model, codes);
}

struct EncodedSnapshot {
    Model model;
    EncodedRecords codes;
};

EncodedSnapshot
load_snapshot_payload(std::istream& input, uint64_t expected_version) {
    char magic[8]{};
    input.read(magic, sizeof(magic));
    require(input and std::memcmp(magic, "VSLRBQ01", 8) == 0, "invalid snapshot magic");
    require(read_u64(input) == expected_version, "unsupported snapshot version");
    const uint64_t dim = read_u64(input);
    const uint64_t count = read_u64(input);
    const uint64_t centroid_size = read_u64(input);
    const uint64_t flips_size = read_u64(input);
    require(dim > 0 and dim <= (1U << 20U) and centroid_size == dim, "invalid snapshot model");
    const uint64_t plane_bytes = (dim + 7) / 8;
    require(flips_size == K_ROUNDS * plane_bytes and count <= 1000000, "invalid snapshot layout");
    Model model{dim, std::vector<float>(dim), std::vector<uint8_t>(flips_size)};
    for (float& value : model.centroid) {
        value = read_float(input);
    }
    read_bytes(input, model.flips);
    EncodedRecords codes(dim);
    codes.Resize(count);
    for (uint64_t id = 0; id < count; ++id) {
        auto& metadata = codes.metadata[id];
        metadata.norm = read_float(input);
        metadata.code_norm = read_float(input);
        metadata.error = read_float(input);
        metadata.filter_norm = read_float(input);
        metadata.filter_error = read_float(input);
        metadata.lower_bound_error = read_float(input);
        require(metadata.norm > 0.0F and metadata.code_norm > 0.0F and
                    metadata.filter_norm > 0.0F and metadata.filter_error >= 1e-5F and
                    metadata.filter_error <= 1.0F and metadata.lower_bound_error >= 0.0F,
                "invalid snapshot metadata");
        read_bytes(input, codes.filters.data() + id * codes.FilterBytes(), codes.FilterBytes());
        read_bytes(input,
                   codes.supplements.data() + id * codes.SupplementBytes(),
                   codes.SupplementBytes());
    }
    return {std::move(model), std::move(codes)};
}

std::pair<Model, EncodedRecords>
load_snapshot(std::istream& input) {
    auto loaded = load_snapshot_payload(input, 1);
    require(input.peek() == std::char_traits<char>::eof(), "snapshot trailing bytes");
    return {std::move(loaded.model), std::move(loaded.codes)};
}

void
validate_graph_topology(const GraphTopology& graph, uint64_t count) {
    require(graph.offsets.size() == count + 1 and graph.offsets.front() == 0,
            "invalid graph offset layout");
    require(graph.neighbors.size() <= count * 64, "graph edge count exceeds limit");
    for (uint64_t slot = 0; slot < count; ++slot) {
        const uint64_t begin = graph.offsets[slot];
        const uint64_t end = graph.offsets[slot + 1];
        require(begin <= end and end <= graph.neighbors.size() and end - begin <= 64,
                "invalid graph adjacency bounds");
        for (uint64_t edge = begin; edge < end; ++edge) {
            const uint64_t neighbor = graph.neighbors[edge];
            require(neighbor < count and neighbor != slot, "invalid graph neighbor");
            require(std::find(graph.neighbors.begin() + static_cast<int64_t>(begin),
                              graph.neighbors.begin() + static_cast<int64_t>(edge),
                              neighbor) == graph.neighbors.begin() + static_cast<int64_t>(edge),
                    "duplicate graph neighbor");
        }
    }
    require(graph.offsets.back() == graph.neighbors.size(), "graph edge count mismatch");
}

void
save_graph_snapshot(std::ostream& output,
                    const Model& model,
                    const EncodedRecords& codes,
                    const GraphTopology& graph) {
    validate_graph_topology(graph, codes.Size());
    write_snapshot_payload(output, 2, model, codes);
    write_u64(output, graph.offsets.size());
    write_u64(output, graph.neighbors.size());
    for (uint64_t offset : graph.offsets) {
        write_u64(output, offset);
    }
    for (uint64_t neighbor : graph.neighbors) {
        write_u64(output, neighbor);
    }
}

struct GraphSnapshot {
    Model model;
    EncodedRecords codes;
    GraphTopology graph;
};

GraphSnapshot
load_graph_snapshot(std::istream& input) {
    auto loaded = load_snapshot_payload(input, 2);
    const uint64_t offset_count = read_u64(input);
    const uint64_t neighbor_count = read_u64(input);
    require(offset_count == loaded.codes.Size() + 1 and neighbor_count <= loaded.codes.Size() * 64,
            "invalid graph snapshot layout");
    GraphTopology graph{std::vector<uint64_t>(offset_count), std::vector<uint64_t>(neighbor_count)};
    for (uint64_t& offset : graph.offsets) {
        offset = read_u64(input);
    }
    for (uint64_t& neighbor : graph.neighbors) {
        neighbor = read_u64(input);
    }
    validate_graph_topology(graph, loaded.codes.Size());
    require(input.peek() == std::char_traits<char>::eof(), "snapshot trailing bytes");
    return {std::move(loaded.model), std::move(loaded.codes), std::move(graph)};
}

void
save_mutable_snapshot(std::ostream& output, const MutableGraphState& state) {
    state.Validate();
    write_snapshot_payload(output, 3, state.GetModel(), state.GetCodes());
    write_u64(output, state.GetMaxDegree());
    write_u64(output, state.GetEfSearch());
    write_u64(output, state.GetIds().size());
    for (int64_t id : state.GetIds()) {
        write_i64(output, id);
    }
    const auto graph = state.GetGraph();
    write_u64(output, graph.offsets.size());
    write_u64(output, graph.neighbors.size());
    for (uint64_t offset : graph.offsets) {
        write_u64(output, offset);
    }
    for (uint64_t neighbor : graph.neighbors) {
        write_u64(output, neighbor);
    }
}

MutableGraphState
load_mutable_snapshot(std::istream& input) {
    auto loaded = load_snapshot_payload(input, 3);
    const uint64_t max_degree = read_u64(input);
    const uint64_t ef_search = read_u64(input);
    const uint64_t id_count = read_u64(input);
    require(id_count == loaded.codes.Size(), "invalid mutable snapshot ID count");
    std::vector<int64_t> ids(id_count);
    std::unordered_set<int64_t> unique_ids;
    for (int64_t& id : ids) {
        id = read_i64(input);
        require(unique_ids.insert(id).second, "duplicate mutable snapshot ID");
    }
    const uint64_t offset_count = read_u64(input);
    const uint64_t neighbor_count = read_u64(input);
    require(offset_count == loaded.codes.Size() + 1 and neighbor_count <= loaded.codes.Size() * 64,
            "invalid mutable snapshot graph layout");
    GraphTopology graph{std::vector<uint64_t>(offset_count), std::vector<uint64_t>(neighbor_count)};
    for (uint64_t& offset : graph.offsets) {
        offset = read_u64(input);
    }
    for (uint64_t& neighbor : graph.neighbors) {
        neighbor = read_u64(input);
    }
    validate_graph_topology(graph, loaded.codes.Size());
    require(input.peek() == std::char_traits<char>::eof(), "snapshot trailing bytes");
    return {std::move(loaded.model),
            std::move(loaded.codes),
            graph,
            std::move(ids),
            max_degree,
            ef_search};
}

void
high_dim_transform_self_test() {
    for (uint64_t dim : {768ULL, 960ULL}) {
        constexpr uint64_t count = 4;
        std::vector<float> base(count * dim);
        for (uint64_t i = 0; i < base.size(); ++i) {
            base[i] = std::sin(static_cast<float>(i) * 0.017F) +
                      0.25F * std::cos(static_cast<float>(i) * 0.031F);
        }
        const auto model = train(base, count, dim, 71);
        std::vector<float> original(base.begin(), base.begin() + static_cast<int64_t>(dim));
        std::vector<float> transformed = original;
        model.Transform(transformed);
        double original_norm = 0.0;
        double transformed_norm = 0.0;
        for (uint64_t d = 0; d < dim; ++d) {
            original_norm += original[d] * original[d];
            transformed_norm += transformed[d] * transformed[d];
        }
        require(std::fabs(original_norm - transformed_norm) <= 1e-4 * std::max(1.0, original_norm),
                "high-dimensional FHT changed vector norm");
        const auto first = encode(model, base.data());
        const auto second = encode(model, base.data());
        const uint64_t plane_bytes = (dim + 7) / 8;
        require(first.filter == second.filter and first.supplement == second.supplement and
                    first.filter.size() == plane_bytes * K_FILTER_BITS and
                    first.supplement.size() == plane_bytes * K_SUPPLEMENT_BITS,
                "high-dimensional encoding mismatch");
        EncodedRecords codes(dim);
        codes.Append(first);
        std::stringstream snapshot(std::ios::in | std::ios::out | std::ios::binary);
        save_snapshot(snapshot, model, codes);
        snapshot.seekg(0);
        auto [loaded_model, loaded_codes] = load_snapshot(snapshot);
        require(loaded_model.dim == dim and loaded_codes.Size() == 1 and
                    loaded_model.centroid == model.centroid and loaded_model.flips == model.flips,
                "high-dimensional snapshot mismatch");
    }
}

void
self_test() {
    high_dim_transform_self_test();
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
    EncodedRecords codes(dim);
    codes.Reserve(count);
    for (uint64_t id = 0; id < count; ++id) {
        codes.Append(encode(first, base.data() + id * dim));
    }
    require(codes.filters.size() == count * codes.FilterBytes() and
                codes.supplements.size() == count * codes.SupplementBytes(),
            "encoded records are not compact");
    require(codes.At(1).filter == codes.At(0).filter + codes.FilterBytes() and
                codes.At(1).supplement == codes.At(0).supplement + codes.SupplementBytes(),
            "encoded records are not contiguous");
    float query_norm = 0.0F;
    const auto normalized_query = normalize(first, base.data() + dim, query_norm);
    const auto full = top_k(count, 10, [&](uint64_t id) {
        const auto coarse = filter_estimate(normalized_query, query_norm, codes.At(id));
        return full_distance(normalized_query, query_norm, codes.At(id), coarse.centered_ip);
    });
    const auto filtered = filtered_search(normalized_query, query_norm, codes, 10);
    require(filtered.reordered > 0 and filtered.reordered <= count,
            "invalid filtered search count");
    const auto graph = build_graph_topology(base, count, dim, 8, 32);
    const auto graph_result = graph_search(normalized_query, query_norm, codes, graph, 10, 32);
    require(graph_result.neighbors.size() == 10 and graph_result.visited > 0 and
                graph_result.visited <= count and graph_result.reordered >= 10 and
                graph_result.reordered <= 32,
            "invalid graph search result");
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
                loaded_model.flips == first.flips and loaded_codes.Size() == codes.Size(),
            "snapshot model mismatch");
    float loaded_query_norm = 0.0F;
    const auto loaded_query = normalize(loaded_model, base.data() + dim, loaded_query_norm);
    const auto loaded = filtered_search(loaded_query, loaded_query_norm, loaded_codes, 10);
    const auto loaded_graph =
        graph_search(loaded_query, loaded_query_norm, loaded_codes, graph, 10, 32);
    require(loaded_graph.visited == graph_result.visited and
                loaded_graph.reordered == graph_result.reordered and
                std::equal(graph_result.neighbors.begin(),
                           graph_result.neighbors.end(),
                           loaded_graph.neighbors.begin(),
                           [](const auto& left, const auto& right) {
                               return left.id == right.id and left.distance == right.distance;
                           }),
            "snapshot graph search mismatch");
    std::stringstream repeated_snapshot(std::ios::in | std::ios::out | std::ios::binary);
    save_snapshot(repeated_snapshot, loaded_model, loaded_codes);
    require(repeated_snapshot.str() == bytes, "snapshot bytes changed after round-trip");
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
    std::stringstream graph_snapshot(std::ios::in | std::ios::out | std::ios::binary);
    save_graph_snapshot(graph_snapshot, first, codes, graph);
    const std::string graph_bytes = graph_snapshot.str();
    graph_snapshot.seekg(0);
    auto restored = load_graph_snapshot(graph_snapshot);
    require(restored.model.centroid == first.centroid and restored.model.flips == first.flips and
                restored.codes.Size() == codes.Size() and
                restored.graph.offsets == graph.offsets and
                restored.graph.neighbors == graph.neighbors,
            "graph snapshot mismatch");
    const auto restored_graph =
        graph_search(normalized_query, query_norm, restored.codes, restored.graph, 10, 32);
    require(restored_graph.visited == graph_result.visited and
                restored_graph.reordered == graph_result.reordered and
                std::equal(graph_result.neighbors.begin(),
                           graph_result.neighbors.end(),
                           restored_graph.neighbors.begin(),
                           [](const auto& left, const auto& right) {
                               return left.id == right.id and left.distance == right.distance;
                           }),
            "restored graph search mismatch");
    std::stringstream repeated_graph(std::ios::in | std::ios::out | std::ios::binary);
    save_graph_snapshot(repeated_graph, restored.model, restored.codes, restored.graph);
    require(repeated_graph.str() == graph_bytes, "graph snapshot bytes changed after round-trip");

    const uint64_t plane_bytes_for_snapshot = (dim + 7) / 8;
    const uint64_t record_bytes =
        6 * sizeof(float) + plane_bytes_for_snapshot * (K_FILTER_BITS + K_SUPPLEMENT_BITS);
    const uint64_t graph_start = 6 * sizeof(uint64_t) + dim * sizeof(float) +
                                 K_ROUNDS * plane_bytes_for_snapshot + count * record_bytes;
    const uint64_t offsets_start = graph_start + 2 * sizeof(uint64_t);
    const uint64_t neighbors_start = offsets_start + graph.offsets.size() * sizeof(uint64_t);
    require(graph.offsets[1] >= 2, "test graph needs two first-node neighbors");
    auto overwrite_u64 = [](std::string& value, uint64_t offset, uint64_t replacement_value) {
        require(offset <= value.size() and value.size() - offset >= sizeof(uint64_t),
                "test mutation outside snapshot");
        for (uint64_t i = 0; i < sizeof(uint64_t); ++i) {
            value[offset + i] = static_cast<char>((replacement_value >> (8U * i)) & 0xffU);
        }
    };
    std::vector<std::string> invalid_graphs{graph_bytes.substr(0, graph_bytes.size() - 1),
                                            graph_bytes + std::string(1, '\0')};
    auto wrong_offsets = graph_bytes;
    overwrite_u64(wrong_offsets, graph_start, count);
    invalid_graphs.push_back(std::move(wrong_offsets));
    auto too_many_neighbors = graph_bytes;
    overwrite_u64(too_many_neighbors, graph_start + sizeof(uint64_t), count * 64 + 1);
    invalid_graphs.push_back(std::move(too_many_neighbors));
    auto nonzero_first_offset = graph_bytes;
    overwrite_u64(nonzero_first_offset, offsets_start, 1);
    invalid_graphs.push_back(std::move(nonzero_first_offset));
    auto decreasing_offsets = graph_bytes;
    overwrite_u64(decreasing_offsets, offsets_start + sizeof(uint64_t), graph.offsets[2] + 1);
    invalid_graphs.push_back(std::move(decreasing_offsets));
    auto mismatched_last_offset = graph_bytes;
    overwrite_u64(mismatched_last_offset,
                  offsets_start + count * sizeof(uint64_t),
                  graph.neighbors.size() + 1);
    invalid_graphs.push_back(std::move(mismatched_last_offset));
    auto out_of_range_neighbor = graph_bytes;
    overwrite_u64(out_of_range_neighbor, neighbors_start, count);
    invalid_graphs.push_back(std::move(out_of_range_neighbor));
    auto self_loop = graph_bytes;
    overwrite_u64(self_loop, neighbors_start, 0);
    invalid_graphs.push_back(std::move(self_loop));
    auto duplicate_neighbor = graph_bytes;
    overwrite_u64(duplicate_neighbor, neighbors_start + sizeof(uint64_t), graph.neighbors[0]);
    invalid_graphs.push_back(std::move(duplicate_neighbor));
    for (const std::string& invalid : invalid_graphs) {
        std::stringstream damaged(invalid, std::ios::in | std::ios::binary);
        try {
            static_cast<void>(load_graph_snapshot(damaged));
            throw std::runtime_error("damaged graph snapshot was accepted");
        } catch (const std::runtime_error& error) {
            require(std::string(error.what()) != "damaged graph snapshot was accepted",
                    "damaged graph snapshot was accepted");
        }
    }

    std::vector<int64_t> mutable_ids(count);
    std::iota(mutable_ids.begin(), mutable_ids.end(), 1000);
    MutableGraphState mutable_graph(first, codes, graph, mutable_ids, 8, 32);
    const auto fixed_centroid = mutable_graph.GetModel().centroid;
    const auto fixed_flips = mutable_graph.GetModel().flips;
    require(not mutable_graph.Add(mutable_ids.front(), base.data()),
            "duplicate mutable graph Add succeeded");
    require(not mutable_graph.Update(-1, base.data()), "missing mutable graph Update succeeded");
    require(not mutable_graph.Remove(-1), "missing mutable graph Remove succeeded");

    std::vector<float> mutation(dim);
    int64_t next_id = 1000 + static_cast<int64_t>(count);
    for (uint64_t step = 0; step < 180; ++step) {
        for (uint64_t d = 0; d < dim; ++d) {
            mutation[d] = distribution(generator) + static_cast<float>(step % 7) * 0.01F;
        }
        if (step % 3 == 0) {
            const int64_t id = mutable_ids[step % mutable_ids.size()];
            require(mutable_graph.Update(id, mutation.data()), "mutable graph Update failed");
        } else if (step % 3 == 1 and mutable_ids.size() > 32) {
            const uint64_t position = (step * 7) % mutable_ids.size();
            const int64_t id = mutable_ids[position];
            require(mutable_graph.Remove(id), "mutable graph Remove failed");
            mutable_ids.erase(mutable_ids.begin() + static_cast<int64_t>(position));
        } else {
            require(mutable_graph.Add(next_id, mutation.data()), "mutable graph Add failed");
            mutable_ids.push_back(next_id);
            ++next_id;
        }
        mutable_graph.Validate();
        require(mutable_graph.GetModel().centroid == fixed_centroid and
                    mutable_graph.GetModel().flips == fixed_flips,
                "mutable graph retrained the fixed RaBitQ model");
    }
    require(mutable_graph.Add(-17, mutation.data()), "negative mutable graph ID Add failed");
    mutable_ids.push_back(-17);
    require(mutable_graph.GetIds().size() == mutable_ids.size(),
            "mutable graph active ID count mismatch");
    const auto mutable_result = mutable_graph.Search(base.data() + 2 * dim, 10);
    require(mutable_result.neighbors.size() == 10 and mutable_result.visited > 0,
            "mutable graph search failed after CRUD");

    std::stringstream mutable_snapshot(std::ios::in | std::ios::out | std::ios::binary);
    save_mutable_snapshot(mutable_snapshot, mutable_graph);
    const std::string mutable_bytes = mutable_snapshot.str();
    mutable_snapshot.seekg(0);
    auto restored_mutable = load_mutable_snapshot(mutable_snapshot);
    restored_mutable.Validate();
    require(restored_mutable.GetIds() == mutable_graph.GetIds() and
                restored_mutable.GetGraph().offsets == mutable_graph.GetGraph().offsets and
                restored_mutable.GetGraph().neighbors == mutable_graph.GetGraph().neighbors and
                restored_mutable.GetModel().centroid == fixed_centroid and
                restored_mutable.GetModel().flips == fixed_flips,
            "mutable graph snapshot mismatch");
    const auto restored_mutable_result = restored_mutable.Search(base.data() + 2 * dim, 10);
    require(restored_mutable_result.visited == mutable_result.visited and
                restored_mutable_result.reordered == mutable_result.reordered and
                std::equal(mutable_result.neighbors.begin(),
                           mutable_result.neighbors.end(),
                           restored_mutable_result.neighbors.begin(),
                           [](const auto& left, const auto& right) {
                               return left.id == right.id and left.distance == right.distance;
                           }),
            "mutable graph search changed after restore");
    std::stringstream repeated_mutable(std::ios::in | std::ios::out | std::ios::binary);
    save_mutable_snapshot(repeated_mutable, restored_mutable);
    require(repeated_mutable.str() == mutable_bytes,
            "mutable graph snapshot bytes changed after round-trip");
    const uint64_t mutable_payload_end = 6 * sizeof(uint64_t) + dim * sizeof(float) +
                                         K_ROUNDS * plane_bytes_for_snapshot +
                                         mutable_graph.Size() * record_bytes;
    auto wrong_id_count = mutable_bytes;
    overwrite_u64(
        wrong_id_count, mutable_payload_end + 2 * sizeof(uint64_t), mutable_graph.Size() + 1);
    auto duplicate_id = mutable_bytes;
    overwrite_u64(duplicate_id,
                  mutable_payload_end + 4 * sizeof(uint64_t),
                  static_cast<uint64_t>(mutable_graph.GetIds().front()));
    for (const std::string& invalid : {mutable_bytes.substr(0, mutable_bytes.size() - 1),
                                       mutable_bytes + std::string(1, '\0'),
                                       wrong_id_count,
                                       duplicate_id}) {
        std::stringstream damaged(invalid, std::ios::in | std::ios::binary);
        try {
            static_cast<void>(load_mutable_snapshot(damaged));
            throw std::runtime_error("damaged mutable snapshot was accepted");
        } catch (const std::runtime_error& error) {
            require(std::string(error.what()) != "damaged mutable snapshot was accepted",
                    "damaged mutable snapshot was accepted");
        }
    }
}

uint64_t
read_dimension(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (not input) {
        throw std::runtime_error("cannot open " + path.string());
    }
    int32_t dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    require(input and dim > 0, "invalid record dimension");
    return static_cast<uint64_t>(dim);
}

uint64_t
parse_positive(const char* value) {
    const std::string text(value);
    try {
        size_t consumed = 0;
        const uint64_t parsed = std::stoull(text, &consumed);
        require(consumed == text.size() and parsed > 0, "invalid positive integer");
        return parsed;
    } catch (const std::exception&) {
        throw std::runtime_error("invalid positive integer");
    }
}

void
run(const std::filesystem::path& root,
    uint64_t max_degree,
    uint64_t ef_search,
    const std::filesystem::path& snapshot_path = {}) {
    const uint64_t dim = read_dimension(root / "base.fvecs");
    constexpr uint64_t k = 10;
    const auto base = read_records<float>(root / "base.fvecs", dim);
    const auto queries = read_records<float>(root / "queries.fvecs", dim);
    const auto truth = read_records<int32_t>(root / "groundtruth.ivecs", k);
    require(queries.count == truth.count and base.count >= k and
                base.count <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()),
            "inconsistent SIFT dataset");
    const auto build_start = Clock::now();
    const auto model = train(base.values, base.count, dim, 47);
    EncodedRecords codes(dim);
    codes.Reserve(base.count);
    for (uint64_t id = 0; id < base.count; ++id) {
        codes.Append(encode(model, base.values.data() + id * dim));
    }
    const double build_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - build_start).count();
    require(max_degree >= 4 and max_degree <= 64, "max_degree must be in [4, 64]");
    require(ef_search >= k and ef_search <= base.count, "ef_search must be in [10, base_count]");
    const auto graph_build_start = Clock::now();
    const auto graph = build_graph_topology(base.values, base.count, dim, max_degree, ef_search);
    const double graph_build_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - graph_build_start).count();
    if (not snapshot_path.empty()) {
        require(not std::filesystem::exists(snapshot_path), "snapshot output already exists");
        std::ofstream output(snapshot_path, std::ios::binary);
        require(static_cast<bool>(output), "cannot create graph snapshot");
        save_graph_snapshot(output, model, codes, graph);
    }
    uint64_t full_hits = 0;
    uint64_t filtered_hits = 0;
    uint64_t graph_hits = 0;
    uint64_t agreement = 0;
    uint64_t graph_agreement = 0;
    uint64_t reordered = 0;
    uint64_t graph_visited = 0;
    uint64_t graph_reordered = 0;
    std::vector<double> search_us;
    std::vector<double> graph_search_us;
    for (uint64_t q = 0; q < queries.count; ++q) {
        float query_norm = 0.0F;
        const auto query = normalize(model, queries.values.data() + q * dim, query_norm);
        const auto full = top_k(base.count, k, [&](uint64_t id) {
            const auto coarse = filter_estimate(query, query_norm, codes.At(id));
            return full_distance(query, query_norm, codes.At(id), coarse.centered_ip);
        });
        const auto start = Clock::now();
        const auto filtered = filtered_search(query, query_norm, codes, k);
        search_us.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        const auto graph_start = Clock::now();
        const auto graph_result = graph_search(query, query_norm, codes, graph, k, ef_search);
        graph_search_us.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - graph_start).count());
        const int32_t* expected = truth.values.data() + q * k;
        for (uint64_t i = 0; i < k; ++i) {
            require(expected[i] >= 0 and static_cast<uint64_t>(expected[i]) < base.count,
                    "ground-truth ID outside base");
        }
        full_hits += hits(full, expected, k);
        filtered_hits += hits(filtered.neighbors, expected, k);
        graph_hits += hits(graph_result.neighbors, expected, k);
        reordered += filtered.reordered;
        graph_visited += graph_result.visited;
        graph_reordered += graph_result.reordered;
        for (uint64_t i = 0; i < k; ++i) {
            agreement += static_cast<uint64_t>(full[i].id == filtered.neighbors[i].id);
            graph_agreement += static_cast<uint64_t>(full[i].id == graph_result.neighbors[i].id);
        }
    }
    std::sort(search_us.begin(), search_us.end());
    std::sort(graph_search_us.begin(), graph_search_us.end());
    const uint64_t opportunities = queries.count * k;
    const uint64_t plane_bytes = (dim + 7) / 8;
    std::cout << "base_count,query_count,dim,max_degree,ef_search,build_encode_ms,graph_build_ms,"
                 "full_recall_at_10,"
                 "filtered_recall_at_10,graph_recall_at_10,filtered_full_agreement,"
                 "graph_full_agreement,mean_reordered,reorder_ratio,search_p50_us,"
                 "mean_graph_visited,mean_graph_reordered,graph_search_p50_us,filter_bytes,"
                 "supplement_bytes,metadata_bytes,graph_bytes\n";
    std::cout << std::fixed << std::setprecision(6) << base.count << ',' << queries.count << ','
              << dim << ',' << max_degree << ',' << ef_search << ',' << build_ms << ','
              << graph_build_ms << ','
              << static_cast<double>(full_hits) / static_cast<double>(opportunities) << ','
              << static_cast<double>(filtered_hits) / static_cast<double>(opportunities) << ','
              << static_cast<double>(graph_hits) / static_cast<double>(opportunities) << ','
              << static_cast<double>(agreement) / static_cast<double>(opportunities) << ','
              << static_cast<double>(graph_agreement) / static_cast<double>(opportunities) << ','
              << static_cast<double>(reordered) / static_cast<double>(queries.count) << ','
              << static_cast<double>(reordered) / static_cast<double>(queries.count * base.count)
              << ',' << search_us[search_us.size() / 2] << ','
              << static_cast<double>(graph_visited) / static_cast<double>(queries.count) << ','
              << static_cast<double>(graph_reordered) / static_cast<double>(queries.count) << ','
              << graph_search_us[graph_search_us.size() / 2] << ','
              << base.count * plane_bytes * K_FILTER_BITS << ','
              << base.count * plane_bytes * K_SUPPLEMENT_BITS << ','
              << base.count * 6 * sizeof(float) << ','
              << graph.offsets.size() * sizeof(uint64_t) + graph.neighbors.size() * sizeof(uint64_t)
              << '\n';
}

void
run_loaded(const std::filesystem::path& root,
           const std::filesystem::path& snapshot_path,
           uint64_t ef_search) {
    std::ifstream input(snapshot_path, std::ios::binary);
    require(static_cast<bool>(input), "cannot open graph snapshot");
    const auto load_start = Clock::now();
    auto snapshot = load_graph_snapshot(input);
    const double load_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - load_start).count();
    constexpr uint64_t k = 10;
    require(ef_search >= k and ef_search <= snapshot.codes.Size(),
            "ef_search must be in [10, base_count]");
    const auto queries = read_records<float>(root / "queries.fvecs", snapshot.model.dim);
    const auto truth = read_records<int32_t>(root / "groundtruth.ivecs", k);
    require(queries.count == truth.count and snapshot.codes.Size() >= k,
            "inconsistent graph dataset");
    uint64_t graph_hits = 0;
    uint64_t graph_visited = 0;
    uint64_t graph_reordered = 0;
    std::vector<double> graph_search_us;
    for (uint64_t q = 0; q < queries.count; ++q) {
        float query_norm = 0.0F;
        const auto query =
            normalize(snapshot.model, queries.values.data() + q * snapshot.model.dim, query_norm);
        const auto start = Clock::now();
        const auto found =
            graph_search(query, query_norm, snapshot.codes, snapshot.graph, k, ef_search);
        graph_search_us.push_back(
            std::chrono::duration<double, std::micro>(Clock::now() - start).count());
        const int32_t* expected = truth.values.data() + q * k;
        for (uint64_t i = 0; i < k; ++i) {
            require(expected[i] >= 0 and static_cast<uint64_t>(expected[i]) < snapshot.codes.Size(),
                    "ground-truth ID outside base");
        }
        graph_hits += hits(found.neighbors, expected, k);
        graph_visited += found.visited;
        graph_reordered += found.reordered;
    }
    std::sort(graph_search_us.begin(), graph_search_us.end());
    const uint64_t opportunities = queries.count * k;
    std::cout << "base_count,query_count,dim,ef_search,load_ms,graph_recall_at_10,"
                 "mean_graph_visited,mean_graph_reordered,graph_search_p50_us,snapshot_bytes\n";
    std::cout << std::fixed << std::setprecision(6) << snapshot.codes.Size() << ',' << queries.count
              << ',' << snapshot.model.dim << ',' << ef_search << ',' << load_ms << ','
              << static_cast<double>(graph_hits) / static_cast<double>(opportunities) << ','
              << static_cast<double>(graph_visited) / static_cast<double>(queries.count) << ','
              << static_cast<double>(graph_reordered) / static_cast<double>(queries.count) << ','
              << graph_search_us[graph_search_us.size() / 2] << ','
              << std::filesystem::file_size(snapshot_path) << '\n';
}

void
run_crud(const std::filesystem::path& root,
         const std::filesystem::path& snapshot_path,
         uint64_t rounds,
         uint64_t crud_ops,
         uint64_t query_count,
         uint64_t max_degree,
         uint64_t ef_search) {
    const uint64_t dim = read_dimension(root / "base.fvecs");
    auto base = read_records<float>(root / "base.fvecs", dim);
    require(base.count >= 10 and query_count <= base.count and max_degree >= 4 and
                max_degree <= 64 and ef_search >= 10 and ef_search <= base.count,
            "invalid CRUD stability configuration");
    if (not snapshot_path.parent_path().empty()) {
        require(std::filesystem::exists(snapshot_path.parent_path()),
                "snapshot directory does not exist");
    }

    const auto encode_start = Clock::now();
    auto model = train(base.values, base.count, dim, 47);
    EncodedRecords codes(dim);
    codes.Reserve(base.count);
    for (uint64_t slot = 0; slot < base.count; ++slot) {
        codes.Append(encode(model, base.values.data() + slot * dim));
    }
    const double build_encode_ms = milliseconds(encode_start, Clock::now());

    const auto graph_start = Clock::now();
    const auto initial_graph =
        build_graph_topology(base.values, base.count, dim, max_degree, ef_search);
    const double graph_build_ms = milliseconds(graph_start, Clock::now());
    std::vector<int64_t> ids(base.count);
    std::iota(ids.begin(), ids.end(), 0);
    MutableGraphState state(
        std::move(model), std::move(codes), initial_graph, std::move(ids), max_degree, ef_search);

    std::cout
        << "round,count,dim,crud_ops,queries,max_degree,ef_search,build_encode_ms,graph_build_ms,"
           "update_p50_us,update_p99_us,remove_p50_us,remove_p99_us,add_p50_us,add_p99_us,"
           "compact_ms,search_p50_us,search_p99_us,full_self_top1_recall,"
           "graph_self_top1_recall,graph_full_top1_agreement,graph_full_positional_agreement,"
           "mean_visited,mean_reordered,save_ms,load_ms,snapshot_bytes,state_rss_kib,"
           "roundtrip_rss_kib,peak_rss_kib,result_checksum\n";

    constexpr uint64_t k = 10;
    for (uint64_t round = 0; round < rounds; ++round) {
        std::vector<double> update_us;
        std::vector<double> remove_us;
        std::vector<double> add_us;
        update_us.reserve(crud_ops);
        remove_us.reserve(crud_ops);
        add_us.reserve(crud_ops);
        for (uint64_t operation = 0; operation < crud_ops; ++operation) {
            const uint64_t id = (round * 65537ULL + operation * 8191ULL) % base.count;
            float* vector = base.values.data() + id * dim;
            for (uint64_t d = 0; d < dim; ++d) {
                const auto delta_index =
                    static_cast<int64_t>((round * 13 + operation * 7 + d * 3) % 17);
                vector[d] += static_cast<float>(delta_index - 8) * 0.125F;
            }

            auto start = Clock::now();
            require(state.Update(static_cast<int64_t>(id), vector), "CRUD Update failed");
            update_us.push_back(microseconds(start, Clock::now()));

            start = Clock::now();
            require(state.Remove(static_cast<int64_t>(id)), "CRUD Remove failed");
            remove_us.push_back(microseconds(start, Clock::now()));

            start = Clock::now();
            require(state.Add(static_cast<int64_t>(id), vector), "CRUD Add failed");
            add_us.push_back(microseconds(start, Clock::now()));
        }
        state.Validate();
        require(state.Size() == base.count, "CRUD changed the active record count");
        std::sort(update_us.begin(), update_us.end());
        std::sort(remove_us.begin(), remove_us.end());
        std::sort(add_us.begin(), add_us.end());

        const auto compact_start = Clock::now();
        const auto topology = state.GetGraph();
        const double compact_ms = milliseconds(compact_start, Clock::now());
        std::vector<double> search_us;
        search_us.reserve(query_count);
        std::vector<std::vector<Candidate>> expected;
        expected.reserve(query_count);
        uint64_t full_self_hits = 0;
        uint64_t graph_self_hits = 0;
        uint64_t top1_agreement = 0;
        uint64_t positional_agreement = 0;
        uint64_t visited = 0;
        uint64_t reordered = 0;
        uint64_t checksum = 1469598103934665603ULL;
        for (uint64_t query_number = 0; query_number < query_count; ++query_number) {
            const uint64_t id = (round * 104729ULL + query_number * 65537ULL) % base.count;
            const float* vector = base.values.data() + id * dim;
            float query_norm = 0.0F;
            const auto query = normalize(state.GetModel(), vector, query_norm);
            const auto full = top_k(state.Size(), k, [&](uint64_t slot) {
                const auto code = state.GetCodes().At(slot);
                const auto coarse = filter_estimate(query, query_norm, code);
                return full_distance(query, query_norm, code, coarse.centered_ip);
            });
            const auto start = Clock::now();
            auto found = graph_search(query, query_norm, state.GetCodes(), topology, k, ef_search);
            search_us.push_back(microseconds(start, Clock::now()));
            require(found.neighbors.size() == k, "CRUD search result size changed");
            full_self_hits += state.IdAt(full.front().id) == static_cast<int64_t>(id) ? 1 : 0;
            graph_self_hits +=
                state.IdAt(found.neighbors.front().id) == static_cast<int64_t>(id) ? 1 : 0;
            top1_agreement +=
                state.IdAt(found.neighbors.front().id) == state.IdAt(full.front().id) ? 1 : 0;
            for (uint64_t position = 0; position < k; ++position) {
                positional_agreement +=
                    state.IdAt(found.neighbors[position].id) == state.IdAt(full[position].id) ? 1
                                                                                              : 0;
            }
            visited += found.visited;
            reordered += found.reordered;
            checksum = result_checksum(found.neighbors, state, checksum);
            expected.push_back(std::move(found.neighbors));
        }
        std::sort(search_us.begin(), search_us.end());

        const auto save_start = Clock::now();
        {
            std::ofstream output(snapshot_path, std::ios::binary | std::ios::trunc);
            require(static_cast<bool>(output), "cannot create mutable CRUD snapshot");
            save_mutable_snapshot(output, state);
            output.close();
            require(static_cast<bool>(output), "cannot close mutable CRUD snapshot");
        }
        const double save_ms = milliseconds(save_start, Clock::now());
        const uint64_t snapshot_bytes = std::filesystem::file_size(snapshot_path);
        const uint64_t state_rss_kib = current_rss_kib();

        const auto load_start = Clock::now();
        std::ifstream input(snapshot_path, std::ios::binary);
        require(static_cast<bool>(input), "cannot open mutable CRUD snapshot");
        auto restored = load_mutable_snapshot(input);
        const double load_ms = milliseconds(load_start, Clock::now());
        const auto restored_topology = restored.GetGraph();
        uint64_t restored_checksum = 1469598103934665603ULL;
        for (uint64_t query_number = 0; query_number < query_count; ++query_number) {
            const uint64_t id = (round * 104729ULL + query_number * 65537ULL) % base.count;
            const float* vector = base.values.data() + id * dim;
            float query_norm = 0.0F;
            const auto query = normalize(restored.GetModel(), vector, query_norm);
            const auto found = graph_search(
                query, query_norm, restored.GetCodes(), restored_topology, k, ef_search);
            require(found.neighbors.size() == expected[query_number].size(),
                    "restored CRUD result size changed");
            require(std::equal(found.neighbors.begin(),
                               found.neighbors.end(),
                               expected[query_number].begin(),
                               [](const auto& left, const auto& right) {
                                   return left.id == right.id and left.distance == right.distance;
                               }),
                    "restored CRUD search result changed");
            restored_checksum = result_checksum(found.neighbors, restored, restored_checksum);
        }
        require(restored_checksum == checksum, "restored CRUD checksum changed");

        const uint64_t opportunities = query_count * k;
        std::cout << std::fixed << std::setprecision(6) << round + 1 << ',' << state.Size() << ','
                  << dim << ',' << crud_ops << ',' << query_count << ',' << max_degree << ','
                  << ef_search << ',' << build_encode_ms << ',' << graph_build_ms << ','
                  << percentile(update_us, 0.50) << ',' << percentile(update_us, 0.99) << ','
                  << percentile(remove_us, 0.50) << ',' << percentile(remove_us, 0.99) << ','
                  << percentile(add_us, 0.50) << ',' << percentile(add_us, 0.99) << ','
                  << compact_ms << ',' << percentile(search_us, 0.50) << ','
                  << percentile(search_us, 0.99) << ','
                  << static_cast<double>(full_self_hits) / static_cast<double>(query_count) << ','
                  << static_cast<double>(graph_self_hits) / static_cast<double>(query_count) << ','
                  << static_cast<double>(top1_agreement) / static_cast<double>(query_count) << ','
                  << static_cast<double>(positional_agreement) / static_cast<double>(opportunities)
                  << ',' << static_cast<double>(visited) / static_cast<double>(query_count) << ','
                  << static_cast<double>(reordered) / static_cast<double>(query_count) << ','
                  << save_ms << ',' << load_ms << ',' << snapshot_bytes << ',' << state_rss_kib
                  << ',' << current_rss_kib() << ',' << peak_rss_kib() << ',' << checksum << '\n';
    }
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
        if (argc == 6 and std::string(argv[1]) == "--save") {
            run(argv[2], parse_positive(argv[4]), parse_positive(argv[5]), argv[3]);
            return 0;
        }
        if (argc == 5 and std::string(argv[1]) == "--load") {
            run_loaded(argv[2], argv[3], parse_positive(argv[4]));
            return 0;
        }
        if (argc == 9 and std::string(argv[1]) == "--crud") {
            run_crud(argv[2],
                     argv[3],
                     parse_positive(argv[4]),
                     parse_positive(argv[5]),
                     parse_positive(argv[6]),
                     parse_positive(argv[7]),
                     parse_positive(argv[8]));
            return 0;
        }
        if (argc == 2 or argc == 4) {
            const uint64_t max_degree = argc == 4 ? parse_positive(argv[2]) : 16;
            const uint64_t ef_search = argc == 4 ? parse_positive(argv[3]) : 128;
            run(argv[1], max_degree, ef_search);
            return 0;
        }
        std::cerr << "usage: lite_rabitq_codec_probe [DATASET_DIR [MAX_DEGREE EF_SEARCH] | "
                     "--save DATASET_DIR SNAPSHOT MAX_DEGREE EF_SEARCH | "
                     "--load DATASET_DIR SNAPSHOT EF_SEARCH | "
                     "--crud DATASET_DIR SNAPSHOT ROUNDS CRUD_OPS QUERIES MAX_DEGREE EF_SEARCH | "
                     "--self-test]\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
