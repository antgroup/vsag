// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "quantization_simulation.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "neighbor_search.h"
#include "statistics.h"
#include "vector_math.h"

namespace vsag::dataset_profile {

namespace {

constexpr double SQ8_TRUNC_RATE = 0.001;
constexpr double SQ4_TRUNC_RATE = 0.05;

struct Codec {
    std::string name;
    double bits_per_dim{0.0};
    bool encode_queries{false};
    std::function<float(float, uint64_t)> transform;
};

float
scalar_round_trip(float value, float lower, float diff, uint32_t levels) {
    float delta = 1.0F;
    if (diff >= std::numeric_limits<float>::epsilon()) {
        delta = (value - lower) / diff;
    }
    delta = std::min(1.0F, std::max(0.0F, delta));
    const auto code = static_cast<uint32_t>(static_cast<float>(levels - 1) * delta);
    return static_cast<float>(code) / static_cast<float>(levels - 1) * diff + lower;
}

std::vector<std::vector<uint64_t>>
top_k_ids(const std::vector<float>& base,
          uint64_t base_rows,
          const std::vector<float>& queries,
          uint64_t query_rows,
          uint64_t dim,
          Metric metric,
          uint64_t k) {
    std::vector<std::vector<uint64_t>> result(query_rows);
    const auto signed_queries = static_cast<int64_t>(query_rows);
    // clang-format off
#pragma omp parallel for schedule(dynamic, 4) default(none) \
    shared(base, base_rows, queries, dim, metric, k, result, signed_queries)
    // clang-format on
    for (int64_t q = 0; q < signed_queries; ++q) {
        const float* query = queries.data() + static_cast<uint64_t>(q) * dim;
        TopK heap(k);
        for (uint64_t r = 0; r < base_rows; ++r) {
            const float* row = base.data() + r * dim;
            // Rows are pre-normalized for cosine, so 1 - <q, x> ranks like cosine distance.
            const double distance = metric == Metric::L2 ? l2_sqr(query, row, dim)
                                                         : 1.0 - inner_product(query, row, dim);
            heap.Push(distance, r);
        }
        for (const auto& neighbor : heap.Sorted()) {
            result[q].push_back(neighbor.index);
        }
    }
    return result;
}

void
normalize_rows(std::vector<float>& data, uint64_t rows, uint64_t dim) {
    for (uint64_t r = 0; r < rows; ++r) {
        float* row = data.data() + r * dim;
        const double norm = std::sqrt(squared_norm(row, dim));
        if (norm > 0.0) {
            for (uint64_t d = 0; d < dim; ++d) {
                row[d] = static_cast<float>(row[d] / norm);
            }
        }
    }
}

}  // namespace

void
truncated_bounds(const std::vector<float>& data,
                 uint64_t rows,
                 uint64_t dim,
                 double rate,
                 std::vector<float>& lower,
                 std::vector<float>& upper) {
    lower.assign(dim, 0.0F);
    upper.assign(dim, 0.0F);
    if (rows == 0) {
        return;
    }
    auto ignore = static_cast<uint64_t>(static_cast<double>(rows - 1) * rate);
    ignore = std::max<uint64_t>(ignore, 1);
    ignore = std::min(ignore, rows);
    std::vector<float> column(rows);
    for (uint64_t d = 0; d < dim; ++d) {
        for (uint64_t r = 0; r < rows; ++r) {
            column[r] = data[r * dim + d];
        }
        std::nth_element(
            column.begin(), column.begin() + static_cast<int64_t>(ignore - 1), column.end());
        lower[d] = column[ignore - 1];
        std::nth_element(
            column.begin(), column.begin() + static_cast<int64_t>(rows - ignore), column.end());
        upper[d] = column[rows - ignore];
    }
}

JsonType
simulate_quantization(const std::vector<float>& base,
                      uint64_t base_rows,
                      const std::vector<float>& queries,
                      uint64_t query_rows,
                      uint64_t dim,
                      Metric metric,
                      uint64_t k) {
    std::vector<float> train = base;
    std::vector<float> probe = queries;
    if (metric == Metric::COSINE) {
        normalize_rows(train, base_rows, dim);
        normalize_rows(probe, query_rows, dim);
    }
    k = std::min(k, base_rows);
    const auto exact = top_k_ids(train, base_rows, probe, query_rows, dim, metric, k);

    std::vector<float> sq8_lower;
    std::vector<float> sq8_upper;
    truncated_bounds(train, base_rows, dim, SQ8_TRUNC_RATE, sq8_lower, sq8_upper);
    std::vector<float> sq4_lower;
    std::vector<float> sq4_upper;
    truncated_bounds(train, base_rows, dim, SQ4_TRUNC_RATE, sq4_lower, sq4_upper);
    const float global_min = *std::min_element(train.begin(), train.end());
    const float global_max = *std::max_element(train.begin(), train.end());
    float sq4u_lower = *std::max_element(sq4_lower.begin(), sq4_lower.end());
    float sq4u_upper = *std::min_element(sq4_upper.begin(), sq4_upper.end());
    if (sq4u_lower > sq4u_upper) {
        std::swap(sq4u_lower, sq4u_upper);
    }

    std::vector<Codec> codecs;
    codecs.push_back(
        {"fp16", 16.0, false, [](float v, uint64_t) { return round_through_fp16(v); }});
    codecs.push_back(
        {"bf16", 16.0, false, [](float v, uint64_t) { return round_through_bf16(v); }});
    codecs.push_back({"sq8", 8.0, false, [&](float v, uint64_t d) {
                          return scalar_round_trip(
                              v, sq8_lower[d], sq8_upper[d] - sq8_lower[d], 256);
                      }});
    codecs.push_back({"sq4", 4.0, false, [&](float v, uint64_t d) {
                          return scalar_round_trip(
                              v, sq4_lower[d], sq4_upper[d] - sq4_lower[d], 16);
                      }});
    codecs.push_back({"sq8_uniform", 8.0, true, [&](float v, uint64_t) {
                          return scalar_round_trip(v, global_min, global_max - global_min, 256);
                      }});
    codecs.push_back({"sq4_uniform", 4.0, true, [&](float v, uint64_t) {
                          return scalar_round_trip(v, sq4u_lower, sq4u_upper - sq4u_lower, 16);
                      }});

    JsonType schemes = JsonType::object();
    std::vector<float> decoded(train.size());
    std::vector<float> decoded_queries(probe.size());
    for (const auto& codec : codecs) {
        double relative_error = 0.0;
        uint64_t error_rows = 0;
        for (uint64_t r = 0; r < base_rows; ++r) {
            double error = 0.0;
            double norm = 0.0;
            for (uint64_t d = 0; d < dim; ++d) {
                const float original = train[r * dim + d];
                const float value = codec.transform(original, d);
                decoded[r * dim + d] = value;
                error += (static_cast<double>(original) - value) *
                         (static_cast<double>(original) - value);
                norm += static_cast<double>(original) * original;
            }
            if (norm > 0.0) {
                relative_error += std::sqrt(error / norm);
                ++error_rows;
            }
        }
        for (uint64_t i = 0; i < probe.size(); ++i) {
            decoded_queries[i] =
                codec.encode_queries ? codec.transform(probe[i], i % dim) : probe[i];
        }
        const auto approximate =
            top_k_ids(decoded, base_rows, decoded_queries, query_rows, dim, metric, k);
        double recall = 0.0;
        for (uint64_t q = 0; q < query_rows; ++q) {
            std::vector<uint64_t> truth = exact[q];
            std::vector<uint64_t> found = approximate[q];
            std::sort(truth.begin(), truth.end());
            std::sort(found.begin(), found.end());
            std::vector<uint64_t> common;
            std::set_intersection(
                truth.begin(), truth.end(), found.begin(), found.end(), std::back_inserter(common));
            recall += truth.empty()
                          ? 1.0
                          : static_cast<double>(common.size()) / static_cast<double>(truth.size());
        }
        schemes[codec.name] =
            JsonType{{"bits_per_dim", codec.bits_per_dim},
                     {"code_bytes_per_vector",
                      std::ceil(codec.bits_per_dim * static_cast<double>(dim) / 8.0)},
                     {"encodes_query", codec.encode_queries},
                     {"recall_at_k",
                      json_number(recall / static_cast<double>(std::max<uint64_t>(query_rows, 1)))},
                     {"mean_relative_reconstruction_error",
                      json_number(safe_ratio(relative_error, static_cast<double>(error_rows)))}};
    }
    return JsonType{{"base_rows", base_rows},
                    {"query_rows", query_rows},
                    {"k", k},
                    {"sq8_bounds", "per-dimension, trunc rate 0.001"},
                    {"sq4_bounds", "per-dimension, trunc rate 0.05"},
                    {"sq8_uniform_bounds", JsonType::array({global_min, global_max})},
                    {"sq4_uniform_bounds", JsonType::array({sq4u_lower, sq4u_upper})},
                    {"schemes", schemes}};
}

}  // namespace vsag::dataset_profile
