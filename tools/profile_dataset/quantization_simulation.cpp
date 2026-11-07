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
#include <stdexcept>

#include "neighbor_search.h"
#include "statistics.h"
#include "vector_math.h"

namespace vsag::dataset_profile {

namespace {

constexpr double SQ8_TRUNC_RATE = 0.001;
constexpr double SQ4_TRUNC_RATE = 0.05;
/// kScalarQuantizationUpperClamp in scalar_quantization_utils.h.
constexpr float UPPER_CLAMP = 0.999F;
/// Metadata of the uniform quantizers: a uint64 norm of the codes and float sums.
constexpr uint64_t NORM_BYTES = 8;
constexpr uint64_t SUM_BYTES = 4;

uint64_t
align_up(uint64_t value, uint64_t base) {
    return (value + base - 1) / base * base;
}

/// ScalarQuantizer::encode_one_with_scratch: per-dimension range, divided by the range.
uint32_t
per_dimension_code(float value, float lower, float diff, uint32_t levels) {
    float delta = 1.0F;
    if (diff >= std::numeric_limits<float>::epsilon()) {
        delta = 1.0F * (value - lower) / diff;
    }
    return static_cast<uint32_t>(static_cast<float>(levels - 1) * clamp_scalar_delta(delta));
}

/// SQ8UniformQuantizer / SQ4UniformQuantizer::EncodeOneImpl: one range, multiplied by its
/// inverse.
uint32_t
uniform_code(float value, float lower, float inv_diff, uint32_t levels) {
    const float delta = (value - lower) * inv_diff;
    return static_cast<uint32_t>(static_cast<float>(levels - 1) * clamp_scalar_delta(delta));
}

using DistanceFn = std::function<double(uint64_t, uint64_t)>;

/// Ids of the k nearest base rows of every query under distance(query, row), in the total
/// order of TopK (distance, then row id).
std::vector<std::vector<uint64_t>>
top_k_ids(uint64_t base_rows, uint64_t query_rows, uint64_t k, const DistanceFn& distance) {
    std::vector<std::vector<uint64_t>> result(query_rows);
    const auto signed_queries = static_cast<int64_t>(query_rows);
    // clang-format off
#pragma omp parallel for schedule(dynamic, 4) default(none) \
    shared(base_rows, k, distance, result, signed_queries)
    // clang-format on
    for (int64_t q = 0; q < signed_queries; ++q) {
        TopK heap(k);
        for (uint64_t r = 0; r < base_rows; ++r) {
            // Codes that overflow (fp16 above 65504) can give inf - inf; rank NaN last so the
            // order stays total.
            const double value = distance(static_cast<uint64_t>(q), r);
            heap.Push(std::isnan(value) ? std::numeric_limits<double>::infinity() : value, r);
        }
        for (const auto& neighbor : heap.Sorted()) {
            result[q].push_back(neighbor.index);
        }
    }
    return result;
}

/// Distance between fp32 (or decoded) vectors. Rows are pre-normalized for cosine, so
/// 1 - <q, x> ranks like cosine distance.
double
vector_distance(const float* query, const float* row, uint64_t dim, Metric metric) {
    return metric == Metric::L2 ? l2_sqr(query, row, dim) : 1.0 - inner_product(query, row, dim);
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

/// Codes of a uniform quantizer with the metadata VSAG stores next to them.
struct UniformCodes {
    std::vector<uint8_t> codes;
    /// Sum of the encoded (normalized for cosine) values, accumulated in float.
    std::vector<float> sums;
    /// Sum of squared codes.
    std::vector<uint64_t> norms;
};

UniformCodes
encode_uniform(const std::vector<float>& data,
               uint64_t rows,
               uint64_t dim,
               float lower,
               float inv_diff,
               uint32_t levels) {
    UniformCodes result;
    result.codes.resize(rows * dim);
    result.sums.assign(rows, 0.0F);
    result.norms.assign(rows, 0);
    for (uint64_t r = 0; r < rows; ++r) {
        for (uint64_t d = 0; d < dim; ++d) {
            const float value = data[r * dim + d];
            const uint32_t code = uniform_code(value, lower, inv_diff, levels);
            result.codes[r * dim + d] = static_cast<uint8_t>(code);
            result.sums[r] += value;
            result.norms[r] += static_cast<uint64_t>(code) * code;
        }
    }
    return result;
}

/// Mean over rows with a non-zero norm of ||x - decoded(x)|| / ||x||.
double
reconstruction_error(const std::vector<float>& original,
                     const std::vector<float>& decoded,
                     uint64_t rows,
                     uint64_t dim) {
    double total = 0.0;
    uint64_t counted = 0;
    for (uint64_t r = 0; r < rows; ++r) {
        double error = 0.0;
        double norm = 0.0;
        for (uint64_t d = 0; d < dim; ++d) {
            const double value = original[r * dim + d];
            const double difference = value - decoded[r * dim + d];
            error += difference * difference;
            norm += value * value;
        }
        if (norm > 0.0) {
            total += std::sqrt(error / norm);
            ++counted;
        }
    }
    return safe_ratio(total, static_cast<double>(counted));
}

double
recall(const std::vector<std::vector<uint64_t>>& exact,
       const std::vector<std::vector<uint64_t>>& approximate) {
    double sum = 0.0;
    for (uint64_t q = 0; q < exact.size(); ++q) {
        std::vector<uint64_t> truth = exact[q];
        std::vector<uint64_t> found = approximate[q];
        std::sort(truth.begin(), truth.end());
        std::sort(found.begin(), found.end());
        std::vector<uint64_t> common;
        std::set_intersection(
            truth.begin(), truth.end(), found.begin(), found.end(), std::back_inserter(common));
        sum += truth.empty()
                   ? 1.0
                   : static_cast<double>(common.size()) / static_cast<double>(truth.size());
    }
    return sum / static_cast<double>(std::max<uint64_t>(exact.size(), 1));
}

}  // namespace

float
clamp_scalar_delta(float delta) {
    if (!(delta >= 0.0F)) {
        return 0.0F;
    }
    if (delta > UPPER_CLAMP) {
        return 1.0F;
    }
    return delta;
}

uint64_t
quantizer_code_bytes(const std::string& name, uint64_t dim, Metric metric) {
    if (name == "fp16" || name == "bf16") {
        return dim * 2;
    }
    if (name == "sq8" || name == "sq4") {
        return align_up(dim * (name == "sq8" ? 8 : 4), 8) / 8;
    }
    if (name == "sq8_uniform") {
        // SQ8UniformQuantizer: l2 stores the code norm, ip and cosine the value sum.
        const uint64_t metadata = metric == Metric::L2 ? NORM_BYTES : SUM_BYTES;
        return align_up(dim, metadata) + metadata;
    }
    if (name == "sq4_uniform") {
        // SQ4UniformQuantizer: l2 and cosine reserve the code norm, ip and cosine the value sum
        // and the code sum.
        const bool norm = metric != Metric::IP;
        const bool sums = metric != Metric::L2;
        uint64_t align = 1;
        align = norm ? std::max(align, NORM_BYTES) : align;
        align = sums ? std::max(align, SUM_BYTES) : align;
        uint64_t bytes = align_up((dim + 1) / 2, align);
        bytes += norm ? align_up(NORM_BYTES, align) : 0;
        bytes += sums ? 2 * align_up(SUM_BYTES, align) : 0;
        return bytes;
    }
    throw std::invalid_argument("unknown quantizer " + name);
}

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
    const auto exact = top_k_ids(base_rows, query_rows, k, [&](uint64_t q, uint64_t r) {
        return vector_distance(probe.data() + q * dim, train.data() + r * dim, dim, metric);
    });

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

    JsonType schemes = JsonType::object();
    auto record = [&](const std::string& name,
                      double bits_per_dim,
                      bool encodes_query,
                      const std::vector<float>& decoded,
                      const std::vector<std::vector<uint64_t>>& approximate) {
        schemes[name] =
            JsonType{{"bits_per_dim", bits_per_dim},
                     {"code_bytes_per_vector", quantizer_code_bytes(name, dim, metric)},
                     {"encodes_query", encodes_query},
                     {"recall_at_k", json_number(recall(exact, approximate))},
                     {"mean_relative_reconstruction_error",
                      json_number(reconstruction_error(train, decoded, base_rows, dim))}};
    };

    // fp16 / bf16: HalfPrecisionQuantizer encodes base rows and queries.
    std::vector<float> decoded(train.size());
    std::vector<float> decoded_queries(probe.size());
    for (const bool bf16 : {false, true}) {
        for (uint64_t i = 0; i < train.size(); ++i) {
            decoded[i] = bf16 ? round_through_bf16(train[i]) : round_through_fp16(train[i]);
        }
        for (uint64_t i = 0; i < probe.size(); ++i) {
            decoded_queries[i] = bf16 ? round_through_bf16(probe[i]) : round_through_fp16(probe[i]);
        }
        const auto approximate = top_k_ids(base_rows, query_rows, k, [&](uint64_t q, uint64_t r) {
            return vector_distance(
                decoded_queries.data() + q * dim, decoded.data() + r * dim, dim, metric);
        });
        record(bf16 ? "bf16" : "fp16", 16.0, true, decoded, approximate);
    }

    // sq8 / sq4: ScalarQuantizer decodes base rows per dimension; queries stay fp32.
    for (const uint32_t bits : {8U, 4U}) {
        const uint32_t levels = 1U << bits;
        const std::vector<float>& lower = bits == 8 ? sq8_lower : sq4_lower;
        const std::vector<float>& upper = bits == 8 ? sq8_upper : sq4_upper;
        for (uint64_t r = 0; r < base_rows; ++r) {
            for (uint64_t d = 0; d < dim; ++d) {
                const float diff = upper[d] - lower[d];
                const uint32_t code =
                    per_dimension_code(train[r * dim + d], lower[d], diff, levels);
                decoded[r * dim + d] =
                    static_cast<float>(code) / (static_cast<float>(levels) - 1.0F) * diff +
                    lower[d];
            }
        }
        const auto approximate = top_k_ids(base_rows, query_rows, k, [&](uint64_t q, uint64_t r) {
            return vector_distance(probe.data() + q * dim, decoded.data() + r * dim, dim, metric);
        });
        record(bits == 8 ? "sq8" : "sq4", bits, false, decoded, approximate);
    }

    // sq8_uniform / sq4_uniform: one range; distances come from the codes and, for ip and
    // cosine, from the stored sums of the original values (ComputeImpl).
    for (const uint32_t bits : {8U, 4U}) {
        const uint32_t levels = 1U << bits;
        const float lower = bits == 8 ? global_min : sq4u_lower;
        const float diff = (bits == 8 ? global_max : sq4u_upper) - lower;
        const float inv_diff = diff == 0.0F ? 0.0F : 1.0F / diff;
        const double step = diff / (static_cast<double>(levels) - 1.0);
        const auto rate = static_cast<float>(step * step);
        const UniformCodes base_codes =
            encode_uniform(train, base_rows, dim, lower, inv_diff, levels);
        const UniformCodes query_codes =
            encode_uniform(probe, query_rows, dim, lower, inv_diff, levels);
        for (uint64_t i = 0; i < train.size(); ++i) {
            decoded[i] = static_cast<float>(
                base_codes.codes[i] / (static_cast<double>(levels) - 1.0) * diff + lower);
        }
        const auto approximate = top_k_ids(base_rows, query_rows, k, [&](uint64_t q, uint64_t r) {
            const uint8_t* a = query_codes.codes.data() + q * dim;
            const uint8_t* b = base_codes.codes.data() + r * dim;
            uint64_t products = 0;
            for (uint64_t d = 0; d < dim; ++d) {
                products += static_cast<uint64_t>(a[d]) * b[d];
            }
            const auto codes_ip = static_cast<float>(products);
            if (metric == Metric::L2) {
                return static_cast<double>((static_cast<float>(query_codes.norms[q]) +
                                            static_cast<float>(base_codes.norms[r]) -
                                            2.0F * codes_ip) *
                                           rate);
            }
            const float score = lower * (query_codes.sums[q] + base_codes.sums[r]) +
                                rate * codes_ip + lower * lower;
            return static_cast<double>(1.0F - score);
        });
        record(bits == 8 ? "sq8_uniform" : "sq4_uniform", bits, true, decoded, approximate);
    }

    return JsonType{
        {"base_rows", base_rows},
        {"query_rows", query_rows},
        {"k", k},
        {"sq8_bounds", "per-dimension, trunc rate 0.001"},
        {"sq4_bounds", "per-dimension, trunc rate 0.05"},
        {"sq8_uniform_bounds", JsonType::array({global_min, global_max})},
        {"sq4_uniform_bounds", JsonType::array({sq4u_lower, sq4u_upper})},
        {"scoring",
         "As in VSAG: sq8 and sq4 compare fp32 queries with decoded rows; fp16, bf16 and the "
         "uniform quantizers encode queries too; uniform ip and cosine scores add code products "
         "to the stored sums of the original values. code_bytes_per_vector includes the stored "
         "metadata and alignment."},
        {"schemes", schemes}};
}

}  // namespace vsag::dataset_profile
