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

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

#include "metric_type.h"
#include "quantization/computer.h"
#include "quantizer.h"
#include "simd/basic_func.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

namespace vsag {
enum class QuantizerType {
    QUANTIZER_TYPE_SQ8 = 0,
    QUANTIZER_TYPE_SQ8_UNIFORM = 1,
    QUANTIZER_TYPE_SQ4 = 2,
    QUANTIZER_TYPE_SQ4_UNIFORM = 3,
    QUANTIZER_TYPE_FP32 = 4,
    QUANTIZER_TYPE_FP16 = 5,
    QUANTIZER_TYPE_BF16 = 6,
    QUANTIZER_TYPE_INT8 = 7,
    QUANTIZER_TYPE_PQ = 8,
    QUANTIZER_TYPE_PQFS = 9,
    QUANTIZER_TYPE_RABITQ = 10,
    QUANTIZER_TYPE_SPARSE = 11,
    QUANTIZER_TYPE_TQ = 12,
};
}  // namespace vsag

using namespace vsag;

template <typename T, typename DataT = float>
void
TestQuantizerAdapterEncodeDecode(Quantizer<T>& quant,
                                 int64_t dim,
                                 int count,
                                 float error = 1e-5,
                                 bool retrain = true,
                                 const std::string& dtype = DATATYPE_FLOAT32) {
    std::vector<DataT> vecs;
    if constexpr (std::is_same<DataT, float>::value == true) {
        vecs = fixtures::generate_vectors(count, dim, true);
    } else if constexpr (std::is_same<DataT, int8_t>::value == true) {
        vecs = fixtures::generate_int8_codes(count, dim, true);
    } else if constexpr (std::is_same<DataT, uint16_t>::value == true) {
        auto [codes, floats] = (dtype == DATATYPE_FLOAT16)
                                   ? fixtures::generate_fp16_codes_with_floats(count, dim, 42)
                                   : fixtures::generate_bf16_codes_with_floats(count, dim, 42);
        vecs = std::move(codes);
    } else {
        static_assert("Unsupported DataT type");
    }
    if (retrain) {
        quant.ReTrain(reinterpret_cast<float*>(vecs.data()), count);
    }
    for (uint64_t i = 0; i < count; ++i) {
        std::vector<uint8_t> codes(quant.GetCodeSize());
        quant.EncodeOne(reinterpret_cast<float*>(vecs.data() + i * dim), codes.data());
        std::vector<DataT> out_vec(dim);
        quant.DecodeOne(codes.data(), reinterpret_cast<float*>(out_vec.data()));
        float sum = 0.0F;
        for (int j = 0; j < dim; ++j) {
            if constexpr (std::is_same<DataT, uint16_t>::value == true) {
                float original = (dtype == DATATYPE_FLOAT16)
                                     ? generic::FP16ToFloat(vecs[i * dim + j])
                                     : generic::BF16ToFloat(vecs[i * dim + j]);
                float decoded = (dtype == DATATYPE_FLOAT16) ? generic::FP16ToFloat(out_vec[j])
                                                            : generic::BF16ToFloat(out_vec[j]);
                sum += std::abs(original - decoded);
            } else {
                sum += std::abs(static_cast<float>(vecs[i * dim + j]) -
                                static_cast<float>(out_vec[j]));
            }
        }
        REQUIRE(sum < error * dim);
    }

    std::vector<uint8_t> codes(quant.GetCodeSize() * count);
    quant.EncodeBatch(reinterpret_cast<float*>(vecs.data()), codes.data(), count);
    std::vector<DataT> out_vec(dim * count);
    quant.DecodeBatch(codes.data(), reinterpret_cast<float*>(out_vec.data()), count);
    for (int64_t i = 0; i < count; ++i) {
        float sum = 0.0F;
        for (int j = 0; j < dim; ++j) {
            if constexpr (std::is_same<DataT, uint16_t>::value == true) {
                float original = (dtype == DATATYPE_FLOAT16)
                                     ? generic::FP16ToFloat(vecs[i * dim + j])
                                     : generic::BF16ToFloat(vecs[i * dim + j]);
                float decoded = (dtype == DATATYPE_FLOAT16)
                                    ? generic::FP16ToFloat(out_vec[i * dim + j])
                                    : generic::BF16ToFloat(out_vec[i * dim + j]);
                sum += std::abs(original - decoded);
            } else {
                sum += std::abs(static_cast<float>(vecs[i * dim + j]) -
                                static_cast<float>(out_vec[i * dim + j]));
            }
        }
        REQUIRE(sum < error * dim);
    }
}

template <typename T, MetricType metric, typename DataT = float>
void
TestQuantizerAdapterComputeCodes(Quantizer<T>& quantizer,
                                 uint64_t dim,
                                 uint32_t count,
                                 float error = 1e-4f,
                                 bool retrain = true,
                                 const std::string& dtype = DATATYPE_FLOAT32) {
    std::vector<DataT> vecs;
    if constexpr (std::is_same<DataT, float>::value == true) {
        vecs = fixtures::generate_vectors(count, dim, false);
    } else if constexpr (std::is_same<DataT, int8_t>::value == true) {
        vecs = fixtures::generate_int8_codes(count, dim, false);
    } else if constexpr (std::is_same<DataT, uint16_t>::value == true) {
        auto [codes, floats] = (dtype == DATATYPE_FLOAT16)
                                   ? fixtures::generate_fp16_codes_with_floats(count, dim, 43)
                                   : fixtures::generate_bf16_codes_with_floats(count, dim, 43);
        vecs = std::move(codes);
    } else {
        static_assert("Unsupported DataT type");
    }

    if (retrain) {
        quantizer.ReTrain(reinterpret_cast<float*>(vecs.data()), count);
    }

    for (int64_t i = 0; i < count; ++i) {
        auto idx1 = random() % count;
        auto idx2 = random() % count;
        std::vector<uint8_t> codes1(quantizer.GetCodeSize());
        std::vector<uint8_t> codes2(quantizer.GetCodeSize());
        quantizer.EncodeOne(reinterpret_cast<float*>(vecs.data() + idx1 * dim), codes1.data());
        quantizer.EncodeOne(reinterpret_cast<float*>(vecs.data() + idx2 * dim), codes2.data());
        float gt = 0.0;
        float value = quantizer.Compute(codes1.data(), codes2.data());
        if constexpr (std::is_same<DataT, int8_t>::value == true) {
            if constexpr (metric == MetricType::METRIC_TYPE_IP ||
                          metric == MetricType::METRIC_TYPE_COSINE) {
                gt = 1 - INT8InnerProduct(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
            } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
                gt = INT8L2Sqr(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
            }
        } else if constexpr (std::is_same<DataT, uint16_t>::value == true) {
            std::vector<float> float_vec1(dim), float_vec2(dim);
            for (uint64_t d = 0; d < dim; ++d) {
                if (dtype == DATATYPE_FLOAT16) {
                    float_vec1[d] = generic::FP16ToFloat(vecs[idx1 * dim + d]);
                    float_vec2[d] = generic::FP16ToFloat(vecs[idx2 * dim + d]);
                } else {
                    float_vec1[d] = generic::BF16ToFloat(vecs[idx1 * dim + d]);
                    float_vec2[d] = generic::BF16ToFloat(vecs[idx2 * dim + d]);
                }
            }
            if constexpr (metric == MetricType::METRIC_TYPE_IP ||
                          metric == MetricType::METRIC_TYPE_COSINE) {
                gt = 1 - InnerProduct(float_vec1.data(), float_vec2.data(), &dim);
            } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
                gt = L2Sqr(float_vec1.data(), float_vec2.data(), &dim);
            }
        } else {
            std::vector<float> float_vec1(dim), float_vec2(dim);
            for (uint64_t d = 0; d < dim; ++d) {
                float_vec1[d] = static_cast<float>(vecs[idx1 * dim + d]);
                float_vec2[d] = static_cast<float>(vecs[idx2 * dim + d]);
            }
            if constexpr (metric == MetricType::METRIC_TYPE_IP ||
                          metric == MetricType::METRIC_TYPE_COSINE) {
                gt = 1 - InnerProduct(float_vec1.data(), float_vec2.data(), &dim);
            } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
                gt = L2Sqr(float_vec1.data(), float_vec2.data(), &dim);
            }
        }
        if (gt != 0.0) {
            REQUIRE(std::abs((gt - value) / gt) < error);
        } else {
            REQUIRE(std::abs(gt - value) < error);
        }
    }
}

template <typename T, MetricType metric, typename DataT = float>
void
TestQuantizerAdapterComputer(Quantizer<T>& quant,
                             uint64_t dim,
                             uint32_t count,
                             float error = 1e-5f,
                             float related_error = 1.0f,
                             bool retrain = true,
                             float unbounded_numeric_error_rate = 1.0f,
                             float unbounded_related_error_rate = 1.0f,
                             const std::string& dtype = DATATYPE_FLOAT32) {
    auto query_count = 10;
    bool need_normalize = false;
    std::vector<DataT> vecs;
    std::vector<DataT> queries;
    if constexpr (std::is_same<DataT, float>::value == true) {
        vecs = fixtures::generate_vectors(count, dim, need_normalize);
        queries = fixtures::generate_vectors(query_count, dim, need_normalize, 165);
    } else if constexpr (std::is_same<DataT, int8_t>::value == true) {
        vecs = fixtures::generate_int8_codes(count, dim);
        queries = fixtures::generate_int8_codes(query_count, dim, 165);
    } else if constexpr (std::is_same<DataT, uint16_t>::value == true) {
        auto [codes_v, floats_v] = (dtype == DATATYPE_FLOAT16)
                                       ? fixtures::generate_fp16_codes_with_floats(count, dim)
                                       : fixtures::generate_bf16_codes_with_floats(count, dim);
        vecs = std::move(codes_v);
        auto [codes_q, floats_q] =
            (dtype == DATATYPE_FLOAT16)
                ? fixtures::generate_fp16_codes_with_floats(query_count, dim, 165)
                : fixtures::generate_bf16_codes_with_floats(query_count, dim, 165);
        queries = std::move(codes_q);
    } else {
        static_assert("Unsupported DataT type");
    }

    for (int d = 0; d < dim; d++) {
        vecs[d] = 0;
    }
    for (int d = 0; d < dim; d++) {
        queries[query_count * dim / 2 + d] = 0;
    }

    if (retrain) {
        quant.ReTrain(reinterpret_cast<float*>(vecs.data()), count);
    }

    auto gt_func = [&](int base_idx, int query_idx) -> float {
        if constexpr (metric == vsag::MetricType::METRIC_TYPE_IP or
                      metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            if constexpr (std::is_same<DataT, int8_t>::value == true) {
                return 1 - INT8InnerProduct(vecs.data() + base_idx * dim,
                                            queries.data() + query_idx * dim,
                                            &dim);
            } else if constexpr (std::is_same<DataT, uint16_t>::value == true) {
                std::vector<float> float_base(dim), float_query(dim);
                for (uint64_t d = 0; d < dim; ++d) {
                    if (dtype == DATATYPE_FLOAT16) {
                        float_base[d] = generic::FP16ToFloat(vecs[base_idx * dim + d]);
                        float_query[d] = generic::FP16ToFloat(queries[query_idx * dim + d]);
                    } else {
                        float_base[d] = generic::BF16ToFloat(vecs[base_idx * dim + d]);
                        float_query[d] = generic::BF16ToFloat(queries[query_idx * dim + d]);
                    }
                }
                return 1 - InnerProduct(float_base.data(), float_query.data(), &dim);
            } else {
                return 1 - InnerProduct(vecs.data() + base_idx * dim,
                                        queries.data() + query_idx * dim,
                                        &dim);
            }
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            if constexpr (std::is_same<DataT, int8_t>::value == true) {
                return INT8L2Sqr(
                    vecs.data() + base_idx * dim, queries.data() + query_idx * dim, &dim);
            } else if constexpr (std::is_same<DataT, uint16_t>::value == true) {
                std::vector<float> float_base(dim), float_query(dim);
                for (uint64_t d = 0; d < dim; ++d) {
                    if (dtype == DATATYPE_FLOAT16) {
                        float_base[d] = generic::FP16ToFloat(vecs[base_idx * dim + d]);
                        float_query[d] = generic::FP16ToFloat(queries[query_idx * dim + d]);
                    } else {
                        float_base[d] = generic::BF16ToFloat(vecs[base_idx * dim + d]);
                        float_query[d] = generic::BF16ToFloat(queries[query_idx * dim + d]);
                    }
                }
                return L2Sqr(float_base.data(), float_query.data(), &dim);
            } else {
                return L2Sqr(vecs.data() + base_idx * dim, queries.data() + query_idx * dim, &dim);
            }
        }
    };

    float count_unbounded_related_error = 0, count_unbounded_numeric_error = 0;
    for (int i = 0; i < query_count; ++i) {
        auto computer = quant.FactoryComputer();
        computer->SetQuery(reinterpret_cast<float*>(queries.data() + i * dim));

        std::vector<uint8_t> codes1(quant.GetCodeSize() * count, 0);
        std::vector<float> dists1(count);
        for (int j = 0; j < count; ++j) {
            auto gt = gt_func(j, i);
            uint8_t* code = codes1.data() + j * quant.GetCodeSize();
            quant.EncodeOne(reinterpret_cast<float*>(vecs.data() + j * dim), code);
            quant.ComputeDist(*computer, code, dists1.data() + j);
            REQUIRE(quant.ComputeDist(*computer, code) == dists1[j]);
            if (std::abs(gt - dists1[j]) > error) {
                count_unbounded_numeric_error++;
            }
            if (std::abs(gt - dists1[j]) > std::abs(related_error * gt)) {
                count_unbounded_related_error++;
            }
        }

        std::vector<uint8_t> codes2(quant.GetCodeSize() * count);
        std::vector<float> dists2(count);
        quant.EncodeBatch(reinterpret_cast<float*>(vecs.data()), codes2.data(), count);
        quant.ScanBatchDists(*computer, count, codes2.data(), dists2.data());
        for (int j = 0; j < count; ++j) {
            REQUIRE(fixtures::dist_t(dists1[j]) == fixtures::dist_t(dists2[j]));
        }
    }
    REQUIRE(count_unbounded_numeric_error / (query_count * count) <= unbounded_numeric_error_rate);
    REQUIRE(count_unbounded_related_error / (query_count * count) <= unbounded_related_error_rate);
}

template <typename QuantT, MetricType metric, typename DataT>
void
TestQuantizerAdapterSerializeAndDeserialize(Quantizer<QuantT>& quant1,
                                            Quantizer<QuantT>& quant2,
                                            int dim,
                                            int count,
                                            float error = 1e-5f,
                                            float related_error = 1.0f,
                                            float unbounded_numeric_error_rate = 1.0f,
                                            float unbounded_related_error_rate = 1.0f,
                                            bool retrain = true,
                                            const std::string& dtype = DATATYPE_FLOAT32) {
    std::vector<DataT> vecs;
    if constexpr (std::is_same<DataT, float>::value == true) {
        vecs = fixtures::generate_vectors(count, dim, false);
    } else if constexpr (std::is_same<DataT, int8_t>::value == true) {
        vecs = fixtures::generate_int8_codes(count, dim, false);
    } else if constexpr (std::is_same<DataT, uint16_t>::value == true) {
        auto [codes, floats] = (dtype == DATATYPE_FLOAT16)
                                   ? fixtures::generate_fp16_codes_with_floats(count, dim, 43)
                                   : fixtures::generate_bf16_codes_with_floats(count, dim, 43);
        vecs = std::move(codes);
    } else {
        static_assert("Unsupported DataT type");
    }

    quant1.ReTrain(reinterpret_cast<float*>(vecs.data()), count);

    test_serializion(quant1, quant2);

    REQUIRE(quant1.GetCodeSize() == quant2.GetCodeSize());
    REQUIRE(quant1.GetDim() == quant2.GetDim());

    TestQuantizerAdapterEncodeDecode<QuantT, DataT>(quant2, dim, count, error, true, dtype);
    TestQuantizerAdapterComputeCodes<QuantT, metric, DataT>(quant2, dim, count, error, true, dtype);
    TestQuantizerAdapterComputer<QuantT, metric, DataT>(quant2,
                                                        dim,
                                                        count,
                                                        error,
                                                        related_error,
                                                        true,
                                                        unbounded_numeric_error_rate,
                                                        unbounded_related_error_rate,
                                                        dtype);
}
