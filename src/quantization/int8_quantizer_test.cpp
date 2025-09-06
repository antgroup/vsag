
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

#include "quantization/int8_quantizer.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <vector>

#include "catch2/catch_test_macros.hpp"
#include "fixtures.h"
#include "impl/allocator/safe_allocator.h"
#include "metric_type.h"
#include "quantization/computer.h"
#include "quantization/quantizer.h"
#include "quantizer_test.h"
#include "simd/basic_func.h"

namespace vsag {

constexpr auto dims = {64, 128};
constexpr auto counts = {10, 101};

template <MetricType metric>
void
TestQuantizerEncodeDecodeINT8(Quantizer<INT8Quantizer<metric>>& quant,
                              int64_t dim,
                              int count,
                              float error = 1e-5,
                              bool retrain = true) {
    auto vecs = fixtures::generate_int8_codes(count, dim);
    if (retrain) {
        quant.ReTrain(reinterpret_cast<DataTypePtr>(vecs.data()), count);
    }
    // Test EncodeOne & DecodeOne
    for (uint64_t i = 0; i < count; ++i) {
        std::vector<uint8_t> codes(quant.GetCodeSize());
        quant.EncodeOne(reinterpret_cast<DataTypePtr>(vecs.data() + i * dim), codes.data());
        std::vector<int8_t> out_vec(dim);
        quant.DecodeOne(codes.data(), reinterpret_cast<DataTypePtr>(out_vec.data()));
        float sum = 0.0F;
        for (int j = 0; j < dim; ++j) {
            sum += std::abs(static_cast<DataType>(vecs[i * dim + j]) -
                            static_cast<DataType>(out_vec[j]));
        }
        REQUIRE(sum < error * dim);
    }

    // Test EncodeBatch & DecodeBatch
    std::vector<uint8_t> codes(quant.GetCodeSize() * count);
    quant.EncodeBatch(reinterpret_cast<DataTypePtr>(vecs.data()), codes.data(), count);
    std::vector<int8_t> out_vec(dim * count);
    quant.DecodeBatch(codes.data(), reinterpret_cast<DataTypePtr>(out_vec.data()), count);
    for (int64_t i = 0; i < count; ++i) {
        float sum = 0.0F;
        for (int j = 0; j < dim; ++j) {
            sum += std::abs(static_cast<DataType>(vecs[i * dim + j]) -
                            static_cast<DataType>(out_vec[i * dim + j]));
        }
        REQUIRE(sum < error * dim);
    }
}

template <MetricType metric>
void
TestQuantizerEncodeDecodeMetricINT8(int64_t dim, int64_t count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    INT8Quantizer<metric> quantizer(dim, allocator.get());
    TestQuantizerEncodeDecodeINT8(quantizer, dim, count, error);
}

TEST_CASE("INT8 Quantizer Encode and Decode", "[ut][INT8Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE};
    float error = 2e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestQuantizerEncodeDecodeMetricINT8<metrics[0]>(dim, count, error);
            TestQuantizerEncodeDecodeMetricINT8<metrics[1]>(dim, count, error);
            TestQuantizerEncodeDecodeMetricINT8<metrics[2]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestComputeCodesINT8(Quantizer<INT8Quantizer<metric>>& quantizer,
                     size_t dim,
                     uint32_t count,
                     float error = 1e-4f,
                     bool retrain = true) {
    auto vecs = fixtures::generate_int8_codes(count, dim);
    if (retrain) {
        quantizer.ReTrain(reinterpret_cast<DataTypePtr>(vecs.data()), count);
    }
    for (int i = 0; i < count; ++i) {
        auto idx1 = random() % count;
        auto idx2 = random() % count;
        std::vector<uint8_t> codes1(quantizer.GetCodeSize());
        std::vector<uint8_t> codes2(quantizer.GetCodeSize());
        quantizer.EncodeOne(reinterpret_cast<DataTypePtr>(vecs.data() + idx1 * dim), codes1.data());
        quantizer.EncodeOne(reinterpret_cast<DataTypePtr>(vecs.data() + idx2 * dim), codes2.data());
        float gt = 0.0;
        float value = quantizer.Compute(codes1.data(), codes2.data());
        if constexpr (metric == vsag::MetricType::METRIC_TYPE_IP) {
            gt = INT8InnerProduct(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            gt = INT8L2Sqr(vecs.data() + idx1 * dim, vecs.data() + idx2 * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            gt = 0.0f;
        }
        REQUIRE(std::abs(gt - value) < error);
    }
}

template <MetricType metric>
void
TestComputerINT8(Quantizer<INT8Quantizer<metric>>& quant,
                 size_t dim,
                 uint32_t count,
                 float error = 1e-5f,
                 float related_error = 1.0f,
                 bool retrain = true,
                 float unbounded_numeric_error_rate = 1.0f,
                 float unbounded_related_error_rate = 1.0f) {
    auto query_count = 10;
    bool need_normalize = false;
    auto vecs = fixtures::generate_int8_codes(count, dim);
    auto queries = fixtures::generate_int8_codes(query_count, dim, 165);

    if (retrain) {
        quant.ReTrain(reinterpret_cast<DataTypePtr>(vecs.data()), count);
    }

    auto gt_func = [&](int base_idx, int query_idx) -> float {
        if constexpr (metric == vsag::MetricType::METRIC_TYPE_IP) {
            return INT8InnerProduct(
                vecs.data() + base_idx * dim, queries.data() + query_idx * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_L2SQR) {
            return INT8L2Sqr(vecs.data() + base_idx * dim, queries.data() + query_idx * dim, &dim);
        } else if constexpr (metric == vsag::MetricType::METRIC_TYPE_COSINE) {
            // TODO: cosine
            // std::vector<int8_t> zeros(dim, 0);
            // float norm = INT8L2Sqr(vecs.data() + base_idx * dim, zeros.data(), &dim);
            // float query_norm = INT8L2Sqr(queries.data() + query_idx * dim, zeros.data(), &dim);
            // float dot = INT8InnerProduct(
            //     vecs.data() + base_idx * dim, queries.data() + query_idx * dim, &dim);
            // return 1.0f - dot / (norm * query_norm);
            return 0.0f;
        }
    };

    float count_unbounded_related_error = 0, count_unbounded_numeric_error = 0;
    for (int i = 0; i < query_count; ++i) {
        std::shared_ptr<Computer<INT8Quantizer<metric>>> computer;
        computer = quant.FactoryComputer();
        computer->SetQuery(reinterpret_cast<DataTypePtr>(queries.data() + i * dim));

        //Test Compute One Dist;
        std::vector<uint8_t> codes1(quant.GetCodeSize() * count, 0);
        std::vector<float> dists1(count);
        for (int j = 0; j < count; ++j) {
            auto gt = gt_func(j, i);
            uint8_t* code = codes1.data() + j * quant.GetCodeSize();
            quant.EncodeOne(reinterpret_cast<DataTypePtr>(vecs.data() + j * dim), code);
            quant.ComputeDist(*computer, code, dists1.data() + j);
            REQUIRE(quant.ComputeDist(*computer, code) == dists1[j]);
            if (std::abs(gt - dists1[j]) > error) {
                count_unbounded_numeric_error++;
            }
            if (std::abs(gt - dists1[j]) > std::abs(related_error * gt)) {
                count_unbounded_related_error++;
            }
        }

        // Test Compute Batch
        std::vector<uint8_t> codes2(quant.GetCodeSize() * count, 0);
        std::vector<float> dists2(count);
        quant.EncodeBatch(reinterpret_cast<DataTypePtr>(vecs.data()), codes2.data(), count);
        quant.ScanBatchDists(*computer, count, codes2.data(), dists2.data());
        for (int j = 0; j < count; ++j) {
          REQUIRE(fixtures::dist_t(dists1[j]) == fixtures::dist_t(dists2[j]));
        }
    }
    REQUIRE(count_unbounded_numeric_error / (query_count * count) <= unbounded_numeric_error_rate);
    REQUIRE(count_unbounded_related_error / (query_count * count) <= unbounded_related_error_rate);
}

template <MetricType metric>
void
TestComputeMetricINT8(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    INT8Quantizer<metric> quantizer(dim, allocator.get());
    TestComputeCodesINT8<metric>(quantizer, dim, count, error);
    TestComputerINT8<metric>(quantizer, dim, count, error, 1.01, true, 0.01, 0.01);
}

TEST_CASE("INT8 Compute", "[ut][INT8Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 2e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestComputeMetricINT8<metrics[0]>(dim, count, error);
            // TestComputeMetricINT8<metrics[1]>(dim, count, error);
            TestComputeMetricINT8<metrics[2]>(dim, count, error);
        }
    }
}

template <MetricType metric>
void
TestSerializeAndDeserializeINT8(Quantizer<INT8Quantizer<metric>>& quant1,
                                Quantizer<INT8Quantizer<metric>>& quant2,
                                size_t dim,
                                uint32_t count,
                                float error = 1e-5f,
                                float related_error = 1.0f,
                                float unbounded_numeric_error_rate = 1.0f,
                                float unbounded_related_error_rate = 1.0f) {
    auto vecs = fixtures::generate_int8_codes(count, dim);
    quant1.ReTrain(reinterpret_cast<DataTypePtr>(vecs.data()), count);

    test_serializion(quant1, quant2);

    REQUIRE(quant1.GetCodeSize() == quant2.GetCodeSize());
    REQUIRE(quant1.GetDim() == quant2.GetDim());

    TestQuantizerEncodeDecodeINT8(quant2, dim, count, error, false);
    TestComputerINT8(quant2,
                     dim,
                     count,
                     error,
                     related_error,
                     false,
                     unbounded_numeric_error_rate,
                     unbounded_related_error_rate);
    TestComputeCodesINT8(quant2, dim, count, error * dim * 2.0F, false);
}

template <MetricType metric>
void
TestSerializeAndDeserializeMetricINT8(uint64_t dim, int count, float error = 1e-5) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    INT8Quantizer<metric> quantizer1(dim, allocator.get());
    INT8Quantizer<metric> quantizer2(dim, allocator.get());
    TestSerializeAndDeserializeINT8<metric>(quantizer1, quantizer2, dim, count, error);
}

TEST_CASE("INT8 Serialize and Deserialize", "[ut][INT8Quantizer]") {
    constexpr MetricType metrics[3] = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};
    float error = 2e-5f;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestSerializeAndDeserializeMetricINT8<metrics[0]>(dim, count, error);
            // TODO: impl cosine
            // TestSerializeAndDeserializeMetricINT8<metrics[1]>(dim, count, error);
            TestSerializeAndDeserializeMetricINT8<metrics[2]>(dim, count, error);
        }
    }
}

}  // namespace vsag
