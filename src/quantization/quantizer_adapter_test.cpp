
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

#include "quantizer_adapter.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

#include "index_common_param.h"
#include "metric_type.h"
#include "quantization/product_quantization/product_quantizer.h"
#include "quantization/quantizer_parameter.h"
#include "quantizer_adapter_test.h"
#include "typing.h"
#include "vsag/constants.h"
#include "vsag/engine.h"
#include "vsag/resource.h"
#include "vsag_exception.h"

using namespace vsag;

struct QuantizerTestConfig {
    QuantizerType quantizer_type;
    std::string metric_str;
    MetricType metric_type;
    std::vector<int> dims;
    std::vector<int> counts;
    float error_threshold;
    float error_multiplier_encode_decode;
    float error_multiplier_compute;
    float error_multiplier_serialize;
};

static const std::vector<QuantizerTestConfig> quantizer_test_configs = {
    {QuantizerType::QUANTIZER_TYPE_PQ,
     "l2",
     MetricType::METRIC_TYPE_L2SQR,
     {128, 256},
     {300},
     8.0F / 255.0F,
     10.0F,
     1.0F,
     5.0F},
    // TODO: Add configurations for other quantizer types:
    // - SQ8: dims from fixtures::get_common_used_dims(), counts {10, 101}, error 1e-2f
    // - FP32: dims {64, 128}, counts {10, 101}, error 2e-5f
    // - RaBitQ: dims from fixtures::get_common_used_dims(), counts {100}
};

constexpr auto dims = {64, 128, 256};
constexpr auto counts = {10, 101, 500};
constexpr MetricType metrics[3] = {
    MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};

QuantizerParamPtr
CreateQuantizerParam(const QuantizerType& quantization_type, uint64_t dim) {
    switch (quantization_type) {
        case QuantizerType::QUANTIZER_TYPE_PQ: {
            JsonType params;
            params[PRODUCT_QUANTIZATION_DIM_KEY].SetInt(dim);
            auto pq_param = std::make_shared<ProductQuantizerParameter>();
            pq_param->FromJson(params);
            return pq_param;
        }
        // TODO: Implement parameter creation for other quantizer types:
        // SQ8, SQ4, FP32, FP16, BF16, INT8, PQFS, RABITQ, SPARSE, TQ
        case QuantizerType::QUANTIZER_TYPE_SQ8:
        case QuantizerType::QUANTIZER_TYPE_SQ8_UNIFORM:
        case QuantizerType::QUANTIZER_TYPE_SQ4:
        case QuantizerType::QUANTIZER_TYPE_SQ4_UNIFORM:
        case QuantizerType::QUANTIZER_TYPE_FP32:
        case QuantizerType::QUANTIZER_TYPE_FP16:
        case QuantizerType::QUANTIZER_TYPE_BF16:
        case QuantizerType::QUANTIZER_TYPE_INT8:
        case QuantizerType::QUANTIZER_TYPE_PQFS:
        case QuantizerType::QUANTIZER_TYPE_RABITQ:
        case QuantizerType::QUANTIZER_TYPE_SPARSE:
        case QuantizerType::QUANTIZER_TYPE_TQ:
            return nullptr;
        default:
            return nullptr;
    }
}

IndexCommonParam
CreateIndexCommonParam(uint64_t dim,
                       std::shared_ptr<Resource> res,
                       MetricType metric_type = MetricType::METRIC_TYPE_L2SQR) {
    std::string metric_str;
    switch (metric_type) {
        case MetricType::METRIC_TYPE_L2SQR:
            metric_str = "l2";
            break;
        case MetricType::METRIC_TYPE_IP:
            metric_str = "ip";
            break;
        case MetricType::METRIC_TYPE_COSINE:
            metric_str = "cosine";
            break;
        default:
            metric_str = "l2";
            break;
    }

    JsonType params;
    params[PARAMETER_DTYPE].SetString(DATATYPE_INT8);
    params[PARAMETER_METRIC_TYPE].SetString(metric_str);
    params[PARAMETER_DIM].SetInt(dim);
    return IndexCommonParam::CheckAndCreate(params, res);
}

template <typename QuantT, MetricType metric>
void
TestQuantizerAdapterEncodeDecodeINT8(QuantizerType quantizer_type,
                                     uint64_t dim,
                                     int count,
                                     float error = 1e-5) {
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    try {
        const QuantizerParamPtr quantizer_param = CreateQuantizerParam(quantizer_type, dim);
        if (quantizer_param == nullptr) {
            return;
        }
        const IndexCommonParam common_param =
            CreateIndexCommonParam(dim, std::make_shared<Resource>(resource), metric);
        auto adapter =
            std::make_shared<QuantizerAdapter<QuantT, int8_t>>(quantizer_param, common_param);
        TestQuantizerAdapterEncodeDecode<QuantizerAdapter<QuantT, int8_t>, int8_t>(
            *adapter, dim, count, error);
    } catch (const vsag::VsagException& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        throw;
    }
}

TEST_CASE("QuantizerAdapter Encode and Decode", "[ut][QuantizerAdapter][EncodeDecode]") {
    for (const auto& config : quantizer_test_configs) {
        for (auto dim : config.dims) {
            for (auto count : config.counts) {
                float error = config.error_threshold * config.error_multiplier_encode_decode;
                TestQuantizerAdapterEncodeDecodeINT8<
                    ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                    MetricType::METRIC_TYPE_L2SQR>(config.quantizer_type, dim, count, error);
                // TODO: Add tests for IP and COSINE metrics when needed
            }
        }
    }
}

template <typename QuantT, MetricType metric>
void
TestQuantizerAdapterCompute(QuantizerType quantizer_type,
                            uint64_t dim,
                            int count,
                            float error = 1e-5) {
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    try {
        const QuantizerParamPtr quantizer_param = CreateQuantizerParam(quantizer_type, dim);
        if (quantizer_param == nullptr) {
            return;
        }
        const IndexCommonParam common_param =
            CreateIndexCommonParam(dim, std::make_shared<Resource>(resource), metric);
        auto adapter =
            std::make_shared<QuantizerAdapter<QuantT, int8_t>>(quantizer_param, common_param);
        TestQuantizerAdapterComputeCodes<QuantizerAdapter<QuantT, int8_t>, metric, int8_t>(
            *adapter, dim, count, error);
        TestQuantizerAdapterComputer<QuantizerAdapter<QuantT, int8_t>, metric, int8_t>(
            *adapter, dim, count, error);
    } catch (const vsag::VsagException& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        throw;
    }
}
TEST_CASE("QuantizerAdapter Compute", "[ut][QuantizerAdapter][Compute]") {
    for (const auto& config : quantizer_test_configs) {
        for (auto dim : config.dims) {
            for (auto count : config.counts) {
                float error = config.error_threshold * config.error_multiplier_compute;
                TestQuantizerAdapterCompute<ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                            MetricType::METRIC_TYPE_L2SQR>(
                    config.quantizer_type, dim, count, error);
            }
        }
    }
}

template <typename QuantT, MetricType metric>
void
TestAdapterSerializeAndDeserialize(QuantizerType quantizer_type,
                                   uint64_t dim,
                                   int count,
                                   float error = 1e-5) {
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    try {
        const QuantizerParamPtr quantizer_param = CreateQuantizerParam(quantizer_type, dim);
        if (quantizer_param == nullptr) {
            return;
        }
        const IndexCommonParam common_param =
            CreateIndexCommonParam(dim, std::make_shared<Resource>(resource), metric);
        auto adapter1 =
            std::make_shared<QuantizerAdapter<QuantT, int8_t>>(quantizer_param, common_param);
        auto adapter2 =
            std::make_shared<QuantizerAdapter<QuantT, int8_t>>(quantizer_param, common_param);
        TestQuantizerAdapterSerializeAndDeserialize<QuantizerAdapter<QuantT, int8_t>,
                                                    metric,
                                                    int8_t>(
            *adapter1, *adapter2, dim, count, error);
    } catch (const vsag::VsagException& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        throw;
    }
}
TEST_CASE("QuantizerAdapter Serialize AND Deserialize", "[ut][QuantizerAdapter][Serialize]") {
    for (const auto& config : quantizer_test_configs) {
        for (auto dim : config.dims) {
            for (auto count : config.counts) {
                float error = config.error_threshold * config.error_multiplier_serialize;
                TestAdapterSerializeAndDeserialize<ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                                   MetricType::METRIC_TYPE_L2SQR>(
                    config.quantizer_type, dim, count, error);
            }
        }
    }
}
