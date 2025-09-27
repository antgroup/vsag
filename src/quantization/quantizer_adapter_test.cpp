
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
#include "vsag/engine.h"
#include "vsag/resource.h"
#include "vsag_exception.h"

using namespace vsag;

constexpr auto dims = {64, 128, 512};
constexpr auto counts = {10, 101, 1000};
constexpr MetricType metrics[3] = {
    MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_COSINE, MetricType::METRIC_TYPE_IP};

QuantizerParamPtr
CreateQuantizerParam(const QuantizerType& quantization_type, uint64_t dim) {
    if (quantization_type == QuantizerType::QUANTIZER_TYPE_PQ) {
        JsonType params;
        params[PRODUCT_QUANTIZATION_DIM] = dim;
        auto pq_param = std::make_shared<ProductQuantizerParameter>();
        pq_param->FromJson(params);
        return pq_param;
    }
    return nullptr;
}

IndexCommonParam
CreateIndexCommonParam(uint64_t dim,
                       std::shared_ptr<Resource> res,
                       std::string metric = std::string("l2")) {
    JsonType params;
    params[PARAMETER_DTYPE] = DATATYPE_INT8;
    params[PARAMETER_METRIC_TYPE] = metric;
    params[PARAMETER_DIM] = dim;
    return IndexCommonParam::CheckAndCreate(params, res);
}

template <typename QuantT>
void
TestQuantizerAdapterEncodeDecodeINT8(std::string metric,
                                     uint64_t dim,
                                     int count,
                                     float error = 1e-5) {
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    try {
        const QuantizerParamPtr quantizer_param =
            CreateQuantizerParam(QuantizerType::QUANTIZER_TYPE_PQ, dim);
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
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 8.0F / 255.0F * 10.0F;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestQuantizerAdapterEncodeDecodeINT8<ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>>(
                "l2", dim, count, error);
        }
    }
}

template <typename QuantT, MetricType metric>
void
TestQuantizerAdapterCompute(uint64_t dim, int count, float error = 1e-5) {
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    try {
        const QuantizerParamPtr quantizer_param =
            CreateQuantizerParam(QuantizerType::QUANTIZER_TYPE_PQ, dim);
        const IndexCommonParam common_param =
            CreateIndexCommonParam(dim, std::make_shared<Resource>(resource));
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
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 8.0F / 255.0F;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestQuantizerAdapterCompute<ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                        MetricType::METRIC_TYPE_L2SQR>(dim, count, error);
        }
    }
}

template <typename QuantT, MetricType metric>
void
TestAdapterSerializeAndDeserialize(uint64_t dim, int count, float error = 1e-5) {
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    try {
        const QuantizerParamPtr quantizer_param =
            CreateQuantizerParam(QuantizerType::QUANTIZER_TYPE_PQ, dim);
        const IndexCommonParam common_param =
            CreateIndexCommonParam(dim, std::make_shared<Resource>(resource));
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
    constexpr MetricType metrics[2] = {MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP};
    float error = 8.0F / 255.0F * 5.0F;
    for (auto dim : dims) {
        for (auto count : counts) {
            TestAdapterSerializeAndDeserialize<ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                               MetricType::METRIC_TYPE_L2SQR>(dim, count, error);
        }
    }
}
