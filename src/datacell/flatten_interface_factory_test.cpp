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

#include <array>
#include <tuple>

#include "flatten_datacell.h"
#include "flatten_datacell_parameter.h"
#include "impl/allocator/safe_allocator.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "io/io_headers.h"
#include "multi_vector_datacell.h"
#include "multi_vector_datacell_parameter.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/quantizer_parameter.h"
#include "quantization/sparse_quantization/sparse_quantizer.h"
#include "quantization/transform_quantization/transform_quantizer.h"
#include "quantization/transform_quantization/transform_quantizer_parameter.h"
#include "sparse_vector_datacell.h"
#include "sparse_vector_datacell_parameter.h"
#include "unittest.h"

namespace vsag {
namespace {

IOParamPtr
MakeFactoryIOParam(const std::string& io_type, const std::string& path) {
    JsonType json;
    json[TYPE_KEY].SetString(io_type);
    json[IO_FILE_PATH_KEY].SetString(path);
    return IOParameter::GetIOParameterByJson(json);
}

IndexCommonParam
MakeFactoryCommonParam(MetricType metric = MetricType::METRIC_TYPE_IP) {
    IndexCommonParam common_param;
    common_param.allocator_ = SafeAllocator::FactoryDefaultAllocator();
    common_param.dim_ = 16;
    common_param.metric_ = metric;
    return common_param;
}

FlattenDataCellParamPtr
MakeFlattenFactoryParam(const std::string& quantization,
                        const std::string& io_type,
                        const std::string& path,
                        bool transform = false) {
    auto param = std::make_shared<FlattenDataCellParameter>();
    param->io_parameter = MakeFactoryIOParam(io_type, path);
    if (transform) {
        param->quantizer_parameter =
            TransformQuantizerParameter::CreateDefault("rom," + quantization);
    } else {
        param->quantizer_parameter = QuantizerParameter::CreateDefault(quantization);
    }
    return param;
}

MultiVectorDataCellParamPtr
MakeMultiVectorFactoryParam(const std::string& quantization,
                            const std::string& io_type,
                            const std::string& path) {
    auto param = std::make_shared<MultiVectorDataCellParameter>();
    param->io_parameter = MakeFactoryIOParam(io_type, path);
    param->quantizer_parameter = QuantizerParameter::CreateDefault(quantization);
    return param;
}

SparseVectorDataCellParamPtr
MakeSparseFactoryParam(const std::string& io_type, const std::string& path) {
    auto param = std::make_shared<SparseVectorDataCellParameter>();
    param->io_parameter = MakeFactoryIOParam(io_type, path);
    param->quantizer_parameter = QuantizerParameter::CreateDefault(QUANTIZATION_TYPE_VALUE_SPARSE);
    return param;
}

template <typename IOTmpl>
void
CheckFactoryIO(const std::string& io_type, fixtures::TempDir& dir) {
    auto common_param = MakeFactoryCommonParam();

    auto flatten_param = MakeFlattenFactoryParam(
        QUANTIZATION_TYPE_VALUE_FP32, io_type, dir.GenerateRandomFile(false));
    auto flatten = FlattenInterface::MakeInstance(flatten_param, common_param);
    using FlattenCell =
        FlattenDataCell<FP32Quantizer<MetricType::METRIC_TYPE_IP>, FixedLayout<IOTmpl>>;
    REQUIRE(std::dynamic_pointer_cast<FlattenCell>(flatten) != nullptr);

    auto multi_param = MakeMultiVectorFactoryParam(
        QUANTIZATION_TYPE_VALUE_FP32, io_type, dir.GenerateRandomFile(false));
    auto multi = FlattenInterface::MakeInstance(multi_param, common_param);
    using MultiVectorCell = MultiVectorDataCell<FP32Quantizer<MetricType::METRIC_TYPE_IP>, IOTmpl>;
    REQUIRE(std::dynamic_pointer_cast<MultiVectorCell>(multi) != nullptr);

    common_param.data_type_ = DataTypes::DATA_TYPE_SPARSE;
    common_param.repr_ = RecordRepr::SPARSE;
    auto sparse_param = MakeSparseFactoryParam(io_type, dir.GenerateRandomFile(false));
    auto sparse = FlattenInterface::MakeInstance(sparse_param, common_param);
    using SparseCell = SparseVectorDataCell<SparseQuantizer<MetricType::METRIC_TYPE_IP>, IOTmpl>;
    REQUIRE(std::dynamic_pointer_cast<SparseCell>(sparse) != nullptr);
}

template <MetricType metric>
void
CheckFlattenMetricTypes() {
    auto common_param = MakeFactoryCommonParam(metric);
    auto param = MakeFlattenFactoryParam(
        QUANTIZATION_TYPE_VALUE_FP32, IO_TYPE_VALUE_MEMORY_IO, std::string());
    auto flatten = FlattenInterface::MakeInstance(param, common_param);
    using FlattenCell = FlattenDataCell<FP32Quantizer<metric>, FixedLayout<MemoryIO>>;
    REQUIRE(std::dynamic_pointer_cast<FlattenCell>(flatten) != nullptr);

    auto transform_param = MakeFlattenFactoryParam(
        QUANTIZATION_TYPE_VALUE_FP32, IO_TYPE_VALUE_MEMORY_IO, std::string(), true);
    auto transformed = FlattenInterface::MakeInstance(transform_param, common_param);
    using TransformCell =
        FlattenDataCell<TransformQuantizer<FP32Quantizer<metric>, metric>, FixedLayout<MemoryIO>>;
    REQUIRE(std::dynamic_pointer_cast<TransformCell>(transformed) != nullptr);
}

}  // namespace

TEST_CASE("FlattenInterface factory dispatches every IO for each data-cell family",
          "[ut][FlattenInterface][factory]") {
    fixtures::TempDir dir("flatten_interface_factory");
    CheckFactoryIO<MemoryBlockIO>(IO_TYPE_VALUE_BLOCK_MEMORY_IO, dir);
    CheckFactoryIO<MemoryIO>(IO_TYPE_VALUE_MEMORY_IO, dir);
    CheckFactoryIO<BufferIO>(IO_TYPE_VALUE_BUFFER_IO, dir);
    CheckFactoryIO<AsyncIO>(IO_TYPE_VALUE_ASYNC_IO, dir);
    CheckFactoryIO<UringIO>(IO_TYPE_VALUE_URING_IO, dir);
    CheckFactoryIO<MMapIO>(IO_TYPE_VALUE_MMAP_IO, dir);
    CheckFactoryIO<ReaderIO>(IO_TYPE_VALUE_READER_IO, dir);
}

TEST_CASE("FlattenInterface factory preserves dense quantizer and data-type dispatch",
          "[ut][FlattenInterface][factory]") {
    constexpr std::array<MetricType, 3> metrics = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE};
    const std::array<const char*, 10> quantizations = {QUANTIZATION_TYPE_VALUE_SQ8,
                                                       QUANTIZATION_TYPE_VALUE_FP32,
                                                       QUANTIZATION_TYPE_VALUE_SQ4,
                                                       QUANTIZATION_TYPE_VALUE_SQ4_UNIFORM,
                                                       QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM,
                                                       QUANTIZATION_TYPE_VALUE_BF16,
                                                       QUANTIZATION_TYPE_VALUE_FP16,
                                                       QUANTIZATION_TYPE_VALUE_PQ,
                                                       QUANTIZATION_TYPE_VALUE_PQFS,
                                                       QUANTIZATION_TYPE_VALUE_RABITQ};

    for (const auto metric : metrics) {
        for (const auto* quantization : quantizations) {
            CAPTURE(metric, quantization);
            auto common_param = MakeFactoryCommonParam(metric);
            auto param =
                MakeFlattenFactoryParam(quantization, IO_TYPE_VALUE_MEMORY_IO, std::string());
            auto flatten = FlattenInterface::MakeInstance(param, common_param);
            REQUIRE(flatten != nullptr);
            CHECK(flatten->GetQuantizerName() == quantization);

            auto transform_param =
                MakeFlattenFactoryParam(quantization, IO_TYPE_VALUE_MEMORY_IO, std::string(), true);
            auto transformed = FlattenInterface::MakeInstance(transform_param, common_param);
            REQUIRE(transformed != nullptr);
            CHECK(transformed->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_TQ);
        }
    }

    CheckFlattenMetricTypes<MetricType::METRIC_TYPE_L2SQR>();
    CheckFlattenMetricTypes<MetricType::METRIC_TYPE_IP>();
    CheckFlattenMetricTypes<MetricType::METRIC_TYPE_COSINE>();

    const std::array<std::tuple<DataTypes, const char*, const char*>, 8> typed_quantizations = {
        std::make_tuple(
            DataTypes::DATA_TYPE_INT8, QUANTIZATION_TYPE_VALUE_INT8, QUANTIZATION_TYPE_VALUE_INT8),
        std::make_tuple(
            DataTypes::DATA_TYPE_INT8, QUANTIZATION_TYPE_VALUE_PQ, "QUANTIZATION_ADAPTER_pq"),
        std::make_tuple(
            DataTypes::DATA_TYPE_FP16, QUANTIZATION_TYPE_VALUE_FP16, "QUANTIZATION_ADAPTER_fp16"),
        std::make_tuple(
            DataTypes::DATA_TYPE_FP16, QUANTIZATION_TYPE_VALUE_BF16, "QUANTIZATION_ADAPTER_bf16"),
        std::make_tuple(
            DataTypes::DATA_TYPE_FP16, QUANTIZATION_TYPE_VALUE_PQ, "QUANTIZATION_ADAPTER_pq"),
        std::make_tuple(
            DataTypes::DATA_TYPE_BF16, QUANTIZATION_TYPE_VALUE_FP16, "QUANTIZATION_ADAPTER_fp16"),
        std::make_tuple(
            DataTypes::DATA_TYPE_BF16, QUANTIZATION_TYPE_VALUE_BF16, "QUANTIZATION_ADAPTER_bf16"),
        std::make_tuple(
            DataTypes::DATA_TYPE_BF16, QUANTIZATION_TYPE_VALUE_PQ, "QUANTIZATION_ADAPTER_pq")};
    for (const auto& [data_type, quantization, quantizer_name] : typed_quantizations) {
        CAPTURE(data_type, quantization);
        auto common_param = MakeFactoryCommonParam();
        common_param.data_type_ = data_type;
        auto param = MakeFlattenFactoryParam(quantization, IO_TYPE_VALUE_MEMORY_IO, std::string());
        auto flatten = FlattenInterface::MakeInstance(param, common_param);
        REQUIRE(flatten != nullptr);
        CHECK(flatten->GetQuantizerName() == quantizer_name);
    }
}

TEST_CASE("FlattenInterface factory preserves multi-vector and sparse metric dispatch",
          "[ut][FlattenInterface][factory]") {
    constexpr std::array<MetricType, 3> metrics = {
        MetricType::METRIC_TYPE_L2SQR, MetricType::METRIC_TYPE_IP, MetricType::METRIC_TYPE_COSINE};
    const std::array<const char*, 5> quantizations = {QUANTIZATION_TYPE_VALUE_FP32,
                                                      QUANTIZATION_TYPE_VALUE_FP16,
                                                      QUANTIZATION_TYPE_VALUE_BF16,
                                                      QUANTIZATION_TYPE_VALUE_SQ8_UNIFORM,
                                                      QUANTIZATION_TYPE_VALUE_INT8};
    for (const auto metric : metrics) {
        for (const auto* quantization : quantizations) {
            CAPTURE(metric, quantization);
            auto common_param = MakeFactoryCommonParam(metric);
            auto param =
                MakeMultiVectorFactoryParam(quantization, IO_TYPE_VALUE_MEMORY_IO, std::string());
            auto multi = FlattenInterface::MakeInstance(param, common_param);
            REQUIRE(multi != nullptr);
            CHECK(multi->GetQuantizerName() == quantization);
            CHECK(multi->GetMetricType() == metric);
        }
    }

    auto sparse_param = MakeSparseFactoryParam(IO_TYPE_VALUE_MEMORY_IO, std::string());
    auto common_param = MakeFactoryCommonParam(MetricType::METRIC_TYPE_IP);
    common_param.data_type_ = DataTypes::DATA_TYPE_SPARSE;
    REQUIRE(FlattenInterface::MakeInstance(sparse_param, common_param) != nullptr);
    common_param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    REQUIRE_THROWS_AS(FlattenInterface::MakeInstance(sparse_param, common_param), VsagException);
    common_param.metric_ = MetricType::METRIC_TYPE_COSINE;
    REQUIRE_THROWS_AS(FlattenInterface::MakeInstance(sparse_param, common_param), VsagException);
}

}  // namespace vsag
