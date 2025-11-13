
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

#include "flatten_interface.h"

#include "flatten_datacell_inst.h"
#include "inner_string_params.h"
#include "io/io_headers.h"
#include "quantization/int8_quantizer.h"
#include "quantization/quantizer_adapter.h"
#include "quantization/quantizer_headers.h"
#include "quantization/sparse_quantization/sparse_quantizer.h"
#include "quantization/transform_quantization/transform_quantizer_parameter.h"
#include "sparse_vector_datacell.h"

#define GENERATE_QUANTIZER_CASE(type_suffix, quantizer_class)                                      \
    if (quantization_string == QUANTIZATION_TYPE_VALUE_##type_suffix) {                            \
        return creator::create_flatten_cell<quantizer_class<metric>, IOTemp>(param, common_param); \
    }

#define GENERATE_TQ_CASE(type_suffix, quantizer_class)                                         \
    if (tq_bottom_quantization_string == QUANTIZATION_TYPE_VALUE_##type_suffix) {              \
        return creator::create_tq_cell<quantizer_class<metric>, metric, IOTemp>(param,         \
                                                                                common_param); \
    }

#define GENERATE_RQ_CASE(type_suffix, quantizer_class)                                    \
    if (rq_bottom_quantization_string == QUANTIZATION_TYPE_VALUE_##type_suffix) {         \
        return creator::create_rq_cell<quantizer_class<vsag::MetricType::METRIC_TYPE_IP>, \
                                       metric,                                            \
                                       IOTemp>(param, common_param);                      \
    }

#define GENERATE_TRQ_QUANTIZER_CASE(type_suffix, quantizer_class)                          \
    if (trq_bottom_quantization_string == QUANTIZATION_TYPE_VALUE_##type_suffix) {         \
        return creator::create_trq_cell<quantizer_class<vsag::MetricType::METRIC_TYPE_IP>, \
                                        metric,                                            \
                                        IOTemp>(param, common_param);                      \
    }

namespace vsag {

template <MetricType metric, typename IOTemp>
static FlattenInterfacePtr
make_instance(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param) {
    std::string quantization_string = param->quantizer_parameter->GetTypeName();
    if (common_param.data_type_ == DataTypes::DATA_TYPE_INT8) {
        if (quantization_string == QUANTIZATION_TYPE_VALUE_INT8) {
            return creator::create_flatten_cell<INT8Quantizer<metric>, IOTemp>(param, common_param);
        }

        if (quantization_string == QUANTIZATION_TYPE_VALUE_PQ) {
            return creator::create_quantizer_adapter_cell<ProductQuantizer<metric>, IOTemp>(
                param, common_param);
        }

        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("INT8 data type unsupport {} quantization", quantization_string));
    }

    if (quantization_string == QUANTIZATION_TYPE_VALUE_TQ) {
        auto tq_param =
            std::dynamic_pointer_cast<TransformQuantizerParameter>(param->quantizer_parameter);
        auto tq_bottom_quantization_string = tq_param->GetBottomQuantizationName();

        if (tq_bottom_quantization_string == QUANTIZATION_TYPE_VALUE_RQ) {
            auto rq_param = std::make_shared<ResidualQuantizerParameter>();
            rq_param->FromJson(tq_param->base_quantizer_json_);
            auto trq_bottom_quantization_string = rq_param->GetBottomQuantizationName();

            FOR_EACH_DENSE_QUANTIZER(GENERATE_TRQ_QUANTIZER_CASE)
        }

        FOR_EACH_DENSE_QUANTIZER(GENERATE_TQ_CASE)
    }

    if (quantization_string == QUANTIZATION_TYPE_VALUE_RQ) {
        auto rq_param =
            std::dynamic_pointer_cast<ResidualQuantizerParameter>(param->quantizer_parameter);
        auto rq_bottom_quantization_string = rq_param->GetBottomQuantizationName();
        FOR_EACH_DENSE_QUANTIZER(GENERATE_RQ_CASE)
    }

    FOR_EACH_ALL_QUANTIZER(GENERATE_QUANTIZER_CASE)

    return nullptr;
}

template <typename IOTemp>
static FlattenInterfacePtr
make_instance(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param) {
    auto metric = common_param.metric_;
    if (metric == MetricType::METRIC_TYPE_L2SQR) {
        return make_instance<MetricType::METRIC_TYPE_L2SQR, IOTemp>(param, common_param);
    }
    if (metric == MetricType::METRIC_TYPE_IP) {
        return make_instance<MetricType::METRIC_TYPE_IP, IOTemp>(param, common_param);
    }
    if (metric == MetricType::METRIC_TYPE_COSINE) {
        return make_instance<MetricType::METRIC_TYPE_COSINE, IOTemp>(param, common_param);
    }
    return nullptr;
}

FlattenInterfacePtr
FlattenInterface::MakeInstance(const FlattenInterfaceParamPtr& param,
                               const IndexCommonParam& common_param) {
    auto io_type_name = param->io_parameter->GetTypeName();
    if (io_type_name == IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        return make_instance<MemoryBlockIO>(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_MEMORY_IO) {
        return make_instance<MemoryIO>(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_BUFFER_IO) {
        return make_instance<BufferIO>(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_ASYNC_IO) {
        return make_instance<AsyncIO>(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_MMAP_IO) {
        return make_instance<MMapIO>(param, common_param);
    }
    if (io_type_name == IO_TYPE_VALUE_READER_IO) {
        return make_instance<ReaderIO>(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
