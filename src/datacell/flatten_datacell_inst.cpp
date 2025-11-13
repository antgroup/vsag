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

#include "flatten_datacell_inst.h"

#include "inner_string_params.h"
#include "io/io_headers.h"
#include "quantization/int8_quantizer.h"
#include "quantization/quantizer_adapter.h"
#include "quantization/quantizer_headers.h"
#include "quantization/sparse_quantization/sparse_quantizer.h"
#include "quantization/transform_quantization/transform_quantizer_parameter.h"

namespace vsag {
namespace creator {

template <typename QuantTemp, typename IOTemp>
FlattenInterfacePtr
create_flatten_cell(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param) {
    return make_instance_impl<QuantTemp, IOTemp>(param, common_param);
}

template <typename QuantTemp, typename IOTemp>
FlattenInterfacePtr
create_quantizer_adapter_cell(const FlattenInterfaceParamPtr& param,
                              const IndexCommonParam& common_param) {
    return make_instance_impl<QuantizerAdapter<QuantTemp, int8_t>, IOTemp>(param, common_param);
}

#define INSTANTIATE_FLATTEN_CELL(IOTemp, type_suffix, quantizer_class, metric)         \
    template FlattenInterfacePtr create_flatten_cell<quantizer_class<metric>, IOTemp>( \
        const FlattenInterfaceParamPtr&, const IndexCommonParam&);

#define INSTANTIATE_FLATTEN_FOR_METRIC(type_suffix, quantizer_class, metric) \
    FOR_EACH_IO_TYPE(INSTANTIATE_FLATTEN_CELL, type_suffix, quantizer_class, metric)

#define INSTANTIATE_FLATTEN_ALL_METRICS(type_suffix, quantizer_class)                           \
    INSTANTIATE_FLATTEN_FOR_METRIC(type_suffix, quantizer_class, MetricType::METRIC_TYPE_L2SQR) \
    INSTANTIATE_FLATTEN_FOR_METRIC(type_suffix, quantizer_class, MetricType::METRIC_TYPE_IP)    \
    INSTANTIATE_FLATTEN_FOR_METRIC(type_suffix, quantizer_class, MetricType::METRIC_TYPE_COSINE)

FOR_EACH_ALL_QUANTIZER(INSTANTIATE_FLATTEN_ALL_METRICS)

#define INSTANTIATE_ADAPTER_CELL(IOTemp, quantizer_class, metric)                                \
    template FlattenInterfacePtr create_quantizer_adapter_cell<quantizer_class<metric>, IOTemp>( \
        const FlattenInterfaceParamPtr&, const IndexCommonParam&);

#define INSTANTIATE_ADAPTER_FOR_METRIC(metric) \
    FOR_EACH_IO_TYPE(INSTANTIATE_ADAPTER_CELL, ProductQuantizer, metric)

INSTANTIATE_ADAPTER_FOR_METRIC(MetricType::METRIC_TYPE_L2SQR)
INSTANTIATE_ADAPTER_FOR_METRIC(MetricType::METRIC_TYPE_IP)
INSTANTIATE_ADAPTER_FOR_METRIC(MetricType::METRIC_TYPE_COSINE)

}  // namespace creator
}  // namespace vsag
