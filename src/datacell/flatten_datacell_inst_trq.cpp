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
#include "quantization/transform_quantization/transform_quantizer_parameter.h"

namespace vsag {
namespace creator {

template <typename BottomQuantTemp, MetricType metric, typename IOTemp>
FlattenInterfacePtr
create_trq_cell(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param) {
    using RQ = ResidualQuantizer<BottomQuantTemp, metric>;
    return make_instance_impl<TransformQuantizer<RQ, metric>, IOTemp>(param, common_param);
}

#define INSTANTIATE_TRQ_CELL(IOTemp, type_suffix, quantizer_class, metric)              \
    template FlattenInterfacePtr                                                        \
    create_trq_cell<quantizer_class<vsag::MetricType::METRIC_TYPE_IP>, metric, IOTemp>( \
        const FlattenInterfaceParamPtr&, const IndexCommonParam&);

#define INSTANTIATE_TRQ_FOR_METRIC(type_suffix, quantizer_class, metric) \
    FOR_EACH_IO_TYPE(INSTANTIATE_TRQ_CELL, type_suffix, quantizer_class, metric)

#define INSTANTIATE_TRQ_ALL_METRICS(type_suffix, quantizer_class)                           \
    INSTANTIATE_TRQ_FOR_METRIC(type_suffix, quantizer_class, MetricType::METRIC_TYPE_L2SQR) \
    INSTANTIATE_TRQ_FOR_METRIC(type_suffix, quantizer_class, MetricType::METRIC_TYPE_IP)    \
    INSTANTIATE_TRQ_FOR_METRIC(type_suffix, quantizer_class, MetricType::METRIC_TYPE_COSINE)

FOR_EACH_DENSE_QUANTIZER(INSTANTIATE_TRQ_ALL_METRICS)

}  // namespace creator
}  // namespace vsag
