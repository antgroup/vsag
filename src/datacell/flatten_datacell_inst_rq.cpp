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
#include "quantization/quantizer_headers.h"

namespace vsag {
namespace creator {

template <typename BottomQuantTemp, MetricType metric, typename IOTemp>
FlattenInterfacePtr
create_rq_cell(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param) {
    return make_instance_impl<ResidualQuantizer<BottomQuantTemp, metric>, IOTemp>(param,
                                                                                  common_param);
}

#define INSTANTIATE_RQ_CELL(IOTemp, type_suffix, quantizer_class, metric)              \
    template FlattenInterfacePtr                                                       \
    create_rq_cell<quantizer_class<vsag::MetricType::METRIC_TYPE_IP>, metric, IOTemp>( \
        const FlattenInterfaceParamPtr&, const IndexCommonParam&);

#define INSTANTIATE_RQ_FOR_METRIC(type_suffix, quantizer_class, metric) \
    FOR_EACH_IO_TYPE(INSTANTIATE_RQ_CELL, type_suffix, quantizer_class, metric)

#define INSTANTIATE_RQ_ALL_METRICS(type_suffix, quantizer_class)                           \
    INSTANTIATE_RQ_FOR_METRIC(type_suffix, quantizer_class, MetricType::METRIC_TYPE_L2SQR) \
    INSTANTIATE_RQ_FOR_METRIC(type_suffix, quantizer_class, MetricType::METRIC_TYPE_IP)    \
    INSTANTIATE_RQ_FOR_METRIC(type_suffix, quantizer_class, MetricType::METRIC_TYPE_COSINE)

FOR_EACH_DENSE_QUANTIZER(INSTANTIATE_RQ_ALL_METRICS)

}  // namespace creator
}  // namespace vsag
