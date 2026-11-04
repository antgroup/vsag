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

#include "flatten_interface_factory.h"
#include "index_common_param.h"

namespace vsag {

FlattenInterfacePtr
make_flatten_data_cell_l2(const FlattenInterfaceParamPtr& param,
                          const IndexCommonParam& common_param);

FlattenInterfacePtr
make_flatten_data_cell_ip(const FlattenInterfaceParamPtr& param,
                          const IndexCommonParam& common_param);

FlattenInterfacePtr
make_flatten_data_cell_cosine(const FlattenInterfaceParamPtr& param,
                              const IndexCommonParam& common_param);

// FP16/BF16 input adapters are instantiated in separate translation units.
FlattenInterfacePtr
make_flatten_data_cell_l2_half(const FlattenInterfaceParamPtr& param,
                               const IndexCommonParam& common_param);

FlattenInterfacePtr
make_flatten_data_cell_ip_half(const FlattenInterfaceParamPtr& param,
                               const IndexCommonParam& common_param);

FlattenInterfacePtr
make_flatten_data_cell_cosine_half(const FlattenInterfaceParamPtr& param,
                                   const IndexCommonParam& common_param);

FlattenInterfacePtr
make_flatten_data_cell_hamming(const FlattenInterfaceParamPtr& param,
                               const IndexCommonParam& common_param);

FlattenInterfacePtr
MakeFlattenDataCell(const FlattenInterfaceParamPtr& param, const IndexCommonParam& common_param) {
    const bool half = common_param.data_type_ == DataTypes::DATA_TYPE_FP16 ||
                      common_param.data_type_ == DataTypes::DATA_TYPE_BF16;
    if (common_param.metric_ == MetricType::METRIC_TYPE_L2SQR) {
        return half ? make_flatten_data_cell_l2_half(param, common_param)
                    : make_flatten_data_cell_l2(param, common_param);
    }
    if (common_param.metric_ == MetricType::METRIC_TYPE_IP) {
        return half ? make_flatten_data_cell_ip_half(param, common_param)
                    : make_flatten_data_cell_ip(param, common_param);
    }
    if (common_param.metric_ == MetricType::METRIC_TYPE_COSINE) {
        return half ? make_flatten_data_cell_cosine_half(param, common_param)
                    : make_flatten_data_cell_cosine(param, common_param);
    }
    if (common_param.metric_ == MetricType::METRIC_TYPE_HAMMING) {
        return make_flatten_data_cell_hamming(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
