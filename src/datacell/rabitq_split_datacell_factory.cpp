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

#include "rabitq_split_datacell_factory.h"

#include "index_common_param.h"

namespace vsag {

FlattenInterfacePtr
MakeRaBitQSplitDataCellL2(const FlattenInterfaceParamPtr& param,
                          const IndexCommonParam& common_param);

FlattenInterfacePtr
MakeRaBitQSplitDataCellIP(const FlattenInterfaceParamPtr& param,
                          const IndexCommonParam& common_param);

FlattenInterfacePtr
MakeRaBitQSplitDataCellCosine(const FlattenInterfaceParamPtr& param,
                              const IndexCommonParam& common_param);

FlattenInterfacePtr
MakeRaBitQSplitDataCell(const FlattenInterfaceParamPtr& param,
                        const IndexCommonParam& common_param) {
    if (common_param.metric_ == MetricType::METRIC_TYPE_L2SQR) {
        return MakeRaBitQSplitDataCellL2(param, common_param);
    }
    if (common_param.metric_ == MetricType::METRIC_TYPE_IP) {
        return MakeRaBitQSplitDataCellIP(param, common_param);
    }
    if (common_param.metric_ == MetricType::METRIC_TYPE_COSINE) {
        return MakeRaBitQSplitDataCellCosine(param, common_param);
    }
    return nullptr;
}

}  // namespace vsag
