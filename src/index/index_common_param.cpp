
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

#include "index_common_param.h"

#include <fmt/format-inl.h>

#include <nlohmann/json.hpp>

#include "common.h"
#include "vsag/constants.h"

namespace vsag {

IndexCommonParam
IndexCommonParam::CheckAndCreate(const std::string& json_string) {
    IndexCommonParam result;
    JsonType params = JsonType::parse(json_string);

    // Check DataType
    CHECK_ARGUMENT(params.contains(PARAMETER_DTYPE),
                   fmt::format("parameters must contains {}", PARAMETER_DTYPE));
    const auto datatype_obj = params[PARAMETER_DTYPE];
    CHECK_ARGUMENT(datatype_obj.is_string(),
                   fmt::format("parameters[{}] must string type", PARAMETER_DTYPE));
    std::string datatype = datatype_obj;
    if (datatype == DATATYPE_FLOAT32) {
        result.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    } else if (datatype == DATATYPE_INT8) {
        result.data_type_ = DataTypes::DATA_TYPE_INT8;
    } else {
        throw std::invalid_argument(fmt::format("parameters[{}] must in [{}, {}], now is {}",
                                                PARAMETER_DTYPE,
                                                DATATYPE_FLOAT32,
                                                DATATYPE_INT8,
                                                datatype));
    }

    // Check MetricType
    CHECK_ARGUMENT(params.contains(PARAMETER_METRIC_TYPE),
                   fmt::format("parameters must contains {}", PARAMETER_METRIC_TYPE));
    const auto metric_obj = params[PARAMETER_METRIC_TYPE];
    CHECK_ARGUMENT(metric_obj.is_string(),
                   fmt::format("parameters[{}] must string type", PARAMETER_METRIC_TYPE));
    std::string metric = metric_obj;
    if (metric == METRIC_L2) {
        result.metric_ = MetricType::METRIC_TYPE_L2SQR;
    } else if (metric == METRIC_IP) {
        result.metric_ = MetricType::METRIC_TYPE_IP;
    } else if (metric == METRIC_COSINE) {
        result.metric_ = MetricType::METRIC_TYPE_COSINE;
    } else {
        throw std::invalid_argument(fmt::format("parameters[{}] must in [{}, {}, {}], now is {}",
                                                PARAMETER_METRIC_TYPE,
                                                METRIC_L2,
                                                METRIC_IP,
                                                METRIC_COSINE,
                                                metric));
    }

    // Check Dim
    CHECK_ARGUMENT(params.contains(PARAMETER_DIM),
                   fmt::format("parameters must contain {}", PARAMETER_DIM));
    const auto dim_obj = params[PARAMETER_DIM];
    CHECK_ARGUMENT(dim_obj.is_number_integer(),
                   fmt::format("parameters[{}] must be integer type", PARAMETER_DIM));
    int64_t dim = params[PARAMETER_DIM];
    CHECK_ARGUMENT(dim > 0, fmt::format("parameters[{}] must be greater than 0", PARAMETER_DIM));
    result.dim_ = dim;

    return result;
}

}  // namespace vsag
