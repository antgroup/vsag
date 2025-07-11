
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

#include <fmt/format.h>

#include "common.h"
#include "vsag/constants.h"

namespace vsag {

inline void
fill_datatype(IndexCommonParam& result, JsonType::const_reference datatype_obj) {
    CHECK_ARGUMENT(datatype_obj.is_string(),
                   fmt::format("parameters[{}] must string type", PARAMETER_DTYPE));
    std::string datatype = datatype_obj;
    if (datatype == DATATYPE_FLOAT32) {
        result.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    } else if (datatype == DATATYPE_INT8) {
        result.data_type_ = DataTypes::DATA_TYPE_INT8;
    } else if (datatype == DATATYPE_SPARSE) {
        result.data_type_ = DataTypes::DATA_TYPE_SPARSE;
    } else {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("parameters[{}] must in [{}, {}, {}], now is {}",
                                        PARAMETER_DTYPE,
                                        DATATYPE_FLOAT32,
                                        DATATYPE_INT8,
                                        DATATYPE_SPARSE,
                                        datatype));
    }
}

inline void
fill_metrictype(IndexCommonParam& result, JsonType::const_reference metric_obj) {
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
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("parameters[{}] must in [{}, {}, {}], now is {}",
                                        PARAMETER_METRIC_TYPE,
                                        METRIC_L2,
                                        METRIC_IP,
                                        METRIC_COSINE,
                                        metric));
    }
}

inline void
fill_dim(IndexCommonParam& result, JsonType::const_reference dim_obj) {
    CHECK_ARGUMENT(dim_obj.is_number_integer(),
                   fmt::format("parameters[{}] must be integer type", PARAMETER_DIM));
    int64_t dim = dim_obj.get<int64_t>();
    CHECK_ARGUMENT(dim > 0, fmt::format("parameters[{}] must be greater than 0", PARAMETER_DIM));
    result.dim_ = dim;
}

inline void
fill_extra_info_size(IndexCommonParam& result, JsonType::const_reference extra_info_size_obj) {
    CHECK_ARGUMENT(extra_info_size_obj.is_number_integer(),
                   fmt::format("parameters[{}] must be integer type", EXTRA_INFO_SIZE));
    int64_t extra_info_size = extra_info_size_obj.get<int64_t>();
    result.extra_info_size_ = extra_info_size;
}

IndexCommonParam
IndexCommonParam::CheckAndCreate(JsonType& params, const std::shared_ptr<Resource>& resource) {
    IndexCommonParam result;
    result.allocator_ = resource->GetAllocator();
    result.thread_pool_ = std::dynamic_pointer_cast<SafeThreadPool>(resource->GetThreadPool());

    // Check and Fill DataType
    CHECK_ARGUMENT(params.contains(PARAMETER_DTYPE),
                   fmt::format("parameters must contains {}", PARAMETER_DTYPE));
    const auto datatype_obj = params[PARAMETER_DTYPE];
    fill_datatype(result, datatype_obj);

    // Check and Fill MetricType
    CHECK_ARGUMENT(params.contains(PARAMETER_METRIC_TYPE),
                   fmt::format("parameters must contains {}", PARAMETER_METRIC_TYPE));
    const auto metric_obj = params[PARAMETER_METRIC_TYPE];
    fill_metrictype(result, metric_obj);

    // Check and Fill Dim
    CHECK_ARGUMENT(params.contains(PARAMETER_DIM),
                   fmt::format("parameters must contain {}", PARAMETER_DIM));
    const auto dim_obj = params[PARAMETER_DIM];
    fill_dim(result, dim_obj);

    if (params.contains(EXTRA_INFO_SIZE)) {
        const auto extra_info_size_obj = params[EXTRA_INFO_SIZE];
        fill_extra_info_size(result, extra_info_size_obj);
    }

    return result;
}

}  // namespace vsag
