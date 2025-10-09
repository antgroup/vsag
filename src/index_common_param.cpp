
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

constexpr static const int64_t MAX_DIM_SPARSE = 4096;

static void
fill_datatype(IndexCommonParam& result, const JsonType& datatype_obj) {
    CHECK_ARGUMENT(datatype_obj.IsString(),
                   fmt::format("parameters[{}] must string type", PARAMETER_DTYPE));
    std::string datatype = datatype_obj.GetString();
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
fill_metrictype(IndexCommonParam& result, const JsonType& metric_obj) {
    CHECK_ARGUMENT(metric_obj.IsString(),
                   fmt::format("parameters[{}] must string type", PARAMETER_METRIC_TYPE));
    std::string metric = metric_obj.GetString();
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
fill_dim(IndexCommonParam& result, const JsonType& dim_obj) {
    CHECK_ARGUMENT(dim_obj.IsNumberInteger(),
                   fmt::format("parameters[{}] must be integer type", PARAMETER_DIM));
    int64_t dim = dim_obj.GetInt();
    CHECK_ARGUMENT(dim > 0, fmt::format("parameters[{}] must be greater than 0", PARAMETER_DIM));
    result.dim_ = dim;
}

inline void
fill_extra_info_size(IndexCommonParam& result, const JsonType& extra_info_size_obj) {
    CHECK_ARGUMENT(extra_info_size_obj.IsNumberInteger(),
                   fmt::format("parameters[{}] must be integer type", EXTRA_INFO_SIZE));
    int64_t extra_info_size = extra_info_size_obj.GetInt();
    result.extra_info_size_ = extra_info_size;
}

IndexCommonParam
IndexCommonParam::CheckAndCreate(JsonType& params, const std::shared_ptr<Resource>& resource) {
    IndexCommonParam result;
    result.allocator_ = resource->GetAllocator();
    result.thread_pool_ = std::dynamic_pointer_cast<SafeThreadPool>(resource->GetThreadPool());

    // Check and Fill DataType
    CHECK_ARGUMENT(params.Contains(PARAMETER_DTYPE),
                   fmt::format("parameters must contains {}", PARAMETER_DTYPE));
    fill_datatype(result, params[PARAMETER_DTYPE]);

    // Check and Fill MetricType
    CHECK_ARGUMENT(params.Contains(PARAMETER_METRIC_TYPE),
                   fmt::format("parameters must contains {}", PARAMETER_METRIC_TYPE));
    fill_metrictype(result, params[PARAMETER_METRIC_TYPE]);

    // Check and Fill Dim
    if (params.Contains(PARAMETER_DIM)) {
        fill_dim(result, params[PARAMETER_DIM]);
    } else {
        if (result.data_type_ != DataTypes::DATA_TYPE_SPARSE) {
            throw vsag::VsagException(ErrorType::INVALID_ARGUMENT,
                                      fmt::format("parameters must contain {}", PARAMETER_DIM));
        }
        result.dim_ = MAX_DIM_SPARSE;
    }

    if (params.Contains(EXTRA_INFO_SIZE)) {
        fill_extra_info_size(result, params[EXTRA_INFO_SIZE]);
    }

    if (params.Contains(PARAMETER_USE_OLD_SERIAL_FORMAT) and
        params[PARAMETER_USE_OLD_SERIAL_FORMAT].GetBool()) {
        result.use_old_serial_format_ = true;
    }

    return result;
}

}  // namespace vsag
