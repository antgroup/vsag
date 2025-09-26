
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

#pragma once

#include <cstdint>

#include "data_type.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "metric_type.h"
#include "typing.h"
#include "vsag/resource.h"

namespace vsag {
class IndexCommonParam {
public:
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};
    int64_t dim_{0};
    int64_t extra_info_size_{0};
    std::shared_ptr<Allocator> allocator_{nullptr};
    std::shared_ptr<SafeThreadPool> thread_pool_{nullptr};

    // FIXME(wxyu): this option is used for special purposes, like compatibility testing
    bool use_old_serial_format_{false};

    static IndexCommonParam
    CheckAndCreate(JsonType& params, const std::shared_ptr<Resource>& resource);

    IndexCommonParam&
    operator=(const IndexCommonParam& other) {
        if (this != &other) {
            metric_ = other.metric_;
            data_type_ = other.data_type_;
            dim_ = other.dim_;
            extra_info_size_ = other.extra_info_size_;
            allocator_ = other.allocator_;
            thread_pool_ = other.thread_pool_;
        }
        return *this;
    }
};
}  // namespace vsag
