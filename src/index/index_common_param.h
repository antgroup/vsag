
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
#include "metric_type.h"
#include "typing.h"
#include "vsag/allocator.h"

namespace vsag {
class IndexCommonParam {
public:
    MetricType metric_{MetricType::METRIC_TYPE_L2SQR};
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};
    int64_t dim_{0};
    Allocator* allocator_{nullptr};

    static IndexCommonParam
    CheckAndCreate(const std::string& json_string);
};
}  // namespace vsag
