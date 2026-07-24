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
#include <string>

#include "io/common/io_parameter.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(CacheIOParameter);

class CacheIOParameter : public IOParameter {
public:
    CacheIOParameter();

    explicit CacheIOParameter(const JsonType& json);

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

public:
    uint64_t total_cache_size_{268435456};
    std::string eviction_strategy_{"lru"};
    std::string inner_io_type_;
    JsonType original_json_;
};

}  // namespace vsag
