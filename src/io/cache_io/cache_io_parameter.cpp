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

#include "io/cache_io/cache_io_parameter.h"

#include "inner_string_params.h"

namespace vsag {

CacheIOParameter::CacheIOParameter() : IOParameter(IO_TYPE_VALUE_CACHE_IO) {
}

CacheIOParameter::CacheIOParameter(const vsag::JsonType& json)
    : IOParameter(IO_TYPE_VALUE_CACHE_IO) {
    this->FromJson(json);  // NOLINT(clang-analyzer-optin.cplusplus.VirtualCall)
}

void
CacheIOParameter::FromJson(const JsonType& json) {
    this->original_json_ = json;
    if (json.Contains(CACHE_IO_TOTAL_CACHE_SIZE_KEY)) {
        this->total_cache_size_ = json[CACHE_IO_TOTAL_CACHE_SIZE_KEY].GetUint64();
    }
    if (json.Contains(CACHE_IO_EVICTION_STRATEGY_KEY)) {
        this->eviction_strategy_ = json[CACHE_IO_EVICTION_STRATEGY_KEY].GetString();
        if (this->eviction_strategy_ != "lru") {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                "CacheIO only supports lru eviction_strategy, got: " + this->eviction_strategy_);
        }
    }
    if (json.Contains(CACHE_IO_INNER_IO_TYPE_KEY)) {
        this->inner_io_type_ = json[CACHE_IO_INNER_IO_TYPE_KEY].GetString();
    }
}

JsonType
CacheIOParameter::ToJson() const {
    JsonType json = this->original_json_;
    json[TYPE_KEY].SetString(IO_TYPE_VALUE_CACHE_IO);
    json[CACHE_IO_TOTAL_CACHE_SIZE_KEY].SetUint64(this->total_cache_size_);
    json[CACHE_IO_EVICTION_STRATEGY_KEY].SetString(this->eviction_strategy_);
    if (not this->inner_io_type_.empty()) {
        json[CACHE_IO_INNER_IO_TYPE_KEY].SetString(this->inner_io_type_);
    }
    return json;
}

}  // namespace vsag
