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

#include "io/read_cache/read_cache_parameter.h"

#include "inner_string_params.h"

namespace vsag {

ReadCacheParameter::ReadCacheParameter() : IOParameter(IO_TYPE_VALUE_READ_CACHE) {
}

ReadCacheParameter::ReadCacheParameter(const vsag::JsonType& json)
    : IOParameter(IO_TYPE_VALUE_READ_CACHE) {
    this->FromJson(json);  // NOLINT(clang-analyzer-optin.cplusplus.VirtualCall)
}

void
ReadCacheParameter::FromJson(const JsonType& json) {
    this->original_json_ = json;
    if (json.Contains(READ_CACHE_TOTAL_CACHE_SIZE_KEY)) {
        this->total_cache_size_ = json[READ_CACHE_TOTAL_CACHE_SIZE_KEY].GetUint64();
    }
    if (json.Contains(READ_CACHE_EVICTION_STRATEGY_KEY)) {
        this->eviction_strategy_ = json[READ_CACHE_EVICTION_STRATEGY_KEY].GetString();
        if (this->eviction_strategy_ != "lru") {
            throw VsagException(
                ErrorType::INVALID_ARGUMENT,
                "ReadCache only supports lru eviction_strategy, got: " + this->eviction_strategy_);
        }
    }
    if (json.Contains(READ_CACHE_INNER_IO_TYPE_KEY)) {
        this->inner_io_type_ = json[READ_CACHE_INNER_IO_TYPE_KEY].GetString();
    }
}

JsonType
ReadCacheParameter::ToJson() const {
    JsonType json = this->original_json_;
    json[TYPE_KEY].SetString(IO_TYPE_VALUE_READ_CACHE);
    json[READ_CACHE_TOTAL_CACHE_SIZE_KEY].SetUint64(this->total_cache_size_);
    json[READ_CACHE_EVICTION_STRATEGY_KEY].SetString(this->eviction_strategy_);
    if (not this->inner_io_type_.empty()) {
        json[READ_CACHE_INNER_IO_TYPE_KEY].SetString(this->inner_io_type_);
    }
    return json;
}

IOParamPtr
MakeInnerIOParam(const ReadCacheParameterPtr& cache_param) {
    if (cache_param == nullptr or cache_param->inner_io_type_.empty()) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "ReadCache requires a valid inner_io_type");
    }
    JsonType inner_json = cache_param->original_json_;
    inner_json[TYPE_KEY].SetString(cache_param->inner_io_type_);
    auto inner_param = IOParameter::GetIOParameterByJson(inner_json);
    if (inner_param == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "ReadCache inner_io_type is invalid");
    }
    return inner_param;
}

IOParamPtr
UnwrapReadCacheParam(const IOParamPtr& io_param) {
    auto cache_param = std::dynamic_pointer_cast<ReadCacheParameter>(io_param);
    return cache_param == nullptr ? io_param : MakeInnerIOParam(cache_param);
}

}  // namespace vsag
