

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

#include <cmath>
#include <string>

#include "index/index_common_param.h"
#include "vsag/dataset.h"
#include "vsag/expected.hpp"
#include "vsag_exception.h"

namespace vsag {

template <typename IndexOpParameters>
tl::expected<IndexOpParameters, Error>
try_parse_parameters(const std::string& json_string) {
    try {
        return IndexOpParameters::FromJson(json_string);
    } catch (const VsagException& e) {
        return tl::unexpected<Error>(e.error_);
    } catch (const std::exception& e) {
        return tl::unexpected<Error>(ErrorType::INVALID_ARGUMENT, e.what());
    }
}

template <typename IndexOpParameters>
tl::expected<IndexOpParameters, Error>
try_parse_parameters(JsonType& param_obj, IndexCommonParam index_common_param) {
    try {
        return IndexOpParameters::FromJson(param_obj, index_common_param);
    } catch (const VsagException& e) {
        return tl::unexpected<Error>(e.error_);
    } catch (const std::exception& e) {
        return tl::unexpected<Error>(ErrorType::INVALID_ARGUMENT, e.what());
    }
}

static inline __attribute__((always_inline)) int64_t
ceil_int(const int64_t& value, int64_t base) {
    return ((value + base - 1) / base) * base;
}

std::string
format_map(const std::string& str, const std::unordered_map<std::string, std::string>& mappings);

void
mapping_external_param_to_inner(const JsonType& external_json,
                                ConstParamMap& param_map,
                                JsonType& inner_json);

std::tuple<DatasetPtr, float*, int64_t*>
create_fast_dataset(int64_t dim, Allocator* allocator);

std::vector<int>
select_k_numbers(int64_t n, int k);

uint64_t
next_multiple_of_power_of_two(uint64_t x, uint64_t n);

bool
check_equal_on_string_stream(std::stringstream& s1, std::stringstream& s2);

std::vector<std::string>
split_string(const std::string& str, const char delimiter);

std::string
get_current_time();

template <class T1, class T2>
void
copy_vector(const std::vector<T1>& from, std::vector<T2>& to) {
    static_assert(std::is_same_v<T1, T2> || std::is_convertible_v<T1, T2>);
    to.resize(from.size());
    for (int64_t i = 0; i < from.size(); ++i) {
        to[i] = static_cast<T2>(from[i]);
    }
}

static inline __attribute__((always_inline)) bool
is_approx_zero(const float v) {
    return std::abs(v) < 1e-5;
}

std::string
base64_encode(const std::string& in);

template <typename T>
std::string
base64_encode_obj(T& obj) {
    std::string to_string((char*)&obj, sizeof(obj));
    return base64_encode(to_string);
}

std::string
base64_decode(const std::string& in);

template <typename T>
void
base64_decode_obj(const std::string& in, T& obj) {
    std::string to_string = base64_decode(in);
    memcpy(&obj, to_string.c_str(), sizeof(obj));
}

}  // namespace vsag
