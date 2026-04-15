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

#include "index_common_param.h"
#include "vsag/dataset.h"
#include "vsag/expected.hpp"
#include "vsag_exception.h"

namespace vsag {

/**
 * @file util_functions.h
 * @brief Utility functions for parameter parsing, string formatting, and data operations.
 */

/**
 * @brief Parses JSON string into IndexOpParameters with error handling.
 * @tparam IndexOpParameters The parameter type to parse into.
 * @param json_string JSON string to parse.
 * @return Parsed parameters on success, error on failure.
 */
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

/**
 * @brief Parses JSON object into IndexOpParameters with error handling.
 * @tparam IndexOpParameters The parameter type to parse into.
 * @param param_obj JSON object to parse.
 * @param index_common_param Common index parameters.
 * @return Parsed parameters on success, error on failure.
 */
template <typename IndexOpParameters>
tl::expected<IndexOpParameters, Error>
try_parse_parameters(JsonType param_obj, IndexCommonParam index_common_param) {
    try {
        return IndexOpParameters::FromJson(param_obj, index_common_param);
    } catch (const VsagException& e) {
        return tl::unexpected<Error>(e.error_);
    } catch (const std::exception& e) {
        return tl::unexpected<Error>(ErrorType::INVALID_ARGUMENT, e.what());
    }
}

/**
 * @brief Aligns a value up to a multiple of base.
 * @param value Value to align.
 * @param base Alignment base (must be positive).
 * @return Aligned value.
 */
static inline __attribute__((always_inline)) int64_t
align_up(const int64_t& value, int64_t base) {
    return ((value + base - 1) / base) * base;
}

/**
 * @brief Formats a string by replacing placeholders with values from a map.
 * @param str String with placeholders in format ${key}.
 * @param mappings Map of key-value pairs for substitution.
 * @return Formatted string with placeholders replaced.
 */
std::string
format_map(const std::string& str, const std::unordered_map<std::string, std::string>& mappings);

/**
 * @brief Maps external parameter names to internal names.
 * @param external_json External JSON with original parameter names.
 * @param param_map Mapping from external to internal parameter names.
 * @param inner_json Output JSON with internal parameter names.
 */
void
mapping_external_param_to_inner(const JsonType& external_json,
                                ConstParamMap& param_map,
                                JsonType& inner_json);

/**
 * @brief Creates a fast dataset for efficient operations.
 * @param dim Dimensionality of vectors.
 * @param allocator Allocator for memory management.
 * @return Tuple of dataset pointer, float buffer, and int64 buffer.
 */
std::tuple<DatasetPtr, float*, int64_t*>
create_fast_dataset(int64_t dim, Allocator* allocator);

/**
 * @brief Selects k distinct random numbers from range [0, n).
 * @param n Upper bound of the range (exclusive).
 * @param k Number of elements to select.
 * @return Vector of k selected numbers.
 */
std::vector<int>
select_k_numbers(int64_t n, int k);

/**
 * @brief Computes the next multiple of 2^n.
 * @param x Input value.
 * @param n Power of 2 to align to (result is multiple of 2^n).
 * @return Smallest multiple of 2^n greater than or equal to x.
 */
uint64_t
next_multiple_of_power_of_two(uint64_t x, uint64_t n);

/**
 * @brief Checks if two string streams have equal content.
 * @param s1 First string stream.
 * @param s2 Second string stream.
 * @return True if contents are equal, false otherwise.
 */
bool
check_equal_on_string_stream(std::stringstream& s1, std::stringstream& s2);

/**
 * @brief Splits a string by a delimiter.
 * @param str String to split.
 * @param delimiter Delimiter character.
 * @return Vector of substrings.
 */
std::vector<std::string>
split_string(const std::string& str, const char delimiter);

/**
 * @brief Gets the current time as a formatted string.
 * @return Current time string in a standard format.
 */
std::string
get_current_time();

/**
 * @brief Copies elements from one vector to another with type conversion.
 * @tparam T1 Source element type.
 * @tparam T2 Destination element type.
 * @param from Source vector.
 * @param to Destination vector (will be resized).
 */
template <class T1, class T2>
void
copy_vector(const std::vector<T1>& from, std::vector<T2>& to) {
    static_assert(std::is_same_v<T1, T2> || std::is_convertible_v<T1, T2>);
    to.resize(from.size());
    for (int64_t i = 0; i < from.size(); ++i) {
        to[i] = static_cast<T2>(from[i]);
    }
}

/**
 * @brief Checks if a float value is approximately zero.
 * @param v Value to check.
 * @return True if |v| < 1e-5, false otherwise.
 */
static inline __attribute__((always_inline)) bool
is_approx_zero(const float v) {
    return std::abs(v) < 1e-5;
}

/**
 * @brief Encodes a string to Base64.
 * @param in Input string.
 * @return Base64 encoded string.
 */
std::string
base64_encode(const std::string& in);

/**
 * @brief Encodes a binary object to Base64.
 * @tparam T Object type (must be trivially copyable).
 * @param obj Object to encode.
 * @return Base64 encoded string.
 */
template <typename T>
std::string
base64_encode_obj(T& obj) {
    std::string to_string((char*)&obj, sizeof(obj));
    return base64_encode(to_string);
}

/**
 * @brief Decodes a Base64 string.
 * @param in Base64 encoded string.
 * @return Decoded string.
 */
std::string
base64_decode(const std::string& in);

/**
 * @brief Decodes a Base64 string into a binary object.
 * @tparam T Object type (must be trivially copyable).
 * @param in Base64 encoded string.
 * @param obj Output object to store decoded data.
 */
template <typename T>
void
base64_decode_obj(const std::string& in, T& obj) {
    std::string to_string = base64_decode(in);
    memcpy(&obj, to_string.c_str(), sizeof(obj));
}

/**
 * @brief Extracts vector pointers and data size from a dataset.
 * @param type Data type of vectors.
 * @param dim Dimensionality of vectors.
 * @param base Dataset containing the vectors.
 * @param vectors_ptr Output pointer to vector data.
 * @param data_size_ptr Output pointer to data size.
 */
void
get_vectors(DataTypes type,
            int64_t dim,
            const vsag::DatasetPtr& base,
            void** vectors_ptr,
            uint64_t* data_size_ptr);

/**
 * @brief Sets vector data into a dataset.
 * @param type Data type of vectors.
 * @param dim Dimensionality of vectors.
 * @param base Dataset to set data into.
 * @param vectors_ptr Pointer to vector data.
 * @param num_element Number of elements to set.
 */
void
set_dataset(DataTypes type,
            int64_t dim,
            const DatasetPtr& base,
            const void* vectors_ptr,
            uint32_t num_element);

/**
 * @brief Samples training data from a larger dataset.
 * @param data Source dataset.
 * @param total_elements Total number of elements in source.
 * @param dim Dimensionality of vectors.
 * @param train_sample_count Number of samples to extract.
 * @param allocator Allocator for memory management (optional).
 * @return Dataset containing sampled training data.
 */
DatasetPtr
sample_train_data(const DatasetPtr& data,
                  int64_t total_elements,
                  int64_t dim,
                  int64_t train_sample_count,
                  Allocator* allocator = nullptr);

}  // namespace vsag