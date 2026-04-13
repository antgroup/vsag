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

/**
 * @file function_exists_check.h
 * @brief Template metaprogramming utilities for compile-time member function detection.
 */

#pragma once

#include <type_traits>

namespace vsag {
/// Void type alias for SFINAE purposes.
template <typename...>
using void_t = void;

/**
 * @brief Detector template for compile-time feature detection.
 *
 * Primary template that inherits from std::false_type when the operation
 * Op<Args...> is not valid.
 *
 * @tparam Op Template template parameter representing the operation to detect.
 * @tparam AlwaysVoid Always void when used in primary template.
 * @tparam Args Arguments to pass to the operation.
 */
template <template <typename...> class Op, typename, typename... Args>
struct detector : std::false_type {};

/**
 * @brief Specialization of detector when the operation is valid.
 *
 * When Op<Args...> is a valid type, this specialization is selected
 * and inherits from std::true_type, also providing the detected type.
 *
 * @tparam Op Template template parameter representing the operation to detect.
 * @tparam Args Arguments to pass to the operation.
 */
template <template <typename...> class Op, typename... Args>
struct detector<Op, void_t<Op<Args...>>, Args...> : std::true_type {
    using type = Op<Args...>;  ///< The detected type
};

/**
 * @brief Variable template for checking if an operation is detected.
 *
 * @tparam Op Template template parameter representing the operation to detect.
 * @tparam Args Arguments to pass to the operation.
 * @return true if Op<Args...> is a valid type, false otherwise.
 */
template <template <typename...> class Op, typename... Args>
constexpr bool is_detected_v = detector<Op, void, Args...>::value;

/**
 * @brief Macro to generate a trait for detecting a member function.
 *
 * Creates a has_FuncName<T> trait that checks if type T has a member
 * function named FuncName with the specified return type and parameters.
 *
 * @param FuncName The name of the member function to detect.
 * @param ReturnType The expected return type of the function.
 * @param ... The parameter types (passed to the function call expression).
 *
 * Example usage:
 * @code
 * GENERATE_HAS_MEMBER_FUNCTION(size, size_t)
 * // Now has_size<T>::value is true if T has a size() member returning size_t
 * @endcode
 */
#define GENERATE_HAS_MEMBER_FUNCTION(FuncName, ReturnType, ...)                   \
    template <typename T>                                                         \
    using has_##FuncName##_t = decltype(std::declval<T>().FuncName(__VA_ARGS__)); \
                                                                                  \
    template <typename T>                                                         \
    struct has_##FuncName                                                         \
        : std::conjunction<                                                       \
              detector<has_##FuncName##_t, void, T>,                              \
              std::is_same<typename detector<has_##FuncName##_t, void, T>::type, ReturnType>> {};

/**
 * @brief Macro to generate a trait for detecting a static class function.
 *
 * Creates a has_static_FuncName<T> trait that checks if type T has a static
 * member function named FuncName with the specified return type and parameters.
 *
 * @param FuncName The name of the static member function to detect.
 * @param ReturnType The expected return type of the function.
 * @param ... The parameter types (passed to the function call expression).
 *
 * Example usage:
 * @code
 * GENERATE_HAS_STATIC_CLASS_FUNCTION(Create, std::shared_ptr<T>)
 * // Now has_static_Create<T>::value is true if T has a static Create() method
 * @endcode
 */
#define GENERATE_HAS_STATIC_CLASS_FUNCTION(FuncName, ReturnType, ...)                   \
    template <typename T>                                                               \
    using has_static_##FuncName##_t = decltype(T::FuncName(__VA_ARGS__));               \
                                                                                        \
    template <typename T>                                                               \
    struct has_static_##FuncName                                                        \
        : std::conjunction<                                                             \
              detector<has_static_##FuncName##_t, void, T>,                             \
              std::is_same<typename detector<has_static_##FuncName##_t, void, T>::type, \
                           ReturnType>> {};

}  // namespace vsag