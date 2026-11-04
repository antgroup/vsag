
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

#include <type_traits>

namespace vsag {
template <typename...>
using void_t = void;

template <template <typename...> class Op, typename, typename... Args>
struct Detector : std::false_type {};

template <template <typename...> class Op, typename... Args>
struct Detector<Op, void_t<Op<Args...>>, Args...> : std::true_type {
    using type = Op<Args...>;
};

template <template <typename...> class Op, typename... Args>
constexpr bool is_detected_v = Detector<Op, void, Args...>::value;

#define GENERATE_HAS_MEMBER_FUNCTION(FuncName, ReturnType, ...)                   \
    template <typename T>                                                         \
    using has_##FuncName##_t = decltype(std::declval<T>().FuncName(__VA_ARGS__)); \
                                                                                  \
    template <typename T>                                                         \
    struct Has##FuncName                                                          \
        : std::conjunction<                                                       \
              Detector<has_##FuncName##_t, void, T>,                              \
              std::is_same<typename Detector<has_##FuncName##_t, void, T>::type, ReturnType>> {};

#define GENERATE_HAS_STATIC_CLASS_FUNCTION(FuncName, ReturnType, ...)                   \
    template <typename T>                                                               \
    using has_static_##FuncName##_t = decltype(T::FuncName(__VA_ARGS__));               \
                                                                                        \
    template <typename T>                                                               \
    struct HasStatic##FuncName                                                          \
        : std::conjunction<                                                             \
              Detector<has_static_##FuncName##_t, void, T>,                             \
              std::is_same<typename Detector<has_static_##FuncName##_t, void, T>::type, \
                           ReturnType>> {};

}  // namespace vsag
