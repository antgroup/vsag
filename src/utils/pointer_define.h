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
 * @file pointer_define.h
 * @brief Macros for defining smart pointer type aliases.
 */

#pragma once

namespace vsag {

/**
 * @brief Define smart pointer type aliases for a class.
 *
 * This macro generates forward declaration and four pointer type aliases:
 * - ClassNamePtr: std::shared_ptr<ClassName>
 * - ClassNameUPtr: std::unique_ptr<ClassName>
 * - ClassNameConstPtr: std::shared_ptr<const ClassName>
 * - ClassNameConstUPtr: std::unique_ptr<const ClassName>
 *
 * @param class_name The name of the class to define pointers for.
 */
#define DEFINE_POINTER(class_name)                                  \
    class class_name;                                               \
    using class_name##Ptr = std::shared_ptr<class_name>;            \
    using class_name##UPtr = std::unique_ptr<class_name>;           \
    using class_name##ConstPtr = std::shared_ptr<const class_name>; \
    using class_name##ConstUPtr = std::unique_ptr<const class_name>;

/**
 * @brief Define smart pointer type aliases with custom pointer name prefix.
 *
 * Similar to DEFINE_POINTER but allows specifying a different name prefix
 * for the pointer types, useful when the pointer name differs from the class name.
 *
 * @param pointer_name The prefix for pointer type names.
 * @param class_name The name of the class to define pointers for.
 */
#define DEFINE_POINTER2(pointer_name, class_name)                     \
    class class_name;                                                 \
    using pointer_name##Ptr = std::shared_ptr<class_name>;            \
    using pointer_name##UPtr = std::unique_ptr<class_name>;           \
    using pointer_name##ConstPtr = std::shared_ptr<const class_name>; \
    using pointer_name##ConstUPtr = std::unique_ptr<const class_name>;
}  // namespace vsag