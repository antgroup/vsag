
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

namespace vsag {

#define DEFINE_POINTER(class_name)                                  \
    class class_name;                                               \
    using class_name##Ptr = std::shared_ptr<class_name>;            \
    using class_name##UPtr = std::unique_ptr<class_name>;           \
    using class_name##ConstPtr = std::shared_ptr<const class_name>; \
    using class_name##ConstUPtr = std::unique_ptr<const class_name>;

#define DEFINE_POINTER2(pointer_name, class_name)                     \
    class class_name;                                                 \
    using pointer_name##Ptr = std::shared_ptr<class_name>;            \
    using pointer_name##UPtr = std::unique_ptr<class_name>;           \
    using pointer_name##ConstPtr = std::shared_ptr<const class_name>; \
    using pointer_name##ConstUPtr = std::unique_ptr<const class_name>;
}  // namespace vsag
