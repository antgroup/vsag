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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "basic_types.h"
#include "impl/allocator/allocator_wrapper.h"

namespace vsag {

class Allocator;

template <typename T, typename... Args>
inline auto
AllocateShared(Allocator* allocator, Args&&... args) {
    return std::allocate_shared<T>(AllocatorWrapper<T>(allocator), std::forward<Args>(args)...);
}

using ConstParamMap = const std::unordered_multimap<std::string, std::vector<std::string>>;

using IdFilterFuncType = std::function<bool(LabelType)>;

template <typename Ref>
struct lvalue_or_rvalue {
    Ref&& ref;

    template <typename Arg>
    constexpr lvalue_or_rvalue(Arg&& arg) noexcept : ref(std::move(arg)) {
    }

    constexpr
    operator Ref&() const& noexcept {
        return ref;
    }
    constexpr
    operator Ref&&() const& noexcept {
        return std::move(ref);
    }
    constexpr Ref&
    operator*() const noexcept {
        return ref;
    }
    constexpr Ref*
    operator->() const noexcept {
        return &ref;
    }
};

}  // namespace vsag
