
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

#include <tsl/robin_map.h>
#include <tsl/robin_set.h>

#include <cstdint>
#include <deque>
#include <queue>
#include <unordered_map>

#include "impl/allocator/allocator_wrapper.h"
#include "json_wrapper.h"

namespace vsag {

using InnerIdType = uint32_t;  // inner id's type; index's vector count may less than 2^31 - 1
using LabelType = int64_t;     // external id's type

using JsonType = JsonWrapper;  // alias for nlohmann::json type
using BucketIdType = int32_t;

template <typename T>
using UnorderedSet = tsl::robin_set<T, std::hash<T>, std::equal_to<T>, vsag::AllocatorWrapper<T>>;

template <typename T>
using Vector = std::vector<T, vsag::AllocatorWrapper<T>>;

template <typename T>
using Deque = std::deque<T, vsag::AllocatorWrapper<T>>;

template <typename KeyType, typename ValType>
using UnorderedMap = tsl::robin_map<KeyType,
                                    ValType,
                                    std::hash<KeyType>,
                                    std::equal_to<KeyType>,
                                    vsag::AllocatorWrapper<std::pair<const KeyType, ValType>>>;

template <typename KeyType, typename ValType>
using PGUnorderedMap = tsl::robin_pg_map<KeyType,
                                         ValType,
                                         std::hash<KeyType>,
                                         std::equal_to<KeyType>,
                                         vsag::AllocatorWrapper<std::pair<const KeyType, ValType>>>;

template <typename T, typename... Args>
inline auto
AllocateShared(Allocator* allocator, Args&&... args) {
    return std::allocate_shared<T>(AllocatorWrapper<T>(allocator), std::forward<Args>(args)...);
}

using ConstParamMap = const std::unordered_multimap<std::string, std::vector<std::string>>;

using IdFilterFuncType = std::function<bool(int64_t)>;

struct CompareByFirst {
    constexpr bool
    operator()(std::pair<float, InnerIdType> const& a,
               std::pair<float, InnerIdType> const& b) const noexcept {
        return a.first < b.first;
    }
};

using MaxHeap = std::priority_queue<std::pair<float, InnerIdType>,
                                    Vector<std::pair<float, InnerIdType>>,
                                    CompareByFirst>;

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
