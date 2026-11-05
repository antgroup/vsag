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

#include <functional>
#include <utility>

#include "impl/allocator/allocator_wrapper.h"

namespace vsag {

template <typename T>
using UnorderedSet = tsl::robin_set<T, std::hash<T>, std::equal_to<T>, vsag::AllocatorWrapper<T>>;

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

}  // namespace vsag
