// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstdint>
#include <future>
#include <vector>

#include "datacell/flatten_optimized_build_interface.h"

namespace vsag {

class HGraph;

class HGraphOptimizedBuildSession {
public:
    explicit HGraphOptimizedBuildSession(HGraph& hgraph);

    HGraphOptimizedBuildSession(const HGraphOptimizedBuildSession&) = delete;
    HGraphOptimizedBuildSession&
    operator=(const HGraphOptimizedBuildSession&) = delete;

    ~HGraphOptimizedBuildSession();

    [[nodiscard]] bool
    Active() const;

    void
    Commit();

private:
    HGraph* hgraph_{nullptr};
    FlattenOptimizedBuildInterfacePtr optimized_build_codes_{nullptr};
};

class HGraphBuildTaskGuard {
public:
    HGraphBuildTaskGuard(std::vector<std::future<void>>& futures, uint64_t capacity);

    HGraphBuildTaskGuard(const HGraphBuildTaskGuard&) = delete;
    HGraphBuildTaskGuard&
    operator=(const HGraphBuildTaskGuard&) = delete;

    ~HGraphBuildTaskGuard();

private:
    std::vector<std::future<void>>& futures_;
};

}  // namespace vsag
