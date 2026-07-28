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
#include <memory>

#include "basic_types.h"
#include "quantization/computer.h"
#include "utils/pointer_define.h"

namespace vsag {

class SafeThreadPool;

struct FlattenOptimizedBuildContext {
    std::shared_ptr<SafeThreadPool> thread_pool{nullptr};
    uint64_t thread_count{1};
};

DEFINE_POINTER(FlattenOptimizedBuildInterface);

class FlattenOptimizedBuildInterface {
public:
    virtual ~FlattenOptimizedBuildInterface() = default;

    virtual bool
    BeginOptimizedBuild(const FlattenOptimizedBuildContext& context) = 0;

    virtual void
    FinalizeOptimizedBuild() = 0;

    virtual void
    AbortOptimizedBuild() noexcept = 0;

    [[nodiscard]] virtual bool
    IsOptimizedBuildActive() const = 0;

    virtual ComputerInterfacePtr
    FactoryComputerForBuild(const void* query, InnerIdType id) = 0;
};

}  // namespace vsag
