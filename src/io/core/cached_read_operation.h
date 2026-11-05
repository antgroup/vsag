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

#include <utility>
#include <variant>

#include "io/core/read_operation.h"

namespace vsag {

template <typename BackendOperation>
class CachedReadOperation {
public:
    explicit CachedReadOperation(bool result)
        : storage_(std::in_place_index<1>, ImmediateOperation(result)) {
    }

    explicit CachedReadOperation(BackendOperation operation)
        : storage_(std::in_place_index<0>, std::move(operation)) {
    }

    explicit CachedReadOperation(ImmediateOperation operation)
        : storage_(std::in_place_index<1>, std::move(operation)) {
    }

    CachedReadOperation(const CachedReadOperation&) = delete;
    CachedReadOperation&
    operator=(const CachedReadOperation&) = delete;

    CachedReadOperation(CachedReadOperation&&) noexcept = default;
    CachedReadOperation&
    operator=(CachedReadOperation&&) noexcept = default;

    [[nodiscard]] bool
    Poll() const {
        return std::visit([](const auto& operation) { return operation.Poll(); }, storage_);
    }

    [[nodiscard]] bool
    Wait() const {
        return std::visit([](const auto& operation) { return operation.Wait(); }, storage_);
    }

    void
    Cancel() {
        std::visit([](auto& operation) { operation.Cancel(); }, storage_);
    }

private:
    std::variant<BackendOperation, ImmediateOperation> storage_;
};

}  // namespace vsag
