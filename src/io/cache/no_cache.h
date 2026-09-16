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

#include "io/core/read_request.h"

namespace vsag {

class NoCache {
public:
    [[nodiscard]] constexpr bool
    Enabled() const {
        return false;
    }

    template <typename Backend>
    using Lease = typename Backend::Lease;

    template <typename Backend>
    using Operation = typename Backend::Operation;

    template <typename Backend>
    [[nodiscard]] bool
    ReadAt(const Backend& backend, uint64_t, const ReadRequest& request) const {
        return backend.ReadAt(request.offset, request.size, request.destination);
    }

    template <typename Backend>
    [[nodiscard]] bool
    ReadMany(const Backend& backend, uint64_t, const ReadRequest* requests, uint64_t count) const {
        return backend.ReadMany(requests, count);
    }

    template <typename Backend>
    [[nodiscard]] bool
    ReadManyContiguous(const Backend& backend,
                       uint64_t,
                       uint8_t* destination,
                       const uint64_t* sizes,
                       const uint64_t* offsets,
                       uint64_t count) const {
        return backend.ReadManyContiguous(destination, sizes, offsets, count);
    }

    template <typename Backend>
    [[nodiscard]] Operation<Backend>
    SubmitReads(const Backend& backend,
                uint64_t,
                const ReadRequest* requests,
                uint64_t count) const {
        return backend.SubmitReads(requests, count);
    }

    template <typename Backend>
    [[nodiscard]] Lease<Backend>
    Acquire(const Backend& backend, uint64_t, uint64_t offset, uint64_t size) const {
        return backend.Acquire(offset, size);
    }

    template <typename Backend>
    [[nodiscard]] const uint8_t*
    LegacyRead(const Backend& backend, uint64_t, uint64_t offset, uint64_t size, bool& need_release)
        const {
        return backend.LegacyRead(offset, size, need_release);
    }

    template <typename Backend>
    void
    Release(const Backend& backend, const uint8_t* data) const {
        backend.Release(data);
    }

    template <typename Backend>
    void
    Invalidate(const Backend&, uint64_t, uint64_t) {
    }

    void
    Clear(uint64_t) {
    }

    template <typename Param>
    void
    Configure(const Param&) {
    }

    template <typename Cache>
    void
    SetShared(std::shared_ptr<Cache>, uint64_t = 0) {
    }
};

}  // namespace vsag
