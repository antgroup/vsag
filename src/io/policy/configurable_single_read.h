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
#include <utility>

#include "io/backend/posix_file.h"
#include "io/core/io_environment.h"
#include "io/core/read_lease.h"
#include "io/policy/buffered_single_read.h"
#include "io/policy/direct_single_read.h"
#include "vsag/allocator.h"
#include "vsag_exception.h"

namespace vsag {

class ConfigurableSingleRead {
public:
    using Lease = ConfigurableLease;

    ConfigurableSingleRead() = default;

    explicit ConfigurableSingleRead(const IOEnvironment& environment)
        : direct_read_(environment.direct_read) {
    }

    [[nodiscard]] bool
    ReadAt(const PosixFile& file, uint64_t offset, uint64_t size, uint8_t* destination) const {
        if (direct_read_) {
            return direct_.ReadAt(file, offset, size, destination);
        }
        return buffered_.ReadAt(file, offset, size, destination);
    }

    [[nodiscard]] Lease
    Acquire(const PosixFile& file, Allocator* allocator, uint64_t offset, uint64_t size) const {
        if (size == 0) {
            return Lease{};
        }
        if (direct_read_) {
            auto buffer = direct_.AcquireBuffer(file, offset, size);
            const uint8_t* data = buffer.Data();
            uint8_t* base = buffer.ReleaseBase();
            return Lease(data, size, ConfigurableOwner::AlignedOwned(base));
        }

        auto* data = static_cast<uint8_t*>(allocator->Allocate(size));
        if (data == nullptr) {
            throw VsagException(ErrorType::NO_ENOUGH_MEMORY,
                                "ConfigurableSingleRead allocation failed");
        }
        ConfigurableOwner owner = ConfigurableOwner::AllocatorOwned(allocator, data);
        (void)buffered_.ReadAt(file, offset, size, data);
        return Lease(data, size, std::move(owner));
    }

    [[nodiscard]] const uint8_t*
    LegacyRead(const PosixFile& file,
               Allocator* allocator,
               uint64_t offset,
               uint64_t size,
               bool& need_release) const {
        if (direct_read_) {
            return direct_.LegacyRead(file, allocator, offset, size, need_release);
        }
        return buffered_.LegacyRead(file, allocator, offset, size, need_release);
    }

    void
    Release(Allocator* allocator, const uint8_t* data) const {
        if (direct_read_) {
            direct_.Release(allocator, data);
        } else {
            buffered_.Release(allocator, data);
        }
    }

    [[nodiscard]] bool
    DirectReadEnabled() const {
        return direct_read_;
    }

private:
    bool direct_read_{false};
    BufferedSingleRead buffered_;
    DirectSingleRead direct_;
};

}  // namespace vsag
