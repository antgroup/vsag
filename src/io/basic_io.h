
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

#include <cstdint>

namespace vsag {

template <typename IOTmpl>
class BasicIO {
public:
    BasicIO<IOTmpl>() = default;

    ~BasicIO() = default;

    inline void
    Write(const uint8_t* data, uint64_t size, uint64_t offset) {
        return cast().WriteImpl(data, size, offset);
    }

    inline bool
    Read(uint8_t* data, uint64_t size, uint64_t offset) const {
        return cast().ReadImpl(data, size, offset);
    }

    [[nodiscard]] inline const uint8_t*
    Read(uint64_t size, uint64_t offset) const {
        return cast().ReadImpl(size, offset);
    }

    inline bool
    MultiRead(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        return cast().MultiReadImpl(datas, sizes, offsets, count);
    }

    inline void
    Prefetch(uint64_t offset, uint64_t cache_line = 64) {
        return cast().PrefetchImpl(offset, cache_line);
    }

private:
    inline IOTmpl&
    cast() {
        return static_cast<IOTmpl&>(*this);
    }

    inline const IOTmpl&
    cast() const {
        return static_cast<const IOTmpl&>(*this);
    }
};
}  // namespace vsag
