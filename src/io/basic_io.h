
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

#include "stream_reader.h"
#include "stream_writer.h"

namespace vsag {

template <typename IOTmpl>
class BasicIO {
public:
    BasicIO<IOTmpl>() = default;

    virtual ~BasicIO() = default;

    inline void
    Write(const uint8_t* data, uint64_t size, uint64_t offset) {
        return cast().WriteImpl(data, size, offset);
    }

    inline bool
    Read(uint64_t size, uint64_t offset, uint8_t* data) const {
        return cast().ReadImpl(size, offset, data);
    }

    [[nodiscard]] inline const uint8_t*
    Read(uint64_t size, uint64_t offset, bool& need_release) const {
        return cast().ReadImpl(size, offset, need_release);  // TODO(LHT129): use IOReadObject
    }

    inline bool
    MultiRead(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        return cast().MultiReadImpl(datas, sizes, offsets, count);
    }

    inline void
    Prefetch(uint64_t offset, uint64_t cache_line = 64) {
        return cast().PrefetchImpl(offset, cache_line);
    }

    inline void
    Serialize(StreamWriter& writer) {
        return cast().SerializeImpl(writer);
    }

    inline void
    Deserialize(StreamReader& reader) {
        return cast().DeserializeImpl(reader);
    }

    inline void
    Release(const uint8_t* data) const {
        return cast().ReleaseImpl(data);
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
