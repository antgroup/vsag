
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

#define GENERATE_HAS_MEMBER_FUNC(funcName, ...)                              \
    template <typename U>                                                    \
    struct has_##funcName {                                                  \
        template <typename T, T>                                             \
        struct SFINAE;                                                       \
        template <typename T>                                                \
        static std::true_type                                                \
        test(SFINAE<decltype(&T::funcName), &T::funcName>*);                 \
        template <typename T>                                                \
        static std::false_type                                               \
        test(...);                                                           \
        static constexpr bool value =                                        \
            std::is_same<decltype(test<U>(nullptr)), std::true_type>::value; \
    };

template <typename IOTmpl>
class BasicIO {
public:
    BasicIO<IOTmpl>() = default;

    virtual ~BasicIO() = default;

    inline void
    Write(const uint8_t* data, uint64_t size, uint64_t offset) {
        if constexpr (has_WriteImpl<IOTmpl>::value) {
            cast().WriteImpl(data, size, offset);
        } else {
            throw std::runtime_error(
                fmt::format("class {} have no func named WriteImpl", typeid(IOTmpl).name()));
        }
    }

    inline bool
    Read(uint64_t size, uint64_t offset, uint8_t* data) const {
        if constexpr (has_ReadImpl<IOTmpl>::value) {
            return cast().ReadImpl(size, offset, data);
        } else {
            throw std::runtime_error(
                fmt::format("class {} have no func named ReadImpl", typeid(IOTmpl).name()));
        }
    }

    [[nodiscard]] inline const uint8_t*
    Read(uint64_t size, uint64_t offset, bool& need_release) const {
        if constexpr (has_DirectReadImpl<IOTmpl>::value) {
            return cast().DirectReadImpl(
                size, offset, need_release);  // TODO(LHT129): use IOReadObject
        } else {
            throw std::runtime_error(
                fmt::format("class {} have no func named DirectReadImpl", typeid(IOTmpl).name()));
        }
    }

    inline bool
    MultiRead(uint8_t* datas, uint64_t* sizes, uint64_t* offsets, uint64_t count) const {
        if constexpr (has_MultiReadImpl<IOTmpl>::value) {
            return cast().MultiReadImpl(datas, sizes, offsets, count);
        } else {
            throw std::runtime_error(
                fmt::format("class {} have no func named MultiReadImpl", typeid(IOTmpl).name()));
        }
    }

    inline void
    Prefetch(uint64_t offset, uint64_t cache_line = 64) {
        if constexpr (has_PrefetchImpl<IOTmpl>::value) {
            cast().PrefetchImpl(offset, cache_line);
        } else {
            throw std::runtime_error(
                fmt::format("class {} have no func named PrefetchImpl", typeid(IOTmpl).name()));
        }
    }

    inline void
    Serialize(StreamWriter& writer) {
        if constexpr (has_SerializeImpl<IOTmpl>::value) {
            cast().SerializeImpl(writer);
        } else {
            throw std::runtime_error(
                fmt::format("class {} have no func named SerializeImpl", typeid(IOTmpl).name()));
        }
    }

    inline void
    Deserialize(StreamReader& reader) {
        if constexpr (has_DeserializeImpl<IOTmpl>::value) {
            cast().DeserializeImpl(reader);
        } else {
            throw std::runtime_error(
                fmt::format("class {} have no func named DeserializeImpl", typeid(IOTmpl).name()));
        }
    }

    inline void
    Release(const uint8_t* data) const {
        if constexpr (has_ReleaseImpl<IOTmpl>::value) {
            cast().ReleaseImpl(data);
        } else {
            throw std::runtime_error(
                fmt::format("class {} have no func named ReleaseImpl", typeid(IOTmpl).name()));
        }
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

private:
    GENERATE_HAS_MEMBER_FUNC(WriteImpl, void (U::*)(const uint8_t*, uint64_t, uint64_t))
    GENERATE_HAS_MEMBER_FUNC(ReadImpl, bool (U::*)(uint64_t, uint64_t, uint8_t*))
    GENERATE_HAS_MEMBER_FUNC(DirectReadImpl, const uint8_t* (U::*)(uint64_t, uint64_t, bool&))
    GENERATE_HAS_MEMBER_FUNC(MultiReadImpl, bool (U::*)(uint8_t*, uint64_t*, uint64_t*, uint64_t))
    GENERATE_HAS_MEMBER_FUNC(PrefetchImpl, void (U::*)(uint64_t, uint64_t))
    GENERATE_HAS_MEMBER_FUNC(SerializeImpl, void (U::*)(StreamWriter&))
    GENERATE_HAS_MEMBER_FUNC(DeserializeImpl, void (U::*)(StreamReader&))
    GENERATE_HAS_MEMBER_FUNC(ReleaseImpl, void (U::*)(const uint8_t*))
};
}  // namespace vsag
