
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

#include <memory>
#include <string>
#include <vector>

namespace vsag {

enum AttrValueType {
    INT32 = 1,
    UINT32 = 2,
    INT64 = 3,
    UINT64 = 4,
    INT8 = 5,
    UINT8 = 6,
    INT16 = 7,
    UINT16 = 8,
    STRING = 9,
};

class Attribute {
public:
    std::string name_{};

    virtual ~Attribute() = default;

    virtual AttrValueType
    GetValueType() const = 0;

    virtual uint64_t
    GetValueCount() const = 0;
};

template <class T>
class AttributeValue : public Attribute {
public:
    AttrValueType
    GetValueType() const override {
        if constexpr (std::is_same_v<T, int32_t>) {
            return AttrValueType::INT32;
        } else if constexpr (std::is_same_v<T, uint32_t>) {
            return AttrValueType::UINT32;
        } else if constexpr (std::is_same_v<T, int64_t>) {
            return AttrValueType::INT64;
        } else if constexpr (std::is_same_v<T, uint64_t>) {
            return AttrValueType::UINT64;
        } else if constexpr (std::is_same_v<T, int8_t>) {
            return AttrValueType::INT8;
        } else if constexpr (std::is_same_v<T, uint8_t>) {
            return AttrValueType::UINT8;
        } else if constexpr (std::is_same_v<T, int16_t>) {
            return AttrValueType::INT16;
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            return AttrValueType::UINT16;
        } else if constexpr (std::is_same_v<T, std::string>) {
            return AttrValueType::STRING;
        }
    }

    uint64_t
    GetValueCount() const override {
        return value_.size();
    }

public:
    std::vector<T> value_{};
};

struct AttributeSet {
public:
    std::vector<Attribute*> attrs_;
};

}  // namespace vsag
