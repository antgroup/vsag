
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

    [[nodiscard]] virtual AttrValueType
    GetValueType() const = 0;

    [[nodiscard]] virtual uint64_t
    GetValueCount() const = 0;

    [[nodiscard]] virtual Attribute*
    DeepCopy() const = 0;

    [[nodiscard]] virtual bool
    Equal(const Attribute* other) const = 0;
};
using AttributePtr = std::shared_ptr<Attribute>;

template <class T>
class AttributeValue : public Attribute {
public:
    [[nodiscard]] AttrValueType
    GetValueType() const override;

    [[nodiscard]] uint64_t
    GetValueCount() const override;

    std::vector<T>&
    GetValue();

    const std::vector<T>&
    GetValue() const;

    Attribute*
    DeepCopy() const override;

    bool
    Equal(const Attribute* other) const override;

private:
    std::vector<T> value_{};
};

struct AttributeSet {
public:
    std::vector<Attribute*> attrs_;
};

}  // namespace vsag
