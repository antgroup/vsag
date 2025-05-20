
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
    STRING = 7,
    FIXSIZE_STRING = 8,
};

class Attribute {
public:
    std::string name_{};
    virtual AttrValueType
    GetValueType() = 0;
};

template <class T>
class AttributeValue : public Attribute {
public:
    std::vector<T> value_{};
};

struct AttributeSet {
public:
    std::vector<Attribute*> attrs_;
};

}  // namespace vsag
