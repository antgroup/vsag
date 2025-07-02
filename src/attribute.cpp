
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

#include "vsag/attribute.h"

namespace vsag {

template <class T>
AttrValueType
AttributeValue<T>::GetValueType() const {
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

template <class T>
uint64_t
AttributeValue<T>::GetValueCount() const {
    return value_.size();
}

template <class T>
std::vector<T>&
AttributeValue<T>::GetValue() {
    return value_;
}

template <class T>
const std::vector<T>&
AttributeValue<T>::GetValue() const {
    return value_;
}

template class AttributeValue<int32_t>;
template class AttributeValue<uint32_t>;
template class AttributeValue<int64_t>;
template class AttributeValue<uint64_t>;
template class AttributeValue<int8_t>;
template class AttributeValue<uint8_t>;
template class AttributeValue<int16_t>;
template class AttributeValue<uint16_t>;
template class AttributeValue<std::string>;

}  // namespace vsag
