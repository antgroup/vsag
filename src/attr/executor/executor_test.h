
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

#include <string>
#include <unordered_set>
#include <vector>

#include "utils/util_functions.h"
#include "vsag/attribute.h"
namespace vsag {
class ExecutorTest {
public:
    static AttributeSet
    MockAttrSet() {
        const std::vector<std::string> attr_keys = {
            "i32_1",
            "i32_2",
            "u32_1",
            "u32_2",
            "i64_1",
            "i64_2",
            "u64_1",
            "u64_2",
            "i16_1",
            "i16_2",
            "i8_1",
            "i8_2",
            "str_1",
            "str_2",
            "str_3",
            "str_4",
        };

        AttributeSet attrset;
        auto attr_count = 10;
        attrset.attrs_.resize(attr_count);
        auto key_size = attr_keys.size();
        auto names = select_k_numbers(key_size, attr_count);
        int i = 0;
        for (auto& attr : attrset.attrs_) {
            auto name = attr_keys[names[i]];
            auto type_str = split_string(name, '_')[0];
            if (type_str == "str") {
                auto ptr = new AttributeValue<std::string>();
                ptr->name_ = name;
                auto vec = select_k_numbers(100, 10);
                for (int j = 0; j < 10; ++j) {
                    std::string value = name + "_" + std::to_string(vec[j]);
                    ptr->GetValue().emplace_back(value);
                }
                attr = ptr;
            } else if (type_str == "i32") {
                auto ptr = new AttributeValue<int32_t>();
                ptr->name_ = name;
                auto vec = select_k_numbers(100, 10);
                copy_vector(vec, ptr->GetValue());
                attr = ptr;
            } else if (type_str == "i64") {
                auto ptr = new AttributeValue<int64_t>();
                ptr->name_ = name;
                auto vec = select_k_numbers(100, 10);
                copy_vector(vec, ptr->GetValue());
                attr = ptr;
            } else if (type_str == "i8") {
                auto ptr = new AttributeValue<int8_t>();
                ptr->name_ = name;
                auto vec = select_k_numbers(100, 10);
                copy_vector(vec, ptr->GetValue());
                attr = ptr;
            } else if (type_str == "i16") {
                auto ptr = new AttributeValue<int16_t>();
                ptr->name_ = name;
                auto vec = select_k_numbers(100, 10);
                copy_vector(vec, ptr->GetValue());
                attr = ptr;
            } else if (type_str == "u64") {
                auto ptr = new AttributeValue<uint64_t>();
                ptr->name_ = name;
                auto vec = select_k_numbers(100, 10);
                copy_vector(vec, ptr->GetValue());
                attr = ptr;
            } else if (type_str == "u32") {
                auto ptr = new AttributeValue<uint32_t>();
                ptr->name_ = name;
                auto vec = select_k_numbers(100, 10);
                copy_vector(vec, ptr->GetValue());
                attr = ptr;
            } else if (type_str == "u16") {
                auto ptr = new AttributeValue<uint16_t>();
                ptr->name_ = name;
                auto vec = select_k_numbers(100, 10);
                copy_vector(vec, ptr->GetValue());
                attr = ptr;
            } else if (type_str == "u8") {
                auto ptr = new AttributeValue<uint8_t>();
                ptr->name_ = name;
                auto vec = select_k_numbers(100, 10);
                copy_vector(vec, ptr->GetValue());
                attr = ptr;
            }
            i++;
        }
        return attrset;
    }

    static void
    DeleteAttrSet(AttributeSet& attr_set) {
        for (auto& attr : attr_set.attrs_) {
            delete attr;
        }
        attr_set.attrs_.clear();
    }
};

template <class T>
std::vector<T>
GetValues(Attribute* attr) {
    auto* attr_ptr = dynamic_cast<AttributeValue<T>*>(attr);
    return attr_ptr->GetValue();
}

template <class T>
std::string
CreateMultiInString(const std::string& name, const std::vector<T>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << "|";
        }
        if constexpr (std::is_same_v<T, std::string>) {
            oss << values[i];
        } else {
            oss << std::to_string(values[i]);
        }
    }
    return "multi_in(" + name + ", \"" + oss.str() + "\", \"|\")";
}

template <class T>
std::string
CreateMultiNotInString(const std::string& name, const std::vector<T>& values) {
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << "|";
        }
        if constexpr (std::is_same_v<T, std::string>) {
            oss << values[i];
        } else {
            oss << std::to_string(values[i]);
        }
    }
    return "multi_notin(" + name + ", \"" + oss.str() + "\", \"|\")";
}

template <class T>
std::vector<T>
GetNoneInteractValues(const std::vector<T>& values, const std::string& name) {
    int64_t count = 10;
    std::vector<T> no_interact_values;
    std::vector<T> result;
    std::unordered_set<T> values_set(values.begin(), values.end());
    for (int i = 0; i < 100; ++i) {
        T key;
        if constexpr (std::is_same_v<T, std::string>) {
            key = name + "_" + std::to_string(i);
        } else {
            key = static_cast<T>(i);
        }
        if (values_set.count(key) == 0) {
            no_interact_values.emplace_back(key);
        }
    }
    auto size = no_interact_values.size();
    auto indexes = select_k_numbers(size, count);
    for (auto& idx : indexes) {
        result.emplace_back(no_interact_values[idx]);
    }
    return result;
}

template <class T>
std::vector<T>
GetInteractValues(const std::vector<T>& values, const std::string& name) {
    std::vector<T> result = GetNoneInteractValues(values, name);
    std::unordered_set<T> values_set(values.begin(), values.end());
    for (int i = 0; i < 100; ++i) {
        T key;
        if constexpr (std::is_same_v<T, std::string>) {
            key = name + "_" + std::to_string(i);
        } else {
            key = static_cast<T>(i);
        }
        if (values_set.count(key) != 0) {
            result.emplace_back(key);
            break;
        }
    }
    return result;
}

template <class T>
static std::string
CreateEqString(const std::string& left, const T& right) {
    std::stringstream ss;
    if constexpr (std::is_same_v<T, std::string>) {
        ss << left << " = \"" << right << "\"";
    } else {
        ss << left << " = " << std::to_string(right);
    }
    return ss.str();
}

template <class T>
static std::string
CreateNqString(const std::string& left, const T& right) {
    std::stringstream ss;
    if constexpr (std::is_same_v<T, std::string>) {
        ss << left << " != \"" << right << "\"";
    } else {
        ss << left << " !=" << std::to_string(right);
    }
    return ss.str();
}

template <class T>
static std::string
CreateOtherString(const std::string& left, const T& right, const std::string& other) {
    std::stringstream ss;
    if constexpr (std::is_same_v<T, std::string>) {
        ss << left << other << " \"" << right << "\"";
    } else {
        ss << left << other << std::to_string(right);
    }
    return ss.str();
}

}  // namespace vsag