
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

#include "param/schema.h"

#include <fmt/format.h>

#include <nlohmann/json.hpp>

#include "common.h"

namespace vsag::param {

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

void
Schema::AddField(FieldPtr field) {
    this->known_keys_.insert(field->Key());
    this->fields_.push_back(std::move(field));
}

void
Schema::Parse(void* instance, const JsonType& parent) const {
    for (const auto& field : this->fields_) {
        field->Parse(instance, parent);
    }
}

void
Schema::Serialize(const void* instance, JsonType& parent) const {
    for (const auto& field : this->fields_) {
        field->Serialize(instance, parent);
    }
}

bool
Schema::Equal(const void* lhs, const void* rhs) const {
    for (const auto& field : this->fields_) {
        if (not field->Equal(lhs, rhs)) {
            return false;
        }
    }
    return true;
}

void
Schema::CheckUnknownKeys(const JsonType& parent,
                         const std::unordered_set<std::string>& extra_known_keys) const {
    auto* inner = parent.GetInnerJson();
    if (inner == nullptr || not inner->is_object()) {
        return;
    }
    for (const auto& item : inner->items()) {
        const auto& key = item.key();
        if (this->known_keys_.count(key) != 0U) {
            continue;
        }
        if (extra_known_keys.count(key) != 0U) {
            continue;
        }
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("unknown parameter key '{}' is not declared by the schema", key));
    }
}

// ---------------------------------------------------------------------------
// ConstantStringField
// ---------------------------------------------------------------------------

ConstantStringField::ConstantStringField(std::string key, std::string value)
    : key_(std::move(key)), value_(std::move(value)) {
}

void
ConstantStringField::Parse(void* /*instance*/, const JsonType& parent) const {
    if (not parent.Contains(this->key_)) {
        return;
    }
    auto actual = parent[this->key_].GetString();
    CHECK_ARGUMENT(
        actual == this->value_,
        fmt::format("parameter '{}' must be '{}', got '{}'", this->key_, this->value_, actual));
}

void
ConstantStringField::Serialize(const void* /*instance*/, JsonType& parent) const {
    parent[this->key_].SetString(this->value_);
}

// ---------------------------------------------------------------------------
// Scalar JSON read/write specializations
// ---------------------------------------------------------------------------

template <>
uint64_t
ReadJsonValue<uint64_t>(const JsonType& json) {
    return json.GetUint64();
}

template <>
int64_t
ReadJsonValue<int64_t>(const JsonType& json) {
    return json.GetInt();
}

template <>
std::string
ReadJsonValue<std::string>(const JsonType& json) {
    return json.GetString();
}

template <>
bool
ReadJsonValue<bool>(const JsonType& json) {
    return json.GetBool();
}

template <>
float
ReadJsonValue<float>(const JsonType& json) {
    return json.GetFloat();
}

template <>
void
WriteJsonValueAt<uint64_t>(JsonType& parent, const std::string& key, const uint64_t& value) {
    parent[key].SetUint64(value);
}

template <>
void
WriteJsonValueAt<int64_t>(JsonType& parent, const std::string& key, const int64_t& value) {
    parent[key].SetInt(value);
}

template <>
void
WriteJsonValueAt<std::string>(JsonType& parent, const std::string& key, const std::string& value) {
    parent[key].SetString(value);
}

template <>
void
WriteJsonValueAt<bool>(JsonType& parent, const std::string& key, const bool& value) {
    parent[key].SetBool(value);
}

template <>
void
WriteJsonValueAt<float>(JsonType& parent, const std::string& key, const float& value) {
    parent[key].SetFloat(value);
}

}  // namespace vsag::param
