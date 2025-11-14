
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

#include "json_wrapper.h"

#include <nlohmann/json.hpp>

namespace vsag {

JsonWrapper::JsonWrapper() {
    json_ = new nlohmann::json();
    owns_json_ = true;
}

JsonWrapper::~JsonWrapper() {
    if (owns_json_) {
        delete json_;
    }
}

JsonWrapper::JsonWrapper(const JsonWrapper& other) {
    if (owns_json_) {
        delete json_;
    }
    json_ = new nlohmann::json();
    owns_json_ = true;
    if (other.json_ != nullptr) {
        *json_ = *other.json_;
    }
}

JsonWrapper&
JsonWrapper::operator=(const JsonWrapper& other) {
    if (this != &other) {
        if (owns_json_) {
            delete json_;
        }
        json_ = new nlohmann::json();
        if (other.json_ != nullptr) {
            *json_ = *other.json_;
        }
        owns_json_ = true;
    }
    return *this;
}

JsonWrapper
JsonWrapper::Parse(const std::string& json_str, const bool allow_exceptions) {
    nlohmann::json json = nlohmann::json::parse(json_str, nullptr, allow_exceptions);
    JsonWrapper wrapper;
    *(wrapper.json_) = json;
    wrapper.owns_json_ = true;
    return wrapper;
}

JsonWrapper::JsonWrapper(nlohmann::json* json) {
    json_ = json;
    owns_json_ = false;
}

JsonWrapper
JsonWrapper::operator[](const std::string& key) const {
    auto& value = (*json_)[key];
    return {&value};
}

bool
JsonWrapper::Contains(const std::string& key) const {
    return json_->contains(key);
}

std::string
JsonWrapper::Dump(int indent) const {
    return json_->dump(indent);
}

bool
JsonWrapper::IsNumberInteger() const {
    return json_->is_number_integer();
}

bool
JsonWrapper::IsString() const {
    return json_->is_string();
}

bool
JsonWrapper::IsDiscarded() const {
    return json_->is_discarded();
}

bool
JsonWrapper::IsArray() const {
    return json_->is_array();
}

void
JsonWrapper::SetString(const std::string& str_value) {
    (*json_) = str_value;
}

void
JsonWrapper::SetJson(const JsonWrapper& json) {
    (*json_) = *json.json_;
}

void
JsonWrapper::SetInt(uint64_t value) {
    (*json_) = value;
}

void
JsonWrapper::SetVector(const std::vector<int32_t>& value) {
    (*json_) = value;
}

void
JsonWrapper::SetFloat(float value) {
    (*json_) = value;
}

void
JsonWrapper::SetBool(bool value) {
    (*json_) = value;
}

std::string
JsonWrapper::GetString() const {
    return (*json_).get<std::string>();
}

int64_t
JsonWrapper::GetInt() const {
    return (*json_).get<int64_t>();
}

float
JsonWrapper::GetFloat() const {
    return (*json_).get<float>();
}

bool
JsonWrapper::GetBool() const {
    return (*json_).get<bool>();
}

std::vector<int32_t>
JsonWrapper::GetVector() const {
    return (*json_).get<std::vector<int32_t>>();
}

void
JsonWrapper::Clear() {
    json_->clear();
}

void
JsonWrapper::Erase(const std::string& key) {
    json_->erase(key);
}

void
JsonWrapper::UpdateJson(const JsonWrapper& json) {
    (*json_).update(*json.json_);
}

}  // namespace vsag
