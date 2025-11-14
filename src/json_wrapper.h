
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

#include <nlohmann/json_fwd.hpp>
#include <string>

namespace vsag {

class JsonWrapper {
public:
    JsonWrapper();

    JsonWrapper(const JsonWrapper& other);

    ~JsonWrapper();

    static JsonWrapper
    Parse(const std::string& json_str, const bool allow_exceptions = true);

    bool
    Contains(const std::string& key) const;

    std::string
    Dump(int indent = -1) const;

    bool
    IsNumberInteger() const;

    bool
    IsString() const;

    bool
    IsDiscarded() const;

    bool
    IsArray() const;

    JsonWrapper&
    operator=(const JsonWrapper& other);

    JsonWrapper
    operator[](const std::string& key) const;

    void
    SetString(const std::string& str_value);

    void
    SetJson(const JsonWrapper& json);

    void
    SetInt(uint64_t value);

    void
    SetFloat(float value);

    void
    SetBool(bool value);

    void
    SetVector(const std::vector<int32_t>& value);

    void
    UpdateJson(const JsonWrapper& json);

    std::string
    GetString() const;

    int64_t
    GetInt() const;

    float
    GetFloat() const;

    bool
    GetBool() const;

    std::vector<int32_t>
    GetVector() const;

    void
    Clear();

    void
    Erase(const std::string& key);

    nlohmann::json*
    GetInnerJson() const {
        return this->json_;
    }

private:
    JsonWrapper(nlohmann::json* json);

    nlohmann::json* json_{nullptr};

    bool owns_json_{false};
};

}  // namespace vsag
