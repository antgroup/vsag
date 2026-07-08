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

#include "vsag/load_parameters.h"

#include <utility>

#include "json_types.h"

namespace vsag {

struct LoadParameters::Impl {
    JsonType json{JsonType::Parse("{}")};
    ReaderSet readers;
    std::string raw_json{"{}"};
    bool valid_json{true};
    bool mutated{false};
};

LoadParameters::LoadParameters() : impl_(std::make_shared<Impl>()) {
}

LoadParameters::LoadParameters(const char* json_string)
    : LoadParameters(json_string == nullptr ? std::string{} : std::string(json_string)) {
}

LoadParameters::LoadParameters(const std::string& json_string) : impl_(std::make_shared<Impl>()) {
    impl_->raw_json = json_string.empty() ? "{}" : json_string;
    impl_->json = JsonType::Parse(impl_->raw_json, false);
    impl_->valid_json = !impl_->json.IsDiscarded() && impl_->json.IsObject();
    if (!impl_->valid_json) {
        impl_->json = JsonType::Parse("{}");
    }
}

LoadParameters::LoadParameters(const LoadParameters& other)
    : impl_(std::make_shared<Impl>(*other.impl_)) {
}

LoadParameters&
LoadParameters::operator=(const LoadParameters& other) {
    if (this != &other) {
        impl_ = std::make_shared<Impl>(*other.impl_);
    }
    return *this;
}

LoadParameters::~LoadParameters() = default;

LoadParameters&
LoadParameters::Set(const std::string& key, const std::string& value) {
    impl_->mutated = true;
    impl_->valid_json = true;
    impl_->json[key].SetString(value);
    return *this;
}

LoadParameters&
LoadParameters::Set(const std::string& key, const char* value) {
    impl_->mutated = true;
    impl_->valid_json = true;
    impl_->json[key].SetString(value == nullptr ? "" : value);
    return *this;
}

LoadParameters&
LoadParameters::Set(const std::string& key, bool value) {
    impl_->mutated = true;
    impl_->valid_json = true;
    impl_->json[key].SetBool(value);
    return *this;
}

LoadParameters&
LoadParameters::Set(const std::string& key, int64_t value) {
    impl_->mutated = true;
    impl_->valid_json = true;
    impl_->json[key].SetInt64(value);
    return *this;
}

LoadParameters&
LoadParameters::Set(const std::string& key, uint64_t value) {
    impl_->mutated = true;
    impl_->valid_json = true;
    impl_->json[key].SetUint64(value);
    return *this;
}

LoadParameters&
LoadParameters::Set(const std::string& key, double value) {
    impl_->mutated = true;
    impl_->valid_json = true;
    impl_->json[key].SetDouble(value);
    return *this;
}

LoadParameters&
LoadParameters::Set(const std::string& key, const LoadParameters& value) {
    impl_->mutated = true;
    impl_->valid_json = true;
    auto nested_json = JsonType::Parse(value.Dump(), false);
    if (nested_json.IsDiscarded()) {
        nested_json = JsonType::Parse("{}");
    }
    impl_->json[key].SetJson(nested_json);
    return *this;
}

LoadParameters&
LoadParameters::SetReader(const std::string& key, ReaderPtr reader) {
    impl_->mutated = true;
    impl_->valid_json = true;
    impl_->readers.Set(key, std::move(reader));
    impl_->json[key].SetString("<Reader>");
    return *this;
}

bool
LoadParameters::HasReader(const std::string& key) const {
    return impl_->readers.Contains(key);
}

ReaderPtr
LoadParameters::GetReader(const std::string& key) const {
    return impl_->readers.Get(key);
}

std::string
LoadParameters::Dump() const {
    if (!impl_->valid_json && !impl_->mutated) {
        return impl_->raw_json;
    }
    return impl_->json.Dump();
}

}  // namespace vsag
