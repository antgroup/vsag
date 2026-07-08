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
#include <memory>
#include <string>

#include "vsag/readerset.h"

namespace vsag {

class LoadParameters {
public:
    LoadParameters();
    LoadParameters(const char* json_string);
    LoadParameters(const std::string& json_string);
    LoadParameters(const LoadParameters& other);
    LoadParameters&
    operator=(const LoadParameters& other);
    ~LoadParameters();

    LoadParameters&
    Set(const std::string& key, const std::string& value);

    LoadParameters&
    Set(const std::string& key, const char* value);

    LoadParameters&
    Set(const std::string& key, bool value);

    LoadParameters&
    Set(const std::string& key, int64_t value);

    LoadParameters&
    Set(const std::string& key, uint64_t value);

    LoadParameters&
    Set(const std::string& key, double value);

    LoadParameters&
    Set(const std::string& key, const LoadParameters& value);

    LoadParameters&
    SetReader(const std::string& key, ReaderPtr reader);

    [[nodiscard]] bool
    HasReader(const std::string& key) const;

    [[nodiscard]] ReaderPtr
    GetReader(const std::string& key) const;

    [[nodiscard]] std::string
    Dump() const;

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace vsag
