
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
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <variant>
#include <vector>

namespace vsag::eval {

class Monitor {
public:
    using JsonType = nlohmann::json;

public:
    explicit Monitor(std::string name);

    virtual ~Monitor() = default;

    virtual void
    Start() = 0;

    virtual void
    Stop() = 0;

    virtual JsonType
    GetResult() = 0;

    [[nodiscard]] std::string
    GetName() const {
        return name_;
    }

public:
    virtual void
    Record(void* input = nullptr){};

public:
    std::string name_{};
    std::mutex record_mutex_;
};

using MonitorPtr = std::shared_ptr<Monitor>;

}  // namespace vsag::eval
