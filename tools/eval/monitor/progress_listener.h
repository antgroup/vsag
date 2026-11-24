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

#include <nlohmann/json.hpp>
#include <string>

namespace vsag::eval {

// Receives progress notifications from running eval cases. Implemented by the HTTP monitor of the
// command-line tool; the evaluation core only depends on this interface.
class ProgressListener {
public:
    virtual ~ProgressListener() = default;

    virtual void
    SetCurrentCase(const std::string& name) = 0;

    virtual void
    MarkCaseCompleted() = 0;

    virtual void
    UpdateMetrics(const nlohmann::json& metrics) = 0;

    // 0.0 - 100.0
    virtual void
    UpdateProgress(float percent_complete) = 0;
};

}  // namespace vsag::eval
