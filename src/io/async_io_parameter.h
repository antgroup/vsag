
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

#include "io_parameter.h"

namespace vsag {
class AsyncIOParameter : public IOParameter {
public:
    AsyncIOParameter();

    explicit AsyncIOParameter(const JsonType& json);

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

public:
    std::string path_{};
};

using AsyncIOParameterPtr = std::shared_ptr<AsyncIOParameter>;

}  // namespace vsag
