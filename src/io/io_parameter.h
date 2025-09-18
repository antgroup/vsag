
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

#include "parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(IOParam, IOParameter);

class IOParameter : public Parameter {
public:
    static IOParamPtr
    GetIOParameterByJson(const JsonType& json);

public:
    inline std::string
    GetTypeName() {
        return this->name_;
    }

protected:
    explicit IOParameter(std::string name);

    ~IOParameter() override = default;

private:
    std::string name_{};
};

}  // namespace vsag
