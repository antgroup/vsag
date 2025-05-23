
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
#include <string>

#include "../common.h"

namespace vsag::eval {

class Formatter;
using FormatterPtr = std::shared_ptr<Formatter>;

class Formatter {
public:
    static FormatterPtr
    Create(const std::string& format);

    virtual std::string
    Format(vsag::eval::JsonType& results) = 0;
};

#define JSON_GET(varname, jsonobj, defaultvalue) \
    std::string varname;                         \
    try {                                        \
        (varname) = jsonobj;                     \
    } catch (...) {                              \
        (varname) = defaultvalue;                \
    }

}  // namespace vsag::eval
