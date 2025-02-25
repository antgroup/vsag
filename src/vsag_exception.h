
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

#include <exception>

#include "vsag/errors.h"

namespace vsag {
class VsagException : std::exception {
public:
    explicit VsagException(Error& error) : error_(error){};

    template <typename... Args>
    explicit VsagException(ErrorType error_type, Args&&... args)
        : error_(error_type, concatenate_message(std::forward<Args>(args)...)) {
    }

    const char*
    what() const noexcept override {
        return error_.message.c_str();
    }

    template <typename... Args>
    static std::string
    concatenate_message(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        return oss.str();
    }

public:
    Error error_;
};
}  // namespace vsag
