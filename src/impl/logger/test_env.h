
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

#include <cstdlib>
#include <string>

#include "unittest.h"

namespace vsag::test {

[[nodiscard]] inline bool
set_env_value(const char* name, const char* value) {
#ifdef _WIN32
    return _putenv_s(name, value) == 0;
#else
    return setenv(name, value, 1) == 0;
#endif
}

[[nodiscard]] inline bool
unset_env_value(const char* name) {
#ifdef _WIN32
    return _putenv_s(name, "") == 0;
#else
    return unsetenv(name) == 0;
#endif
}

class ScopedEnv {
public:
    explicit ScopedEnv(const char* name) : name_(name) {
        save_old_value();
        REQUIRE(unset_env_value(name_));
    }

    ScopedEnv(const char* name, const char* value) : name_(name) {
        save_old_value();
        REQUIRE(set_env_value(name_, value));
    }

    ~ScopedEnv() {
        bool restored = false;
        if (had_old_value_) {
            restored = set_env_value(name_, old_value_.c_str());
        } else {
            restored = unset_env_value(name_);
        }
        (void)restored;
    }

private:
    void
    save_old_value() {
        const auto* old_value = std::getenv(name_);
        if (old_value != nullptr) {
            had_old_value_ = true;
            old_value_ = old_value;
        }
    }

    const char* name_;
    bool had_old_value_{false};
    std::string old_value_;
};

}  // namespace vsag::test
