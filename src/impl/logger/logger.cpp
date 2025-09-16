
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

#include "logger.h"

#include <spdlog/spdlog.h>

namespace vsag ::logger {
void
set_level(level log_level) {
    Options::Instance().logger()->SetLevel((Logger::Level)log_level);
}

void
trace(const std::string& msg) {
    Options::Instance().logger()->Trace(msg);
}

void
debug(const std::string& msg) {
    Options::Instance().logger()->Debug(msg);
}

void
info(const std::string& msg) {
    Options::Instance().logger()->Info(msg);
}

void
warn(const std::string& msg) {
    Options::Instance().logger()->Warn(msg);
}

void
error(const std::string& msg) {
    Options::Instance().logger()->Error(msg);
}

void
critical(const std::string& msg) {
    Options::Instance().logger()->Critical(msg);
}

}  // namespace vsag::logger
