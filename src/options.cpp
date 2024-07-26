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

#include "vsag/options.h"

#include <utility>

#include "default_allocator.h"
#include "default_logger.h"
#include "logger.h"

namespace vsag {

Options&
Options::Instance() {
    static Options s_instance;
    return s_instance;
}

Logger*
Options::logger() {
    static std::shared_ptr<DefaultLogger> s_default_logger = std::make_shared<DefaultLogger>();
    if (not logger_) {
        this->set_logger(s_default_logger.get());
    }
    return logger_;
}

}  // namespace vsag
