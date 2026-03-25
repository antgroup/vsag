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

#include "binding.h"
#include "vsag/vsag.h"

namespace {

void
set_logger_off() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kOFF);
}

void
set_logger_info() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kINFO);
}

void
set_logger_debug() {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::kDEBUG);
}

}  // namespace

void
bind_logging(py::module_& module) {
    module.def(
        "set_logger_off", &set_logger_off, "Disable all logging output from the VSAG library.");

    module.def("set_logger_info",
               &set_logger_info,
               "Set logger level to INFO. Only important information will be logged.");

    module.def("set_logger_debug",
               &set_logger_debug,
               "Set logger level to DEBUG. Detailed debug information will be logged.");
}
