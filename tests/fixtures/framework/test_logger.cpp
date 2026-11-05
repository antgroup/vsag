
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

#include "test_logger.h"

#include <catch2/catch_message.hpp>
#include <mutex>

#include "vsag/logger.h"

namespace fixtures::logger {

TestLogger test_logger;

LoggerStream trace_buff(&test_logger, vsag::Logger::kTRACE);
LoggerStream debug_buff(&test_logger, vsag::Logger::kDEBUG);
LoggerStream info_buff(&test_logger, vsag::Logger::kINFO);
LoggerStream warn_buff(&test_logger, vsag::Logger::kWARN);
LoggerStream error_buff(&test_logger, vsag::Logger::kERR);
LoggerStream critical_buff(&test_logger, vsag::Logger::kCRITICAL);

std::basic_ostream<char> trace(&trace_buff);
std::basic_ostream<char> debug(&debug_buff);
std::basic_ostream<char> info(&info_buff);
std::basic_ostream<char> warn(&warn_buff);
std::basic_ostream<char> error(&error_buff);

}  // namespace fixtures::logger
