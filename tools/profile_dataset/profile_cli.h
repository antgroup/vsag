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

#include <ostream>

namespace vsag::dataset_profile {

class HttpTransport;

/// Exit codes of the profile_dataset command.
constexpr int EXIT_OK = 0;
constexpr int EXIT_FAILURE_INPUT = 1;
constexpr int EXIT_AI_REQUIRED_FAILED = 3;

/// Entry point of the profile_dataset command. The deterministic JSON and HTML reports are
/// written before any AI request, and rewritten with the AI section afterwards, so an
/// unavailable provider never prevents them. `transport` overrides the network layer (tests).
int
run_cli(int argc,
        char** argv,
        std::ostream& out,
        std::ostream& err,
        HttpTransport* transport = nullptr);

}  // namespace vsag::dataset_profile
