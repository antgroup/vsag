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

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "autotune.h"
#include "vsag/options.h"

int
main(int argc, char** argv) try {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <request.json>" << std::endl;
        return 1;
    }

    vsag::autotune::JsonType result;
    try {
        std::ifstream input(argv[1]);
        if (!input.good()) {
            throw std::runtime_error("failed to open request file: " + std::string(argv[1]));
        }
        vsag::autotune::JsonType request;
        input >> request;
        result = vsag::autotune::RunAutoTune(request);
    } catch (const std::exception& error) {
        result = {{"version", 1},
                  {"status", "failed"},
                  {"failure",
                   {{"stage", "cli"}, {"code", "request_file_error"}, {"message", error.what()}}}};
    }

    std::cout << vsag::autotune::FormatResultSummaryForCli(result) << std::endl;
    return result.value("status", std::string()) == "failed" ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << "AutoTune failed: " << error.what() << std::endl;
    return 1;
} catch (...) {
    std::cerr << "AutoTune failed with an unknown error" << std::endl;
    return 1;
}
