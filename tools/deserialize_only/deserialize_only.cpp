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

#include <vsag/vsag.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string
ReadFile(const std::string& path) {
    std::ifstream input(path);
    if (not input) {
        throw std::runtime_error("failed to open create-params file: " + path);
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

}  // namespace

int
main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: deserialize_only <index_name> <create_params_file> <index_path>\n";
        return 2;
    }

    try {
        const std::string index_name = argv[1];
        const std::string create_params = ReadFile(argv[2]);
        const std::string index_path = argv[3];

        auto create_result = vsag::Factory::CreateIndex(index_name, create_params);
        if (not create_result.has_value()) {
            std::cerr << "CreateIndex failed: " << create_result.error().message << '\n';
            return 1;
        }

        std::ifstream index_file(index_path, std::ios::binary);
        if (not index_file) {
            std::cerr << "failed to open index file: " << index_path << '\n';
            return 1;
        }
        auto deserialize_result = create_result.value()->Deserialize(index_file);
        if (not deserialize_result.has_value()) {
            std::cerr << "Deserialize failed: " << deserialize_result.error().message << '\n';
            return 1;
        }
        std::cout << "Deserialize succeeded: " << index_path << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Deserialize failed with exception: " << error.what() << '\n';
        return 1;
    }
}
