
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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

std::string
read_json(const std::string& json_path) {
    std::ifstream infile(json_path);
    if (not infile.is_open() or infile.peek() == std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Missing or empty compatibility input: " + json_path);
    }
    std::stringstream buffer;
    buffer << infile.rdbuf();
    if (infile.bad()) {
        throw std::runtime_error("Cannot read compatibility input: " + json_path);
    }
    return buffer.str();
}

int
main(int argc, char** argv) {
    if (argc != 2 and argc != 3) {
        std::cerr << "Usage: check_compatibility <tag>_<algorithm> [input-directory]" << std::endl;
        return 1;
    }

    try {
        const std::string input_str = argv[1];
        const auto separator = input_str.find('_');
        if (separator == std::string::npos or separator == 0 or
            separator + 1 == input_str.length() or input_str.find('/') != std::string::npos or
            input_str.find('\\') != std::string::npos) {
            throw std::runtime_error("Invalid compatibility identifier: " + input_str);
        }
        const auto algo_name = input_str.substr(separator + 1);
        const char* configured_dir = std::getenv("COMPATIBILITY_INDEX_DIR");
        const std::string input_dir =
            argc == 3 ? argv[2]
                      : (configured_dir != nullptr and configured_dir[0] != '\0' ? configured_dir
                                                                                 : "/tmp");
        if (input_dir.empty()) {
            throw std::runtime_error("Compatibility input directory must not be empty");
        }
        const std::string prefix = input_dir + "/";
        const auto build_json = read_json(prefix + input_str + "_build.json");
        const auto search_json = read_json(prefix + input_str + "_search.json");
        std::ifstream index_file(prefix + input_str + ".index", std::ios::binary);
        if (not index_file.is_open() or index_file.peek() == std::ifstream::traits_type::eof()) {
            throw std::runtime_error("Missing or empty index: " + prefix + input_str + ".index");
        }

        constexpr int64_t dim = 512;
        constexpr int64_t count = 500;
        constexpr int64_t fixture_count = 10000;
        const auto origin_data_path = prefix + "random_512d_10K.bin";
        std::ifstream ifs(origin_data_path, std::ios::binary | std::ios::ate);
        const auto expected_bytes =
            static_cast<std::streamoff>(fixture_count * dim * sizeof(float));
        if (not ifs.is_open() or ifs.tellg() != expected_bytes) {
            throw std::runtime_error("Missing or incomplete vector fixture: " + origin_data_path);
        }
        ifs.seekg(0);
        std::vector<float> data(count * dim);
        const auto query_bytes = static_cast<std::streamsize>(data.size() * sizeof(float));
        if (not ifs.read(reinterpret_cast<char*>(data.data()), query_bytes)) {
            throw std::runtime_error("Cannot read query vectors: " + origin_data_path);
        }

        auto log_error = [&]() { std::cerr << input_str << " failed " << std::endl; };
        auto index = vsag::Factory::CreateIndex(algo_name, build_json);
        if (not index.has_value()) {
            log_error();
            return 1;
        }
        auto algo = index.value();
        auto load_index = algo->Deserialize(index_file);
        if (not load_index.has_value()) {
            log_error();
            return 1;
        }
        for (int64_t i = 0; i < count; ++i) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(data.data() + i * dim)->Owner(false);
            auto knn_result = algo->KnnSearch(query, 1, search_json);
            if (not knn_result.has_value() or knn_result.value()->GetDim() < 1 or
                knn_result.value()->GetIds() == nullptr or knn_result.value()->GetIds()[0] != i) {
                log_error();
                return 1;
            }
        }
        std::cout << input_str << " success " << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << argv[1] << " failed: " << error.what() << std::endl;
        return 1;
    }
}
