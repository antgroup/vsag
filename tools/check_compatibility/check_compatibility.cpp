
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

std::vector<std::string>
split_string(const std::string& input, char delimiter) {
    std::vector<std::string> result;
    std::stringstream ss(input);
    std::string item;

    while (getline(ss, item, delimiter)) {
        result.push_back(item);
    }

    return result;
}

std::string
read_json(const std::string& json_path) {
    std::ifstream infile(json_path);
    std::stringstream buffer;
    buffer << infile.rdbuf();
    infile.close();

    return buffer.str();
}

int
main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "input error" << std::endl;
        return -1;
    }

    std::string input_str = argv[1];
    auto strs = split_string(input_str, '_');
    auto tag_id = strs[0];
    auto algo_name = strs[1];
    std::string index_path = "/tmp/" + input_str + ".index";
    std::string search_json_path = "/tmp/" + input_str + "_search.json";
    std::string build_json_path = "/tmp/" + input_str + "_build.json";

    auto build_json = read_json(build_json_path);
    auto search_json = read_json(search_json_path);

    auto log_error = [&]() { std::cerr << input_str << " failed " << std::endl; };

    auto index = vsag::Factory::CreateIndex(algo_name, build_json);
    if (not index.has_value()) {
        log_error();
        return -1;
    }
    auto algo = index.value();
    std::ifstream index_file(index_path, std::ios::binary);
    auto load_index = algo->Deserialize(index_file);
    if (not load_index.has_value()) {
        log_error();
        return -1;
    }
    int64_t dim = 512;
    auto count = 500;
    std::string origin_data_path = "/tmp/random_512d_10K.bin";
    std::ifstream ifs(origin_data_path, std::ios::binary);
    std::vector<float> data(count * dim);
    ifs.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));

    for (int i = 0; i < count; ++i) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dim)->Float32Vectors(data.data() + i * dim)->Owner(false);
        auto knn_result = algo->KnnSearch(query, 1, search_json);
        if (not knn_result.has_value()) {
            log_error();
            return -1;
        }
        if (knn_result.value()->GetIds()[0] != i) {
            log_error();
            return -1;
        }
    }
    std::cout << input_str << " success " << std::endl;
    return 0;
}
