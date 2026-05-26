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

#include <algorithm>
#include <argparse/argparse.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "algorithm/hgraph/hgraph.h"
#include "index/index_impl.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"

using namespace vsag;

namespace {

constexpr static const char* DEFAULT_BUILD_PARAM = "default";

void
parse_args(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument<std::string>("--index_path", "-i")
        .required()
        .help("Serialized HGraph index file path");

    parser.add_argument<std::string>("--label_path", "-l")
        .required()
        .help("Path to label.bin used for comparison");

    parser.add_argument<std::string>("--build_parameter", "-bp")
        .default_value(std::string(DEFAULT_BUILD_PARAM))
        .help("Optional full build parameter JSON, required when the index uses old serialization format");

    parser.add_argument("--sample_limit", "-s")
        .default_value(10)
        .help("How many mismatched labels to print as samples")
        .scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        throw;
    }
}

std::vector<int64_t>
read_labels(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (not input.is_open()) {
        throw std::runtime_error("failed to open label file: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (size < 0 || (size % static_cast<std::streamoff>(sizeof(int64_t))) != 0) {
        throw std::runtime_error("label.bin size is not aligned to int64: " + path);
    }

    std::vector<int64_t> labels(static_cast<size_t>(size / sizeof(int64_t)));
    if (not labels.empty()) {
        input.read(reinterpret_cast<char*>(labels.data()), size);
    }
    if (not input.good() && not input.eof()) {
        throw std::runtime_error("failed to read label file: " + path);
    }
    return labels;
}

class HGraphIndexLoader {
public:
    explicit HGraphIndexLoader(std::string build_param) : build_param_(std::move(build_param)) {
    }

    IndexPtr
    Load(const std::string& index_path) {
        std::ifstream probe_stream(index_path, std::ios::binary);
        if (not probe_stream.is_open()) {
            throw std::runtime_error("failed to open index file: " + index_path);
        }

        IOStreamReader reader(probe_stream);
        auto footer = Footer::Parse(reader);
        if (footer != nullptr) {
            return load_new_format(index_path, footer);
        }
        if (build_param_ == DEFAULT_BUILD_PARAM) {
            throw std::runtime_error(
                "old-format index requires --build_parameter with full HGraph build JSON");
        }
        return load_with_factory(index_path, build_param_);
    }

private:
    IndexPtr
    load_with_factory(const std::string& index_path, const std::string& build_param) {
        auto create_result = Factory::CreateIndex(INDEX_HGRAPH, build_param);
        if (not create_result.has_value()) {
            throw std::runtime_error("failed to create index: " +
                                     create_result.error().message);
        }
        auto index = create_result.value();
        std::ifstream input(index_path, std::ios::binary);
        if (not input.is_open()) {
            throw std::runtime_error("failed to reopen index file: " + index_path);
        }
        auto deserialize_result = index->Deserialize(input);
        if (not deserialize_result.has_value()) {
            throw std::runtime_error("failed to deserialize index: " +
                                     deserialize_result.error().message);
        }
        return index;
    }

    IndexPtr
    load_new_format(const std::string& index_path, const FooterPtr& footer) {
        auto metadata = footer->GetMetadata();
        auto basic_info = metadata->Get(BASIC_INFO);
        if (not basic_info.Contains(INDEX_PARAM)) {
            throw std::runtime_error("index metadata does not contain build parameters");
        }

        auto index_param = JsonType::Parse(basic_info[INDEX_PARAM].GetString());
        auto index_name = index_param[TYPE_KEY].GetString();
        if (index_name != INDEX_HGRAPH) {
            throw std::runtime_error("only HGraph index is supported, got: " + index_name);
        }

        IndexCommonParam index_common_params;
        index_common_params.dim_ = basic_info[DIM].GetInt();
        index_common_params.extra_info_size_ = basic_info["extra_info_size"].GetInt();
        index_common_params.data_type_ = static_cast<DataTypes>(basic_info["data_type"].GetInt());
        index_common_params.metric_ = static_cast<MetricType>(basic_info["metric"].GetInt());
        index_common_params.allocator_ = Engine::CreateDefaultAllocator();

        auto hgraph_parameter = std::make_shared<HGraphParameter>();
        hgraph_parameter->FromJson(index_param);
        hgraph_parameter->data_type = index_common_params.data_type_;

        auto inner_index = std::make_shared<HGraph>(hgraph_parameter, index_common_params);
        auto index = std::make_shared<IndexImpl<HGraph>>(inner_index, index_common_params);

        std::ifstream input(index_path, std::ios::binary);
        if (not input.is_open()) {
            throw std::runtime_error("failed to reopen index file: " + index_path);
        }
        auto deserialize_result = index->Deserialize(input);
        if (not deserialize_result.has_value()) {
            throw std::runtime_error("failed to deserialize index: " +
                                     deserialize_result.error().message);
        }
        return index;
    }

private:
    std::string build_param_;
};

void
print_samples(const std::vector<int64_t>& values, const std::string& title, int sample_limit) {
    std::cout << title << ": " << values.size() << std::endl;
    const auto count = std::min<int>(sample_limit, static_cast<int>(values.size()));
    for (int i = 0; i < count; ++i) {
        std::cout << "  [" << i << "] " << values[static_cast<size_t>(i)] << std::endl;
    }
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        argparse::ArgumentParser parser("compare_hgraph_ids");
        parse_args(parser, argc, argv);

        const auto index_path = parser.get<std::string>("--index_path");
        const auto label_path = parser.get<std::string>("--label_path");
        const auto build_param = parser.get<std::string>("--build_parameter");
        const auto sample_limit = parser.get<int>("--sample_limit");

        auto labels = read_labels(label_path);
        HGraphIndexLoader loader(build_param);
        auto index = loader.Load(index_path);
        auto export_ids_result = index->ExportIDs();
        if (not export_ids_result.has_value()) {
            throw std::runtime_error("failed to export index ids: " +
                                     export_ids_result.error().message);
        }

        const auto* index_ids_ptr = export_ids_result.value()->GetIds();
        const auto index_count = export_ids_result.value()->GetNumElements();
        std::vector<int64_t> index_ids(index_ids_ptr, index_ids_ptr + index_count);

        std::sort(labels.begin(), labels.end());
        std::sort(index_ids.begin(), index_ids.end());

        std::vector<int64_t> only_in_labels;
        std::vector<int64_t> only_in_index;
        only_in_labels.reserve(4096);
        only_in_index.reserve(4096);

        size_t i = 0;
        size_t j = 0;
        uint64_t intersection_count = 0;
        while (i < labels.size() && j < index_ids.size()) {
            if (labels[i] == index_ids[j]) {
                ++intersection_count;
                ++i;
                ++j;
            } else if (labels[i] < index_ids[j]) {
                only_in_labels.push_back(labels[i]);
                ++i;
            } else {
                only_in_index.push_back(index_ids[j]);
                ++j;
            }
        }
        while (i < labels.size()) {
            only_in_labels.push_back(labels[i]);
            ++i;
        }
        while (j < index_ids.size()) {
            only_in_index.push_back(index_ids[j]);
            ++j;
        }

        std::cout << "label.bin rows: " << labels.size() << std::endl;
        std::cout << "index exported ids: " << index_ids.size() << std::endl;
        std::cout << "intersection count: " << intersection_count << std::endl;
        std::cout << "only in label.bin: " << only_in_labels.size() << std::endl;
        std::cout << "only in index: " << only_in_index.size() << std::endl;

        print_samples(only_in_labels, "sample labels only in label.bin", sample_limit);
        print_samples(only_in_index, "sample labels only in index", sample_limit);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
}