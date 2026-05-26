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

#include <argparse/argparse.hpp>
#include <fstream>
#include <iostream>
#include <string>

#include "algorithm/hgraph/hgraph.h"
#include "build_cache_tool_common.h"
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
        .help("Path to label.bin");

    parser.add_argument<std::string>("--feature_id_path", "-f")
        .help("Path to id_map");

    parser.add_argument<std::string>("--cache_output_path", "-o")
        .required()
        .help("Output cache file path");

    parser.add_argument<std::string>("--build_parameter", "-bp")
        .default_value(std::string(DEFAULT_BUILD_PARAM))
        .help("Optional full build parameter JSON, required when the index uses old serialization format");

    parser.add_argument<std::string>("--alignment_policy", "-ap")
        .default_value(std::string("truncate_to_min"))
        .help("How to handle label.bin and id_map row-count mismatch: strict or truncate_to_min");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        throw;
    }
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

    IndexPtr
    load_with_factory(const std::string& index_path, const std::string& build_param) {
        auto create_result = Factory::CreateIndex(INDEX_HGRAPH, build_param);
        if (not create_result.has_value()) {
            throw std::runtime_error("failed to create HGraph index: " +
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

private:
    std::string build_param_;
};

}  // namespace

int
main(int argc, char** argv) {
    try {
        argparse::ArgumentParser parser("export_hgraph_build_cache");
        parse_args(parser, argc, argv);

        const auto index_path = parser.get<std::string>("--index_path");
        const auto cache_output_path = parser.get<std::string>("--cache_output_path");
        const auto build_param = parser.get<std::string>("--build_parameter");
        const auto alignment_policy =
            tools::ParseAlignmentPolicy(parser.get<std::string>("--alignment_policy"));

        HGraphIndexLoader loader(build_param);
        auto index = loader.Load(index_path);

        const bool has_label_path = parser.is_used("--label_path");
        const bool has_feature_id_path = parser.is_used("--feature_id_path");
        if (has_label_path != has_feature_id_path) {
            std::cerr << "--label_path and --feature_id_path must be provided together"
                      << std::endl;
            return 2;
        }

        if (has_label_path && has_feature_id_path) {
            const auto label_path = parser.get<std::string>("--label_path");
            const auto feature_id_path = parser.get<std::string>("--feature_id_path");
            auto imported_dataset =
                tools::LoadFeatureIdDatasetFromFiles(label_path, feature_id_path, alignment_policy);

            if (imported_dataset.stats.was_truncated) {
                std::cerr << "warning: label.bin and id_map row counts mismatch, truncating to "
                          << imported_dataset.stats.aligned_count << " rows" << std::endl;
            }

            auto prepare_result = index->PrepareFeatureIdsForBuildCache(imported_dataset.dataset);
            if (not prepare_result.has_value()) {
                std::cerr << "prepare feature ids failed: " << prepare_result.error().message
                          << std::endl;
                return 2;
            }
        }

        std::ofstream output(cache_output_path, std::ios::binary);
        if (not output.is_open()) {
            std::cerr << "failed to open cache output file: " << cache_output_path << std::endl;
            return 3;
        }

        auto export_result = index->ExportBuildCache(output);
        if (not export_result.has_value()) {
            std::cerr << "export build cache failed: " << export_result.error().message
                      << std::endl;
            return 4;
        }

        std::cout << "cache exported successfully to " << cache_output_path << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }
}