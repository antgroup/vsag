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
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "build_cache_tool_common.h"

using namespace vsag;

namespace {

enum class ExitCode {
    SUCCESS = 0,
    INVALID_ARGUMENT = 1,
    INPUT_IO_ERROR = 2,
    DATASET_ALIGNMENT_ERROR = 3,
    CREATE_INDEX_ERROR = 4,
    BUILD_WITH_CACHE_ERROR = 5,
    SERIALIZE_INDEX_ERROR = 6,
    EXPORT_CACHE_ERROR = 7,
    GET_STATS_ERROR = 8,
};

uint64_t
now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

double
us_to_seconds(uint64_t duration_us) {
    return static_cast<double>(duration_us) / 1000000.0;
}

void
log_stage(const std::string& message) {
    std::cerr << "[build_hgraph_with_cache] " << message << std::endl;
}

void
log_stage_elapsed(const std::string& stage_name,
                  uint64_t stage_duration_us,
                  uint64_t total_elapsed_us) {
    std::cerr << "[build_hgraph_with_cache] " << stage_name << " finished in "
              << us_to_seconds(stage_duration_us) << "s"
              << ", total_elapsed=" << us_to_seconds(total_elapsed_us) << "s" << std::endl;
}

void
parse_args(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument<std::string>("--cache_input_path", "-c")
        .required()
        .help("Input build cache file path");
    parser.add_argument<std::string>("--vector_path", "-v")
        .required()
        .help("Path to vector_data.bin");
    parser.add_argument<std::string>("--label_path", "-l")
        .required()
        .help("Path to label.bin");
    parser.add_argument<std::string>("--feature_id_path", "-f")
        .required()
        .help("Path to id_map");
    parser.add_argument<std::string>("--index_output_path", "-o")
        .required()
        .help("Output serialized index file path");
    parser.add_argument<std::string>("--build_parameter", "-bp")
        .required()
        .help("Full HGraph build JSON or a path to a JSON file");
    parser.add_argument<std::string>("--cache_output_path", "-co")
        .help("Optional output build cache file path");
    parser.add_argument<std::string>("--alignment_policy", "-ap")
        .default_value(std::string("truncate_to_min"))
        .help("How to handle row-count mismatch: strict or truncate_to_min");
    parser.add_argument("--dimension", "-d")
        .default_value(int64_t{-1})
        .help("Optional explicit dim override")
        .scan<'i', int64_t>();
    parser.add_argument("--hit_refine_rounds")
        .default_value(uint32_t{2})
        .help("Refine rounds for cache-hit nodes")
        .scan<'i', uint32_t>();
    parser.add_argument("--missed_refine_rounds")
        .default_value(uint32_t{4})
        .help("Refine rounds for cache-missed nodes")
        .scan<'i', uint32_t>();
    parser.add_argument("--enable_parallel_refine")
        .default_value(false)
        .implicit_value(true)
        .help("Enable round-internal parallel refine for BuildWithCache");
    parser.add_argument("--refine_parallelism")
        .default_value(uint32_t{0})
        .help("Requested refine worker count, 0 means reuse build_thread_count")
        .scan<'i', uint32_t>();
    parser.add_argument("--print_failed_ids_limit")
        .default_value(size_t{20})
        .help("Maximum number of failed ids to print")
        .scan<'i', size_t>();
    parser.add_argument("--disable_warm_start")
        .default_value(false)
        .implicit_value(true)
        .help("Disable warm start and fall back to normal Build through BuildWithCache option");
    parser.add_argument("--keep_invalid_neighbors")
        .default_value(false)
        .implicit_value(true)
        .help("Keep invalid mapped neighbors instead of dropping them");
    parser.add_argument("--skip_route_graph_build")
        .default_value(false)
        .implicit_value(true)
        .help("Skip route_graph construction and only warm start bottom_graph");
    parser.add_argument("--log_stats_as_json")
        .default_value(false)
        .implicit_value(true)
        .help("Print BuildCacheStats as JSON");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        throw;
    }
}

std::optional<std::string>
get_optional_string_arg(const argparse::ArgumentParser& parser, const std::string& name) {
    if (parser.is_used(name)) {
        return parser.get<std::string>(name);
    }
    return std::nullopt;
}

JsonType
build_stats_json(const tools::DatasetAlignmentStats& import_stats,
                 const BuildCacheStats& cache_stats,
                 uint64_t dataset_load_us,
                 uint64_t build_us,
                 uint64_t serialize_index_us,
                 const std::vector<int64_t>& failed_ids,
                 size_t failed_id_limit,
                 const std::string& index_output_path,
                 const std::optional<std::string>& cache_output_path) {
    JsonType json;
    json["raw_label_count"].SetInt(import_stats.raw_label_count);
    json["raw_feature_id_count"].SetInt(import_stats.raw_feature_id_count);
    json["raw_vector_count"].SetInt(import_stats.raw_vector_count);
    json["aligned_count"].SetInt(import_stats.aligned_count);
    json["was_truncated"].SetBool(import_stats.was_truncated);
    json["failed_id_count"].SetInt(failed_ids.size());
    json["total_nodes"].SetInt(cache_stats.total_nodes);
    json["cached_nodes"].SetInt(cache_stats.cached_nodes);
    json["hit_nodes"].SetInt(cache_stats.hit_nodes);
    json["missed_nodes"].SetInt(cache_stats.missed_nodes);
    json["hit_seed_neighbor_total"].SetInt(cache_stats.hit_seed_neighbor_total);
    json["missed_seed_neighbor_total"].SetInt(cache_stats.missed_seed_neighbor_total);
    json["hit_empty_seed_nodes"].SetInt(cache_stats.hit_empty_seed_nodes);
    json["missed_empty_seed_nodes"].SetInt(cache_stats.missed_empty_seed_nodes);
    json["invalid_neighbors"].SetInt(cache_stats.invalid_neighbors);
    json["dropped_neighbors"].SetInt(cache_stats.dropped_neighbors);
    json["hit_refine_rounds"].SetInt(cache_stats.hit_refine_rounds);
    json["missed_refine_rounds"].SetInt(cache_stats.missed_refine_rounds);
    json["hit_refine_parallelism"].SetInt(cache_stats.hit_refine_parallelism);
    json["missed_refine_parallelism"].SetInt(cache_stats.missed_refine_parallelism);
    json["cache_load_us"].SetInt(cache_stats.cache_load_us);
    json["warm_start_apply_us"].SetInt(cache_stats.warm_start_apply_us);
    json["hit_refine_us"].SetInt(cache_stats.hit_refine_us);
    json["missed_refine_us"].SetInt(cache_stats.missed_refine_us);
    json["route_graph_build_us"].SetInt(cache_stats.route_graph_build_us);
    json["route_graph_levels"].SetInt(cache_stats.route_graph_levels);
    json["cache_hit_rate"].SetFloat(cache_stats.cache_hit_rate);
    json["dataset_load_us"].SetInt(dataset_load_us);
    json["build_us"].SetInt(build_us);
    json["serialize_index_us"].SetInt(serialize_index_us);
    json["index_output_path"].SetString(index_output_path);
    if (cache_output_path.has_value()) {
        json["cache_output_path"].SetString(*cache_output_path);
    }
    const auto sample_count = std::min(failed_ids.size(), failed_id_limit);
    for (size_t i = 0; i < sample_count; ++i) {
        json["failed_ids_sample"][std::to_string(i)].SetInt(static_cast<uint64_t>(failed_ids[i]));
    }
    return json;
}

void
print_stats_text(const tools::DatasetAlignmentStats& import_stats,
                 const BuildCacheStats& cache_stats,
                 uint64_t dataset_load_us,
                 uint64_t build_us,
                 uint64_t serialize_index_us,
                 const std::vector<int64_t>& failed_ids,
                 size_t failed_id_limit,
                 const std::string& index_output_path,
                 const std::optional<std::string>& cache_output_path) {
    std::cout << "raw_label_count=" << import_stats.raw_label_count << std::endl;
    std::cout << "raw_feature_id_count=" << import_stats.raw_feature_id_count << std::endl;
    std::cout << "raw_vector_count=" << import_stats.raw_vector_count << std::endl;
    std::cout << "aligned_count=" << import_stats.aligned_count << std::endl;
    std::cout << "was_truncated=" << (import_stats.was_truncated ? "true" : "false")
              << std::endl;
    std::cout << "failed_id_count=" << failed_ids.size() << std::endl;
    if (not failed_ids.empty()) {
        std::cout << "failed_ids_sample=";
        const auto sample_count = std::min(failed_ids.size(), failed_id_limit);
        for (size_t i = 0; i < sample_count; ++i) {
            if (i != 0) {
                std::cout << ',';
            }
            std::cout << failed_ids[i];
        }
        std::cout << std::endl;
    }
    std::cout << "cache_hit_rate=" << cache_stats.cache_hit_rate << std::endl;
    std::cout << "hit_nodes=" << cache_stats.hit_nodes << std::endl;
    std::cout << "missed_nodes=" << cache_stats.missed_nodes << std::endl;
    std::cout << "hit_seed_neighbor_total=" << cache_stats.hit_seed_neighbor_total << std::endl;
    std::cout << "missed_seed_neighbor_total=" << cache_stats.missed_seed_neighbor_total
              << std::endl;
    std::cout << "hit_empty_seed_nodes=" << cache_stats.hit_empty_seed_nodes << std::endl;
    std::cout << "missed_empty_seed_nodes=" << cache_stats.missed_empty_seed_nodes << std::endl;
    std::cout << "cache_load_us=" << cache_stats.cache_load_us << std::endl;
    std::cout << "warm_start_apply_us=" << cache_stats.warm_start_apply_us << std::endl;
    std::cout << "hit_refine_us=" << cache_stats.hit_refine_us << std::endl;
    std::cout << "missed_refine_us=" << cache_stats.missed_refine_us << std::endl;
    std::cout << "hit_refine_parallelism=" << cache_stats.hit_refine_parallelism << std::endl;
    std::cout << "missed_refine_parallelism=" << cache_stats.missed_refine_parallelism
              << std::endl;
    std::cout << "route_graph_build_us=" << cache_stats.route_graph_build_us << std::endl;
    std::cout << "route_graph_levels=" << cache_stats.route_graph_levels << std::endl;
    std::cout << "dataset_load_us=" << dataset_load_us << std::endl;
    std::cout << "build_us=" << build_us << std::endl;
    std::cout << "serialize_index_us=" << serialize_index_us << std::endl;
    std::cout << "index_output_path=" << index_output_path << std::endl;
    if (cache_output_path.has_value()) {
        std::cout << "cache_output_path=" << *cache_output_path << std::endl;
    }
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        const auto total_begin = now_us();
        argparse::ArgumentParser parser("build_hgraph_with_cache");
        parse_args(parser, argc, argv);

        const auto cache_input_path = parser.get<std::string>("--cache_input_path");
        const auto vector_path = parser.get<std::string>("--vector_path");
        const auto label_path = parser.get<std::string>("--label_path");
        const auto feature_id_path = parser.get<std::string>("--feature_id_path");
        const auto index_output_path = parser.get<std::string>("--index_output_path");
        const auto build_parameter_source = parser.get<std::string>("--build_parameter");
        const auto cache_output_path = get_optional_string_arg(parser, "--cache_output_path");
        const auto alignment_policy =
            tools::ParseAlignmentPolicy(parser.get<std::string>("--alignment_policy"));
        const auto dimension_override = parser.get<int64_t>("--dimension");
        const auto hit_refine_rounds = parser.get<uint32_t>("--hit_refine_rounds");
        const auto missed_refine_rounds = parser.get<uint32_t>("--missed_refine_rounds");
        const auto enable_parallel_refine = parser.get<bool>("--enable_parallel_refine");
        const auto refine_parallelism = parser.get<uint32_t>("--refine_parallelism");
        const auto failed_ids_limit = parser.get<size_t>("--print_failed_ids_limit");
        const auto disable_warm_start = parser.get<bool>("--disable_warm_start");
        const auto keep_invalid_neighbors = parser.get<bool>("--keep_invalid_neighbors");
        const auto skip_route_graph_build = parser.get<bool>("--skip_route_graph_build");
        const auto log_stats_as_json = parser.get<bool>("--log_stats_as_json");

        const auto build_config = tools::ParseBuildParameterSource(build_parameter_source);
        const auto dim = dimension_override > 0 ? dimension_override : build_config.dim;
        if (build_config.dtype != "float32") {
            std::cerr << "only float32 dataset is supported in build_hgraph_with_cache, got dtype="
                      << build_config.dtype << std::endl;
            return static_cast<int>(ExitCode::INVALID_ARGUMENT);
        }

        log_stage("starting prebuild"
                  " cache_input_path=" + cache_input_path +
                  " index_output_path=" + index_output_path +
                  " dim=" + std::to_string(dim) +
                  " hit_refine_rounds=" + std::to_string(hit_refine_rounds) +
                  " missed_refine_rounds=" + std::to_string(missed_refine_rounds) +
                  " enable_parallel_refine=" +
                  (enable_parallel_refine ? std::string("true") : std::string("false")) +
                  " refine_parallelism=" + std::to_string(refine_parallelism) +
                  " warm_start=" + (disable_warm_start ? std::string("false") : std::string("true")) +
                  " build_route_graph=" +
                  (skip_route_graph_build ? std::string("false") : std::string("true")));
        log_stage("loading dataset from input files");

        const auto dataset_load_begin = now_us();
        tools::ImportedDataset imported_dataset;
        try {
            imported_dataset = tools::LoadFloat32DatasetFromFiles(
                label_path, feature_id_path, vector_path, dim, alignment_policy);
        } catch (const std::runtime_error& ex) {
            std::cerr << ex.what() << std::endl;
            const std::string message = ex.what();
            if (message.find("row count mismatch") != std::string::npos ||
                message.find("aligned dataset is empty") != std::string::npos ||
                message.find("duplicate feature_id") != std::string::npos ||
                message.find("contains empty feature id") != std::string::npos) {
                return static_cast<int>(ExitCode::DATASET_ALIGNMENT_ERROR);
            }
            return static_cast<int>(ExitCode::INPUT_IO_ERROR);
        }
        const auto dataset_load_us = now_us() - dataset_load_begin;
        log_stage_elapsed("dataset_load", dataset_load_us, now_us() - total_begin);
        log_stage("dataset rows aligned=" + std::to_string(imported_dataset.stats.aligned_count) +
                  " truncated=" +
                  (imported_dataset.stats.was_truncated ? std::string("true")
                                                       : std::string("false")));

        auto create_result = Factory::CreateIndex("hgraph", build_config.raw_json);
        if (not create_result.has_value()) {
            std::cerr << "failed to create HGraph index: " << create_result.error().message
                      << std::endl;
            return static_cast<int>(ExitCode::CREATE_INDEX_ERROR);
        }
        auto index = create_result.value();

        BuildCacheOptions options;
        options.enable_warm_start = not disable_warm_start;
        options.hit_refine_rounds = hit_refine_rounds;
        options.missed_refine_rounds = missed_refine_rounds;
        options.enable_parallel_refine = enable_parallel_refine;
        options.refine_parallelism = refine_parallelism;
        options.drop_invalid_neighbors = not keep_invalid_neighbors;
        options.build_route_graph = not skip_route_graph_build;

        std::ifstream cache_input(cache_input_path, std::ios::binary);
        if (not cache_input.is_open()) {
            std::cerr << "failed to open cache input file: " << cache_input_path << std::endl;
            return static_cast<int>(ExitCode::INPUT_IO_ERROR);
        }

        log_stage("starting BuildWithCache");
        const auto build_begin = now_us();
        auto build_result = index->BuildWithCache(imported_dataset.dataset, cache_input, options);
        const auto build_us = now_us() - build_begin;
        if (not build_result.has_value()) {
            std::cerr << "build with cache failed: " << build_result.error().message << std::endl;
            return static_cast<int>(ExitCode::BUILD_WITH_CACHE_ERROR);
        }
        const auto failed_ids = build_result.value();
        log_stage_elapsed("build_with_cache", build_us, now_us() - total_begin);
        log_stage("starting index serialization");

        const auto serialize_begin = now_us();
        std::ofstream index_output(index_output_path, std::ios::binary);
        if (not index_output.is_open()) {
            std::cerr << "failed to open index output file: " << index_output_path << std::endl;
            return static_cast<int>(ExitCode::SERIALIZE_INDEX_ERROR);
        }
        auto serialize_result = index->Serialize(index_output);
        const auto serialize_index_us = now_us() - serialize_begin;
        if (not serialize_result.has_value()) {
            std::cerr << "serialize index failed: " << serialize_result.error().message
                      << std::endl;
            return static_cast<int>(ExitCode::SERIALIZE_INDEX_ERROR);
        }
        log_stage_elapsed("serialize_index", serialize_index_us, now_us() - total_begin);

        if (cache_output_path.has_value()) {
            log_stage("starting cache export");
            const auto export_begin = now_us();
            std::ofstream cache_output(*cache_output_path, std::ios::binary);
            if (not cache_output.is_open()) {
                std::cerr << "failed to open cache output file: " << *cache_output_path
                          << std::endl;
                return static_cast<int>(ExitCode::EXPORT_CACHE_ERROR);
            }
            auto export_result = index->ExportBuildCache(cache_output);
            if (not export_result.has_value()) {
                std::cerr << "export build cache failed: " << export_result.error().message
                          << std::endl;
                return static_cast<int>(ExitCode::EXPORT_CACHE_ERROR);
            }
            log_stage_elapsed("export_cache", now_us() - export_begin, now_us() - total_begin);
        }

        auto stats_result = index->GetBuildCacheStats();
        if (not stats_result.has_value()) {
            std::cerr << "get build cache stats failed: " << stats_result.error().message
                      << std::endl;
            return static_cast<int>(ExitCode::GET_STATS_ERROR);
        }

        log_stage("refine summary"
                  " hit_nodes=" + std::to_string(stats_result.value().hit_nodes) +
                  " missed_nodes=" + std::to_string(stats_result.value().missed_nodes) +
                  " hit_seed_neighbor_total=" +
                  std::to_string(stats_result.value().hit_seed_neighbor_total) +
                  " missed_seed_neighbor_total=" +
                  std::to_string(stats_result.value().missed_seed_neighbor_total) +
                  " hit_empty_seed_nodes=" +
                  std::to_string(stats_result.value().hit_empty_seed_nodes) +
                  " missed_empty_seed_nodes=" +
                  std::to_string(stats_result.value().missed_empty_seed_nodes) +
                  " hit_refine_rounds=" +
                  std::to_string(stats_result.value().hit_refine_rounds) +
                  " missed_refine_rounds=" +
                  std::to_string(stats_result.value().missed_refine_rounds) +
                  " hit_refine_parallelism=" +
                  std::to_string(stats_result.value().hit_refine_parallelism) +
                  " missed_refine_parallelism=" +
                  std::to_string(stats_result.value().missed_refine_parallelism));

        if (log_stats_as_json) {
            auto json = build_stats_json(imported_dataset.stats,
                                         stats_result.value(),
                                         dataset_load_us,
                                         build_us,
                                         serialize_index_us,
                                         failed_ids,
                                         failed_ids_limit,
                                         index_output_path,
                                         cache_output_path);
            std::cout << json.Dump() << std::endl;
        } else {
            print_stats_text(imported_dataset.stats,
                             stats_result.value(),
                             dataset_load_us,
                             build_us,
                             serialize_index_us,
                             failed_ids,
                             failed_ids_limit,
                             index_output_path,
                             cache_output_path);
        }

        log_stage_elapsed("total_prebuild", now_us() - total_begin, now_us() - total_begin);

        return static_cast<int>(ExitCode::SUCCESS);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return static_cast<int>(ExitCode::INVALID_ARGUMENT);
    }
}