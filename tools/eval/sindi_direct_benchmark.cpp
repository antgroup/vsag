#include <unistd.h>
#include <vsag/options.h>
#include <vsag/vsag.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.h"
#include "eval_dataset.h"

namespace {

using Clock = std::chrono::steady_clock;
using JsonType = nlohmann::json;

constexpr double kRecallThresholdError = 2e-6;

struct Arguments {
    std::string mode;
    std::string datapath;
    std::string index_path;
    std::string build_parameter;
    std::string search_parameter;
    std::string output_json;
    uint64_t topk{10};
    uint64_t query_count{10000};
    uint64_t sample_ms{50};
};

uint64_t
read_rss_bytes() {
    std::ifstream statm("/proc/self/statm");
    uint64_t total_pages = 0;
    uint64_t resident_pages = 0;
    statm >> total_pages >> resident_pages;
    return resident_pages * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
}

class MemorySampler {
public:
    explicit MemorySampler(uint64_t sample_ms) : sample_ms_(sample_ms) {
    }

    void
    Start() {
        baseline_rss_bytes_ = read_rss_bytes();
        peak_rss_bytes_ = baseline_rss_bytes_;
        running_.store(true);
        sampler_ = std::thread([this]() {
            while (running_.load()) {
                SampleOnce();
                std::this_thread::sleep_for(std::chrono::milliseconds(sample_ms_));
            }
        });
    }

    void
    Stop() {
        running_.store(false);
        if (sampler_.joinable()) {
            sampler_.join();
        }
        SampleOnce();
    }

    uint64_t
    BaselineRssBytes() const {
        return baseline_rss_bytes_;
    }

    uint64_t
    PeakRssBytes() const {
        return peak_rss_bytes_;
    }

    uint64_t
    PeakDeltaBytes() const {
        if (peak_rss_bytes_ <= baseline_rss_bytes_) {
            return 0;
        }
        return peak_rss_bytes_ - baseline_rss_bytes_;
    }

private:
    void
    SampleOnce() {
        auto current_rss_bytes = read_rss_bytes();
        if (current_rss_bytes > peak_rss_bytes_) {
            peak_rss_bytes_ = current_rss_bytes;
        }
    }

private:
    uint64_t sample_ms_{50};
    uint64_t baseline_rss_bytes_{0};
    uint64_t peak_rss_bytes_{0};
    std::atomic<bool> running_{false};
    std::thread sampler_;
};

void
print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --mode build|search|profile"
              << " --datapath <hdf5> --index-path <file>"
              << " --build-parameter <json> --output-json <file>"
              << " [--search-parameter <json>] [--topk 10]"
              << " [--query-count 10000] [--sample-ms 50]" << std::endl;
}

std::unordered_map<std::string, std::string>
parse_raw_args(int argc, char** argv) {
    std::unordered_map<std::string, std::string> values;
    for (int arg_index = 1; arg_index < argc; ++arg_index) {
        std::string key = argv[arg_index];
        if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (key.rfind("--", 0) != 0) {
            throw std::invalid_argument("unexpected argument: " + key);
        }
        if (arg_index + 1 >= argc) {
            throw std::invalid_argument("missing value for argument: " + key);
        }
        values[key] = argv[++arg_index];
    }
    return values;
}

std::string
require_arg(const std::unordered_map<std::string, std::string>& values, const std::string& key) {
    auto iterator = values.find(key);
    if (iterator == values.end() || iterator->second.empty()) {
        throw std::invalid_argument("missing required argument: " + key);
    }
    return iterator->second;
}

std::string
optional_arg(const std::unordered_map<std::string, std::string>& values,
             const std::string& key,
             const std::string& default_value) {
    auto iterator = values.find(key);
    if (iterator == values.end()) {
        return default_value;
    }
    return iterator->second;
}

uint64_t
parse_uint64(const std::string& value, const std::string& key) {
    try {
        return std::stoull(value);
    } catch (const std::exception& exception) {
        throw std::invalid_argument("invalid integer for " + key + ": " + exception.what());
    }
}

Arguments
parse_args(int argc, char** argv) {
    auto values = parse_raw_args(argc, argv);
    Arguments args;
    args.mode = require_arg(values, "--mode");
    args.datapath = require_arg(values, "--datapath");
    args.index_path = require_arg(values, "--index-path");
    args.build_parameter = require_arg(values, "--build-parameter");
    args.output_json = require_arg(values, "--output-json");
    args.search_parameter = optional_arg(values, "--search-parameter", "");
    args.topk = parse_uint64(optional_arg(values, "--topk", "10"), "--topk");
    args.query_count =
        parse_uint64(optional_arg(values, "--query-count", "10000"), "--query-count");
    args.sample_ms = parse_uint64(optional_arg(values, "--sample-ms", "50"), "--sample-ms");
    if (args.sample_ms == 0) {
        args.sample_ms = 50;
    }
    return args;
}

double
elapsed_seconds(const Clock::time_point& start_time, const Clock::time_point& end_time) {
    return std::chrono::duration<double>(end_time - start_time).count();
}

JsonType
dataset_info(const vsag::eval::EvalDatasetPtr& dataset) {
    JsonType info;
    info["filepath"] = dataset->GetFilePath();
    info["vector_type"] = dataset->GetVectorType();
    info["dim"] = dataset->GetDim();
    info["base_count"] = dataset->GetNumberOfBase();
    info["query_count"] = dataset->GetNumberOfQuery();
    info["train_data_type"] = dataset->GetTrainDataType();
    info["test_data_type"] = dataset->GetTestDataType();
    return info;
}

vsag::IndexPtr
create_sindi_index(const std::string& build_parameter) {
    auto index_result = vsag::Factory::CreateIndex("sindi", build_parameter);
    if (!index_result.has_value()) {
        throw std::runtime_error(index_result.error().message);
    }
    return index_result.value();
}

JsonType
index_memory_detail(const vsag::IndexPtr& index) {
    JsonType detail;
    try {
        auto detail_text = index->GetMemoryUsageDetail();
        if (!detail_text.empty()) {
            detail = JsonType::parse(detail_text);
        }
    } catch (const std::exception& exception) {
        detail["error"] = exception.what();
    }
    return detail;
}

void
ensure_sparse_dataset(const vsag::eval::EvalDatasetPtr& dataset) {
    if (dataset->GetVectorType() != vsag::eval::SPARSE_VECTORS) {
        throw std::runtime_error("dataset is not sparse: " + dataset->GetVectorType());
    }
}

void
ensure_parent_dir(const std::string& path) {
    auto parent_path = std::filesystem::path(path).parent_path();
    if (!parent_path.empty()) {
        std::filesystem::create_directories(parent_path);
    }
}

void
write_json_file(const std::string& path, const JsonType& result) {
    ensure_parent_dir(path);
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open output json: " + path);
    }
    output << result.dump(2) << '\n';
}

double
percentile(std::vector<double> values, double rate) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    auto position = static_cast<uint64_t>(rate * static_cast<double>(values.size() - 1));
    return values[position];
}

double
average(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    auto sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double
calculate_recall(const vsag::DatasetPtr& result,
                 const vsag::eval::EvalDatasetPtr& dataset,
                 uint64_t query_index,
                 uint64_t topk) {
    if (topk == 0) {
        return 0.0;
    }
    auto distance_func = dataset->GetDistanceFunc();
    size_t dim = static_cast<size_t>(dataset->GetDim());
    auto* ground_truth_ids = dataset->GetNeighbors(static_cast<int64_t>(query_index));
    auto* query_data = dataset->GetOneTest(static_cast<int64_t>(query_index));

    std::vector<float> ground_truth_distances;
    ground_truth_distances.reserve(topk);
    for (uint64_t result_index = 0; result_index < topk; ++result_index) {
        auto base_id = ground_truth_ids[result_index];
        ground_truth_distances.emplace_back(
            distance_func(query_data, dataset->GetOneTrain(base_id), &dim));
    }
    std::sort(ground_truth_distances.begin(), ground_truth_distances.end());
    auto threshold = ground_truth_distances[topk - 1];

    auto* result_ids = result->GetIds();
    auto result_count = std::min<int64_t>(result->GetDim(), static_cast<int64_t>(topk));

    uint64_t hit_count = 0;
    for (int64_t result_index = 0; result_index < result_count; ++result_index) {
        auto base_id = result_ids[result_index];
        if (base_id < 0 || base_id >= dataset->GetNumberOfBase()) {
            continue;
        }
        auto* base_data = dataset->GetOneTrain(base_id);
        auto distance = distance_func(query_data, base_data, &dim);
        if (distance <= threshold + kRecallThresholdError) {
            ++hit_count;
        }
    }
    return static_cast<double>(hit_count) / static_cast<double>(topk);
}

JsonType
run_build(const Arguments& args) {
    auto dataset = vsag::eval::EvalDataset::Load(args.datapath);
    ensure_sparse_dataset(dataset);
    auto index = create_sindi_index(args.build_parameter);

    auto total_base = dataset->GetNumberOfBase();
    std::vector<int64_t> ids(static_cast<uint64_t>(total_base));
    std::iota(ids.begin(), ids.end(), 0);

    auto base = vsag::Dataset::Make();
    base->NumElements(total_base)
        ->Dim(dataset->GetDim())
        ->Ids(ids.data())
        ->SparseVectors(static_cast<const vsag::SparseVector*>(dataset->GetTrain()))
        ->Owner(false);

    MemorySampler sampler(args.sample_ms);
    sampler.Start();
    auto start_time = Clock::now();
    auto build_result = index->Build(base);
    auto end_time = Clock::now();
    sampler.Stop();

    if (!build_result.has_value()) {
        throw std::runtime_error(build_result.error().message);
    }

    ensure_parent_dir(args.index_path);
    std::ofstream index_output(args.index_path, std::ios::binary);
    if (!index_output) {
        throw std::runtime_error("failed to open index path: " + args.index_path);
    }
    auto serialize_result = index->Serialize(index_output);
    if (!serialize_result.has_value()) {
        throw std::runtime_error(serialize_result.error().message);
    }
    index_output.close();

    auto build_seconds = elapsed_seconds(start_time, end_time);
    auto failed_ids = build_result.value();

    JsonType result;
    result["mode"] = "build";
    result["dataset"] = dataset_info(dataset);
    result["index_path"] = args.index_path;
    result["build_parameter"] = JsonType::parse(args.build_parameter);
    result["build_seconds"] = build_seconds;
    result["build_tps"] = static_cast<double>(total_base) / build_seconds;
    result["failed_id_count"] = failed_ids.size();
    result["index_memory_bytes"] = index->GetMemoryUsage();
    result["index_memory_detail"] = index_memory_detail(index);
    result["peak_rss_bytes"] = sampler.PeakRssBytes();
    result["peak_rss_delta_bytes"] = sampler.PeakDeltaBytes();
    result["baseline_rss_bytes"] = sampler.BaselineRssBytes();
    result["sample_ms"] = args.sample_ms;
    if (std::filesystem::exists(args.index_path)) {
        result["index_file_bytes"] = std::filesystem::file_size(args.index_path);
    }
    return result;
}

JsonType
run_search(const Arguments& args) {
    if (args.search_parameter.empty()) {
        throw std::invalid_argument("--search-parameter is required in search mode");
    }

    auto dataset = vsag::eval::EvalDataset::Load(args.datapath);
    ensure_sparse_dataset(dataset);
    auto index = create_sindi_index(args.build_parameter);

    MemorySampler sampler(args.sample_ms);
    sampler.Start();

    std::ifstream index_input(args.index_path, std::ios::binary);
    if (!index_input) {
        throw std::runtime_error("failed to open index path: " + args.index_path);
    }
    auto load_start_time = Clock::now();
    auto deserialize_result = index->Deserialize(index_input);
    auto load_end_time = Clock::now();
    if (!deserialize_result.has_value()) {
        throw std::runtime_error(deserialize_result.error().message);
    }

    auto available_queries = static_cast<uint64_t>(dataset->GetNumberOfQuery());
    auto run_queries = args.query_count == 0 ? available_queries : args.query_count;
    if (available_queries == 0 || run_queries == 0) {
        throw std::runtime_error("dataset has no query vectors");
    }

    std::vector<double> latencies_ms;
    std::vector<double> recalls;
    latencies_ms.reserve(run_queries);
    recalls.reserve(run_queries);

    auto search_start_time = Clock::now();
    for (uint64_t query_offset = 0; query_offset < run_queries; ++query_offset) {
        auto query_index = query_offset % available_queries;
        auto* query_vector = static_cast<const vsag::SparseVector*>(
            dataset->GetOneTest(static_cast<int64_t>(query_index)));
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(dataset->GetDim())->SparseVectors(query_vector)->Owner(false);

        auto query_start_time = Clock::now();
        auto search_result =
            index->KnnSearch(query, static_cast<int64_t>(args.topk), args.search_parameter);
        auto query_end_time = Clock::now();
        if (!search_result.has_value()) {
            throw std::runtime_error(search_result.error().message);
        }

        auto latency_ms =
            std::chrono::duration<double, std::milli>(query_end_time - query_start_time).count();
        latencies_ms.emplace_back(latency_ms);
        recalls.emplace_back(
            calculate_recall(search_result.value(), dataset, query_index, args.topk));
    }
    auto search_end_time = Clock::now();
    sampler.Stop();

    auto total_latency_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
    auto qps = static_cast<double>(run_queries) * 1000.0 / total_latency_ms;

    JsonType latency_detail;
    latency_detail["p50"] = percentile(latencies_ms, 0.50);
    latency_detail["p80"] = percentile(latencies_ms, 0.80);
    latency_detail["p90"] = percentile(latencies_ms, 0.90);
    latency_detail["p95"] = percentile(latencies_ms, 0.95);
    latency_detail["p99"] = percentile(latencies_ms, 0.99);

    JsonType recall_detail;
    recall_detail["p0"] = percentile(recalls, 0.00);
    recall_detail["p10"] = percentile(recalls, 0.10);
    recall_detail["p30"] = percentile(recalls, 0.30);
    recall_detail["p50"] = percentile(recalls, 0.50);
    recall_detail["p70"] = percentile(recalls, 0.70);
    recall_detail["p90"] = percentile(recalls, 0.90);

    JsonType result;
    result["mode"] = "search";
    result["dataset"] = dataset_info(dataset);
    result["index_path"] = args.index_path;
    result["build_parameter"] = JsonType::parse(args.build_parameter);
    result["search_parameter"] = JsonType::parse(args.search_parameter);
    result["topk"] = args.topk;
    result["search_query_count"] = run_queries;
    result["load_seconds"] = elapsed_seconds(load_start_time, load_end_time);
    result["search_wall_seconds"] = elapsed_seconds(search_start_time, search_end_time);
    result["qps"] = qps;
    result["recall_avg"] = average(recalls);
    result["recall_detail"] = recall_detail;
    result["latency_avg_ms"] = average(latencies_ms);
    result["latency_detail_ms"] = latency_detail;
    result["index_memory_bytes"] = index->GetMemoryUsage();
    result["index_memory_detail"] = index_memory_detail(index);
    result["peak_rss_bytes"] = sampler.PeakRssBytes();
    result["peak_rss_delta_bytes"] = sampler.PeakDeltaBytes();
    result["baseline_rss_bytes"] = sampler.BaselineRssBytes();
    result["sample_ms"] = args.sample_ms;
    return result;
}

JsonType
run_profile(const Arguments& args) {
    if (args.search_parameter.empty()) {
        throw std::invalid_argument("--search-parameter is required in profile mode");
    }

    auto dataset = vsag::eval::EvalDataset::Load(args.datapath);
    ensure_sparse_dataset(dataset);
    auto index = create_sindi_index(args.build_parameter);

    MemorySampler sampler(args.sample_ms);
    sampler.Start();

    std::ifstream index_input(args.index_path, std::ios::binary);
    if (!index_input) {
        throw std::runtime_error("failed to open index path: " + args.index_path);
    }
    auto load_start_time = Clock::now();
    auto deserialize_result = index->Deserialize(index_input);
    auto load_end_time = Clock::now();
    if (!deserialize_result.has_value()) {
        throw std::runtime_error(deserialize_result.error().message);
    }

    auto available_queries = static_cast<uint64_t>(dataset->GetNumberOfQuery());
    auto run_queries = args.query_count == 0 ? available_queries : args.query_count;
    if (available_queries == 0 || run_queries == 0) {
        throw std::runtime_error("dataset has no query vectors");
    }

    std::vector<vsag::SparseVector> profile_queries;
    profile_queries.reserve(run_queries);
    for (uint64_t query_offset = 0; query_offset < run_queries; ++query_offset) {
        auto query_index = query_offset % available_queries;
        profile_queries.push_back(*static_cast<const vsag::SparseVector*>(
            dataset->GetOneTest(static_cast<int64_t>(query_index))));
    }

    auto query = vsag::Dataset::Make();
    query->NumElements(static_cast<int64_t>(profile_queries.size()))
        ->Dim(dataset->GetDim())
        ->SparseVectors(profile_queries.data())
        ->Owner(false);

    JsonType profile_parameter = JsonType::parse(args.search_parameter);
    profile_parameter["analyze"]["profile_search_pipeline"] = true;
    profile_parameter["analyze"]["profile_query_count"] = run_queries;

    vsag::SearchRequest request;
    request.query_ = query;
    request.topk_ = static_cast<int64_t>(args.topk);
    request.params_str_ = profile_parameter.dump();

    auto profile_start_time = Clock::now();
    auto profile_text = index->AnalyzeIndexBySearch(request);
    auto profile_end_time = Clock::now();
    sampler.Stop();

    JsonType result;
    result["mode"] = "profile";
    result["dataset"] = dataset_info(dataset);
    result["index_path"] = args.index_path;
    result["build_parameter"] = JsonType::parse(args.build_parameter);
    result["search_parameter"] = JsonType::parse(args.search_parameter);
    result["topk"] = args.topk;
    result["profile_query_count"] = run_queries;
    result["load_seconds"] = elapsed_seconds(load_start_time, load_end_time);
    result["profile_wall_seconds"] = elapsed_seconds(profile_start_time, profile_end_time);
    result["profile"] = JsonType::parse(profile_text);
    result["index_memory_bytes"] = index->GetMemoryUsage();
    result["index_memory_detail"] = index_memory_detail(index);
    result["peak_rss_bytes"] = sampler.PeakRssBytes();
    result["peak_rss_delta_bytes"] = sampler.PeakDeltaBytes();
    result["baseline_rss_bytes"] = sampler.BaselineRssBytes();
    result["sample_ms"] = args.sample_ms;
    return result;
}

}  // namespace

int
main(int argc, char** argv) {
    try {
        vsag::init();
        vsag::Options::Instance().logger()->SetLevel(vsag::Logger::kOFF);

        auto args = parse_args(argc, argv);
        JsonType result;
        if (args.mode == "build") {
            result = run_build(args);
        } else if (args.mode == "search") {
            result = run_search(args);
        } else if (args.mode == "profile") {
            result = run_profile(args);
        } else {
            throw std::invalid_argument("--mode must be build, search, or profile");
        }

        write_json_file(args.output_json, result);
        std::cout << result.dump(2) << std::endl;
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "sindi_direct_benchmark error: " << exception.what() << std::endl;
        return 1;
    }
}
