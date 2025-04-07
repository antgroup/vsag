
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

#include <sys/stat.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <unordered_set>

#include "../eval/eval_dataset.h"
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"
#include "vsag/vsag.h"

using namespace nlohmann;
using namespace spdlog;
using namespace vsag;
using namespace vsag::eval;

static double
get_recall(const float* distances,
           const float* ground_truth_distances,
           size_t recall_num,
           size_t top_k) {
    std::vector<float> gt_distances(ground_truth_distances, ground_truth_distances + top_k);
    std::sort(gt_distances.begin(), gt_distances.end());
    float threshold = gt_distances[top_k - 1];
    size_t count = 0;
    for (size_t i = 0; i < recall_num; ++i) {
        if (distances[i] <= threshold + 2e-6) {
            ++count;
        }
    }
    return static_cast<double>(count) / static_cast<double>(top_k);
}

json
run_test(const std::string& index_name,
         const std::string& process,
         const std::string& build_parameters,
         const std::string& search_parameters,
         const std::string& dataset_path);

const static std::string DIR_NAME = "/tmp/test_performance/";
const static std::string META_DATA_FILE = "_meta.data";

int
main(int argc, char* argv[]) {
    set_level(level::off);
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_file_path> <process> <index_name> <build_param> <search_param>"
                  << std::endl;
        return -1;
    }

    std::string dataset_filename = argv[1];
    std::string process = argv[2];
    std::string index_name = argv[3];
    std::string build_parameters = argv[4];
    std::string search_parameters = argv[5];

    struct stat st;
    if (stat(DIR_NAME.c_str(), &st) != 0) {
        int status = mkdir(DIR_NAME.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if (status != 0) {
            std::cerr << "Error creating directory: " << DIR_NAME << std::endl;
            return -1;
        }
    }

    auto result =
        run_test(dataset_filename, process, index_name, build_parameters, search_parameters);
    spdlog::debug("done");
    std::cout << result.dump(4) << std::endl;

    return 0;
}

std::unordered_set<int64_t>
get_intersection(const int64_t* neighbors,
                 const int64_t* ground_truth,
                 size_t recall_num,
                 size_t top_k) {
    std::unordered_set<int64_t> neighbors_set(neighbors, neighbors + recall_num);
    std::unordered_set<int64_t> intersection;
    for (size_t i = 0; i < top_k; ++i) {
        if (i < top_k && neighbors_set.count(ground_truth[i])) {
            intersection.insert(ground_truth[i]);
        }
    }
    return intersection;
}

class PerfTools {
public:
    static json
    Build(const std::string& dataset_path,
          const std::string& index_name,
          const std::string& build_parameters) {
        spdlog::debug("index_name: " + index_name);
        spdlog::debug("build_parameters: " + build_parameters);
        vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
        vsag::Engine e(&resource);
        auto index = e.CreateIndex(index_name, build_parameters).value();

        spdlog::debug("dataset_path: " + dataset_path);
        auto eval_dataset = EvalDataset::Load(dataset_path);

        // build
        int64_t total_base = eval_dataset->GetNumberOfBase();
        auto ids = range(total_base);
        auto base = Dataset::Make();
        base->NumElements(total_base)->Dim(eval_dataset->GetDim())->Ids(ids.get())->Owner(false);
        if (eval_dataset->GetTrainDataType() == vsag::DATATYPE_FLOAT32) {
            base->Float32Vectors((const float*)eval_dataset->GetTrain());
        } else if (eval_dataset->GetTrainDataType() == vsag::DATATYPE_INT8) {
            base->Int8Vectors((const int8_t*)eval_dataset->GetTrain());
        }
        auto build_start = std::chrono::steady_clock::now();
        if (auto buildindex = index->Build(base); not buildindex.has_value()) {
            std::cerr << "build error: " << buildindex.error().message << std::endl;
            exit(-1);
        }
        auto build_finish = std::chrono::steady_clock::now();

        // serialize
        auto serialize_result = index->Serialize();
        if (not serialize_result.has_value()) {
            throw std::runtime_error("serialize error: " + serialize_result.error().message);
        }
        vsag::BinarySet& binary_set = serialize_result.value();
        std::filesystem::path dir(DIR_NAME);
        std::map<std::string, size_t> file_sizes;
        for (const auto& key : binary_set.GetKeys()) {
            std::filesystem::path file_path(key);
            std::filesystem::path full_path = dir / file_path;
            vsag::Binary binary = binary_set.Get(key);
            std::ofstream file(full_path.string(), std::ios::binary);
            file.write(reinterpret_cast<char*>(binary.data.get()), binary.size);
            file_sizes[key] = binary.size;
            file.close();
        }

        std::ofstream outfile(dir / META_DATA_FILE);
        for (const auto& pair : file_sizes) {
            outfile << pair.first << " " << pair.second << std::endl;
        }
        outfile.close();
        index = nullptr;

        json output;
        output["index_name"] = index_name;
        output["build_parameters"] = build_parameters;
        output["dataset"] = dataset_path;
        output["num_base"] = total_base;
        double build_time_in_second =
            std::chrono::duration<double>(build_finish - build_start).count();
        output["build_time_in_second"] = build_time_in_second;
        output["tps"] = total_base / build_time_in_second;
        return output;
    }

    static json
    Search(const std::string& dataset_path,
           const std::string& index_name,
           const int64_t top_k,
           const std::string& build_parameters,
           const std::string& search_parameters) {
        // deserialize
        std::filesystem::path dir(DIR_NAME);
        std::map<std::string, size_t> file_sizes;
        std::ifstream infile(dir / META_DATA_FILE);
        std::string filename;
        size_t size;
        while (infile >> filename >> size) {
            file_sizes[filename] = size;
        }
        infile.close();

        auto index = Factory::CreateIndex(index_name, build_parameters).value();
        vsag::ReaderSet reader_set;
        for (const auto& single_file : file_sizes) {
            const std::string& key = single_file.first;
            size = single_file.second;
            std::filesystem::path file_path(key);
            std::filesystem::path full_path = dir / file_path;
            auto reader = vsag::Factory::CreateLocalFileReader(full_path.string(), 0, size);
            reader_set.Set(key, reader);
        }

        index->Deserialize(reader_set);
        unsigned long long memoryUsage = 0;
        std::ifstream statFileAfter("/proc/self/status");
        if (statFileAfter.is_open()) {
            std::string line;
            while (std::getline(statFileAfter, line)) {
                if (line.substr(0, 6) == "VmRSS:") {
                    std::string value = line.substr(6);
                    memoryUsage = std::stoull(value) * 1024;
                    break;
                }
            }
            statFileAfter.close();
        }

        auto eval_dataset = EvalDataset::Load(dataset_path);

        // search
        auto search_start = std::chrono::steady_clock::now();
        double correct = 0;
        int64_t total = eval_dataset->GetNumberOfQuery();
        spdlog::debug("total: " + std::to_string(total));
        std::vector<DatasetPtr> results;
        for (int64_t i = 0; i < total; ++i) {
            auto query = Dataset::Make();
            query->NumElements(1)->Dim(eval_dataset->GetDim())->Owner(false);

            if (eval_dataset->GetTestDataType() == vsag::DATATYPE_FLOAT32) {
                query->Float32Vectors((const float*)eval_dataset->GetOneTest(i));
            } else if (eval_dataset->GetTestDataType() == vsag::DATATYPE_INT8) {
                query->Int8Vectors((const int8_t*)eval_dataset->GetOneTest(i));
            }
            auto filter = [&eval_dataset, i](int64_t base_id) {
                return not eval_dataset->IsMatch(i, base_id);
            };
            auto result = index->KnnSearch(query, top_k, search_parameters, filter);
            if (not result.has_value()) {
                std::cerr << "query error: " << result.error().message << std::endl;
                exit(-1);
            }
            results.emplace_back(result.value());
        }
        auto search_finish = std::chrono::steady_clock::now();

        size_t dim = eval_dataset->GetDim();
        // calculate recall
        for (int64_t i = 0; i < total; ++i) {
            // k@k
            int64_t* ground_truth = eval_dataset->GetNeighbors(i);
            const int64_t* neighbors = results[i]->GetIds();
            auto distances_neighbors = std::shared_ptr<float[]>(new float[top_k]);
            auto distances_gt = std::shared_ptr<float[]>(new float[top_k]);
            auto dist_func = eval_dataset->GetDistanceFunc();
            for (int j = 0; j < top_k; ++j) {
                distances_neighbors[j] = dist_func(
                    eval_dataset->GetOneTrain(neighbors[j]), eval_dataset->GetOneTest(i), &dim);
                distances_gt[j] = dist_func(
                    eval_dataset->GetOneTrain(ground_truth[j]), eval_dataset->GetOneTest(i), &dim);
            }
            auto hit_result =
                get_recall(distances_neighbors.get(), distances_gt.get(), top_k, top_k);
            correct += hit_result;
        }
        spdlog::debug("correct: " + std::to_string(correct));
        float recall = correct / total;

        json output;
        // input
        output["index_name"] = index_name;
        output["search_parameters"] = search_parameters;
        output["dataset"] = dataset_path;
        // for debugging
        double search_time_in_second =
            std::chrono::duration<double>(search_finish - search_start).count();
        output["search_time_in_second"] = search_time_in_second;
        output["correct"] = correct;
        output["num_query"] = total;
        output["top_k"] = top_k;
        // key results
        output["recall"] = recall;
        output["qps"] = total / search_time_in_second;
        output["estimate_used_memory"] = index->GetMemoryUsage();
        output["memory"] = memoryUsage;
        return output;
    }

private:
    static std::shared_ptr<int64_t[]>
    range(int64_t length) {
        auto result = std::shared_ptr<int64_t[]>(new int64_t[length]);
        for (int64_t i = 0; i < length; ++i) {
            result[i] = i;
        }
        return result;
    }
};

bool
valid_and_extrcat_top_k(const std::string& input, int64_t& number) {
    std::regex pattern(R"(^search:(\d+)$)");
    std::smatch match;
    if (std::regex_match(input, match, pattern)) {
        number = std::stoi(match[1].str());
        if (number <= 0) {
            std::cerr << "top k must be set to a value more than 0" << std::endl;
            exit(-1);
        }
        return true;
    }
    if (input == "search") {
        number = 1;
        return true;
    }
    return false;
}

nlohmann::json
run_test(const std::string& dataset_path,
         const std::string& process,
         const std::string& index_name,
         const std::string& build_parameters,
         const std::string& search_parameters) {
    int64_t top_k;
    if (process == "build") {
        return PerfTools::Build(dataset_path, index_name, build_parameters);
    } else if (valid_and_extrcat_top_k(process, top_k)) {
        return PerfTools::Search(
            dataset_path, index_name, top_k, build_parameters, search_parameters);
    } else {
        std::cerr << "process must be search or build." << std::endl;
        exit(-1);
    }
}
