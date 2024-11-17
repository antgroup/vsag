
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

#include "H5Cpp.h"
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"
#include "vsag/vsag.h"

using namespace nlohmann;
using namespace spdlog;
using namespace vsag;

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
getIntersection(const int64_t* neighbors,
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

class TestDataset;
using TestDatasetPtr = std::shared_ptr<TestDataset>;
class TestDataset {
public:
    static TestDatasetPtr
    Load(const std::string& filename) {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        // check datasets exist
        bool has_labels = false;
        {
            auto datasets = get_datasets(file);
            assert(datasets.count("train"));
            assert(datasets.count("test"));
            assert(datasets.count("neighbors"));
            has_labels = datasets.count("train_labels") && datasets.count("test_labels");
        }

        // get and (should check shape)
        auto train_shape = get_shape(file, "train");
        spdlog::debug("train.shape: " + to_string(train_shape));
        auto test_shape = get_shape(file, "test");
        spdlog::debug("test.shape: " + to_string(test_shape));
        auto neighbors_shape = get_shape(file, "neighbors");
        spdlog::debug("neighbors.shape: " + to_string(neighbors_shape));
        assert(train_shape.second == test_shape.second);

        auto obj = std::make_shared<TestDataset>();
        obj->train_shape_ = train_shape;
        obj->test_shape_ = test_shape;
        obj->neighbors_shape_ = neighbors_shape;
        obj->dim_ = train_shape.second;
        obj->number_of_base_ = train_shape.first;
        obj->number_of_query_ = test_shape.first;

        // read from file
        {
            H5::DataSet dataset = file.openDataSet("/train");
            H5::DataSpace dataspace = dataset.getSpace();
            auto data_type = dataset.getDataType();
            H5::PredType type = H5::PredType::ALPHA_I8;
            if (data_type.getClass() == H5T_INTEGER && data_type.getSize() == 1) {
                obj->train_data_type_ = vsag::DATATYPE_INT8;
                type = H5::PredType::ALPHA_I8;
                obj->train_data_size_ = 1;
            } else if (data_type.getClass() == H5T_FLOAT) {
                obj->train_data_type_ = vsag::DATATYPE_FLOAT32;
                type = H5::PredType::NATIVE_FLOAT;
                obj->train_data_size_ = 4;
            } else {
                throw std::runtime_error(
                    fmt::format("wrong data type, data type ({}), data size ({})",
                                (int)data_type.getClass(),
                                data_type.getSize()));
            }
            obj->train_ = std::shared_ptr<char[]>(
                new char[train_shape.first * train_shape.second * obj->train_data_size_]);
            dataset.read(obj->train_.get(), type, dataspace);
        }

        {
            H5::DataSet dataset = file.openDataSet("/test");
            H5::DataSpace dataspace = dataset.getSpace();
            auto data_type = dataset.getDataType();
            H5::PredType type = H5::PredType::ALPHA_I8;
            if (data_type.getClass() == H5T_INTEGER && data_type.getSize() == 1) {
                obj->test_data_type_ = vsag::DATATYPE_INT8;
                type = H5::PredType::ALPHA_I8;
                obj->test_data_size_ = 1;
            } else if (data_type.getClass() == H5T_FLOAT) {
                obj->test_data_type_ = vsag::DATATYPE_FLOAT32;
                type = H5::PredType::NATIVE_FLOAT;
                obj->test_data_size_ = 4;
            } else {
                throw std::runtime_error("wrong data type");
            }
            obj->test_ = std::shared_ptr<char[]>(
                new char[test_shape.first * test_shape.second * obj->test_data_size_]);
            dataset.read(obj->test_.get(), type, dataspace);
        }
        {
            obj->neighbors_ = std::shared_ptr<int64_t[]>(
                new int64_t[neighbors_shape.first * neighbors_shape.second]);
            H5::DataSet dataset = file.openDataSet("/neighbors");
            H5::DataSpace dataspace = dataset.getSpace();
            H5::FloatType datatype(H5::PredType::NATIVE_INT64);
            dataset.read(obj->neighbors_.get(), datatype, dataspace);
        }
        if (has_labels) {
            H5::FloatType datatype(H5::PredType::NATIVE_INT64);

            H5::DataSet train_labels_dataset = file.openDataSet("/train_labels");
            H5::DataSpace train_labels_dataspace = train_labels_dataset.getSpace();
            obj->train_labels_ = std::shared_ptr<int64_t[]>(new int64_t[obj->number_of_base_]);
            train_labels_dataset.read(obj->train_labels_.get(), datatype, train_labels_dataspace);

            H5::DataSet test_labels_dataset = file.openDataSet("/test_labels");
            H5::DataSpace test_labels_dataspace = test_labels_dataset.getSpace();
            obj->test_labels_ = std::shared_ptr<int64_t[]>(new int64_t[obj->number_of_query_]);
            test_labels_dataset.read(obj->test_labels_.get(), datatype, test_labels_dataspace);
        }

        return obj;
    }

public:
    const void*
    GetTrain() const {
        return train_.get();
    }

    const void*
    GetTest() const {
        return test_.get();
    }

    const void*
    GetOneTest(int64_t id) const {
        return test_.get() + id * dim_ * test_data_size_;
    }

    int64_t
    GetNearestNeighbor(int64_t i) const {
        return neighbors_[i * neighbors_shape_.second];
    }

    int64_t*
    GetNeighbors(int64_t i) const {
        return neighbors_.get() + i * neighbors_shape_.second;
    }

    int64_t
    GetNumberOfBase() const {
        return number_of_base_;
    }

    int64_t
    GetNumberOfQuery() const {
        return number_of_query_;
    }

    int64_t
    GetDim() const {
        return dim_;
    }

    std::string
    GetTrainDataType() const {
        return train_data_type_;
    }
    std::string
    GetTestDataType() const {
        return test_data_type_;
    }

    bool
    IsMatch(int64_t query_id, int64_t base_id) {
        if (this->test_labels_ == nullptr || this->train_labels_ == nullptr) {
            return true;
        }
        return test_labels_[query_id] == train_labels_[base_id];
    }

private:
    using shape_t = std::pair<int64_t, int64_t>;
    static std::unordered_set<std::string>
    get_datasets(const H5::H5File& file) {
        std::unordered_set<std::string> datasets;
        H5::Group root = file.openGroup("/");
        hsize_t numObj = root.getNumObjs();
        for (unsigned i = 0; i < numObj; ++i) {
            std::string objname = root.getObjnameByIdx(i);
            H5O_info_t objinfo;
            root.getObjinfo(objname, objinfo);
            if (objinfo.type == H5O_type_t::H5O_TYPE_DATASET) {
                datasets.insert(objname);
            }
        }
        return datasets;
    }

    static shape_t
    get_shape(const H5::H5File& file, const std::string& dataset_name) {
        H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims_out[2];
        int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
        return std::make_pair<int64_t, int64_t>(dims_out[0], dims_out[1]);
    }

    static std::string
    to_string(const shape_t& shape) {
        return "[" + std::to_string(shape.first) + "," + std::to_string(shape.second) + "]";
    }

private:
    std::shared_ptr<char[]> train_;
    std::shared_ptr<char[]> test_;
    std::shared_ptr<int64_t[]> neighbors_;
    std::shared_ptr<int64_t[]> train_labels_;
    std::shared_ptr<int64_t[]> test_labels_;
    shape_t train_shape_;
    shape_t test_shape_;
    shape_t neighbors_shape_;
    int64_t number_of_base_;
    int64_t number_of_query_;
    int64_t dim_;
    size_t train_data_size_;
    size_t test_data_size_;
    std::string train_data_type_;
    std::string test_data_type_;
};

class Test {
public:
    static json
    Build(const std::string& dataset_path,
          const std::string& index_name,
          const std::string& build_parameters) {
        spdlog::debug("index_name: " + index_name);
        spdlog::debug("build_parameters: " + build_parameters);
        auto index = Factory::CreateIndex(index_name, build_parameters).value();

        spdlog::debug("dataset_path: " + dataset_path);
        auto test_dataset = TestDataset::Load(dataset_path);

        // build
        int64_t total_base = test_dataset->GetNumberOfBase();
        auto ids = range(total_base);
        auto base = Dataset::Make();
        base->NumElements(total_base)->Dim(test_dataset->GetDim())->Ids(ids.get())->Owner(false);
        if (test_dataset->GetTrainDataType() == vsag::DATATYPE_FLOAT32) {
            base->Float32Vectors((const float*)test_dataset->GetTrain());
        } else if (test_dataset->GetTrainDataType() == vsag::DATATYPE_INT8) {
            base->Int8Vectors((const int8_t*)test_dataset->GetTrain());
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
            size_t size = single_file.second;
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

        auto test_dataset = TestDataset::Load(dataset_path);

        // search
        auto search_start = std::chrono::steady_clock::now();
        int64_t correct = 0;
        int64_t total = test_dataset->GetNumberOfQuery();
        spdlog::debug("total: " + std::to_string(total));
        std::vector<DatasetPtr> results;
        for (int64_t i = 0; i < total; ++i) {
            auto query = Dataset::Make();
            query->NumElements(1)->Dim(test_dataset->GetDim())->Owner(false);

            if (test_dataset->GetTestDataType() == vsag::DATATYPE_FLOAT32) {
                query->Float32Vectors((const float*)test_dataset->GetOneTest(i));
            } else if (test_dataset->GetTestDataType() == vsag::DATATYPE_INT8) {
                query->Int8Vectors((const int8_t*)test_dataset->GetOneTest(i));
            }
            auto filter = [&test_dataset, i](int64_t base_id) {
                return not test_dataset->IsMatch(i, base_id);
            };
            auto result = index->KnnSearch(query, top_k, search_parameters, filter);
            if (not result.has_value()) {
                std::cerr << "query error: " << result.error().message << std::endl;
                exit(-1);
            }
            results.emplace_back(result.value());
        }
        auto search_finish = std::chrono::steady_clock::now();

        // calculate recall
        for (int64_t i = 0; i < total; ++i) {
            // k@k
            int64_t* neighbors = test_dataset->GetNeighbors(i);
            const int64_t* ground_truth = results[i]->GetIds();
            auto hit_result = getIntersection(neighbors, ground_truth, top_k, top_k);
            correct += hit_result.size();
        }
        spdlog::debug("correct: " + std::to_string(correct));
        float recall = 1.0 * correct / (total * top_k);

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
        return Test::Build(dataset_path, index_name, build_parameters);
    } else if (valid_and_extrcat_top_k(process, top_k)) {
        return Test::Search(dataset_path, index_name, top_k, build_parameters, search_parameters);
    } else {
        std::cerr << "process must be search or build." << std::endl;
        exit(-1);
    }
}
