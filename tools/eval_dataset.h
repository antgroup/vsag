
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

#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <unordered_set>

#include "H5Cpp.h"
#include "vsag/constants.h"

namespace vsag {

class EvalDataset;
using EvalDatasetPtr = std::shared_ptr<EvalDataset>;
class EvalDataset {
public:
    static EvalDatasetPtr
    Load(const std::string& filename);

public:
    [[nodiscard]] const void*
    GetTrain() const {
        return train_.get();
    }

    [[nodiscard]] const void*
    GetTest() const {
        return test_.get();
    }

    [[nodiscard]] const void*
    GetOneTest(int64_t id) const {
        return test_.get() + id * dim_ * test_data_size_;
    }

    [[nodiscard]] int64_t
    GetNearestNeighbor(int64_t i) const {
        return neighbors_[i * neighbors_shape_.second];
    }

    [[nodiscard]] int64_t*
    GetNeighbors(int64_t i) const {
        return neighbors_.get() + i * neighbors_shape_.second;
    }

    [[nodiscard]] float*
    GetDistances(int64_t i) const {
        return distances_.get() + i * neighbors_shape_.second;
    }

    [[nodiscard]] int64_t
    GetNumberOfBase() const {
        return number_of_base_;
    }

    [[nodiscard]] int64_t
    GetNumberOfQuery() const {
        return number_of_query_;
    }

    [[nodiscard]] int64_t
    GetDim() const {
        return dim_;
    }

    [[nodiscard]] std::string
    GetTrainDataType() const {
        return train_data_type_;
    }
    [[nodiscard]] std::string
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
    std::shared_ptr<float[]> distances_;
    std::shared_ptr<int64_t[]> train_labels_;
    std::shared_ptr<int64_t[]> test_labels_;
    shape_t train_shape_;
    shape_t test_shape_;
    shape_t neighbors_shape_;
    int64_t number_of_base_{};
    int64_t number_of_query_{};
    int64_t dim_{};
    size_t train_data_size_{};
    size_t test_data_size_{};
    std::string train_data_type_;
    std::string test_data_type_;
};
}  // namespace vsag
