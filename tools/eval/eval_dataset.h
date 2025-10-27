
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
#include "common.h"
#include "nlohmann/json.hpp"
#include "simd/basic_func.h"
#include "vsag/constants.h"
#include "vsag/dataset.h"

namespace vsag::eval {

class EvalDataset;
using EvalDatasetPtr = std::shared_ptr<EvalDataset>;
class EvalDataset {
public:
    static EvalDatasetPtr
    Load(const std::string& filename);

    static void
    Save(const EvalDatasetPtr& dataset, const std::string& filename);

public:
    [[nodiscard]] const void*
    GetTrain() const {
        if (vector_type_ == DENSE_VECTORS) {
            return train_.get();
        } else {
            return sparse_train_.data();
        }
    }

    [[nodiscard]] const void*
    GetTest() const {
        if (vector_type_ == DENSE_VECTORS) {
            return test_.get();
        } else {
            return sparse_test_.data();
        }
    }

    [[nodiscard]] const std::shared_ptr<int64_t[]>
    GetTrainLabels() const {
        return train_labels_;
    }

    [[nodiscard]] const std::shared_ptr<int64_t[]>
    GetTestLabels() const {
        return test_labels_;
    }

    [[nodiscard]] float
    GetValidRatio(int64_t label) const {
        return valid_ratio_[label];
    }

    [[nodiscard]] const void*
    GetOneTrain(int64_t id) const {
        if (vector_type_ == DENSE_VECTORS) {
            return train_.get() + id * dim_ * train_data_size_;
        } else {
            return sparse_train_.data() + id;
        }
    }

    [[nodiscard]] const void*
    GetOneTest(int64_t id) const {
        if (vector_type_ == DENSE_VECTORS) {
            return test_.get() + id * dim_ * test_data_size_;
        } else {
            return sparse_test_.data() + id;
        }
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

    std::string
    GetVectorType() const {
        return vector_type_;
    }

    std::string
    GetFilePath() {
        return this->file_path_;
    }
    vsag::DistanceFuncType
    GetDistanceFunc() {
        return this->distance_func_;
    }

    using JsonType = nlohmann::json;
    JsonType
    GetInfo() {
        JsonType result;
        JsonType temp;
        temp["filepath"] = this->GetFilePath();
        temp["dim"] = this->GetDim();
        temp["base_count"] = this->GetNumberOfBase();
        temp["query_count"] = this->GetNumberOfQuery();
        temp["data_type"] = this->GetTrainDataType();
        result["dataset_info"] = temp;
        return result;
    };

    ~EvalDataset() {
        for (auto& i : sparse_train_) {
            delete[] i.ids_;
            delete[] i.vals_;
        }
        for (auto& i : sparse_test_) {
            delete[] i.ids_;
            delete[] i.vals_;
        }
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
    vsag::DistanceFuncType distance_func_;

private:
    std::shared_ptr<char[]> train_;
    std::shared_ptr<char[]> test_;
    std::shared_ptr<int64_t[]> neighbors_;
    std::shared_ptr<float[]> distances_;
    std::shared_ptr<int64_t[]> train_labels_{nullptr};
    std::shared_ptr<int64_t[]> test_labels_{nullptr};
    std::shared_ptr<float[]> valid_ratio_;
    shape_t train_shape_;
    shape_t test_shape_;
    shape_t neighbors_shape_;
    int64_t number_of_base_{};
    int64_t number_of_query_{};
    int64_t number_of_label_{};
    int64_t dim_{};
    size_t train_data_size_{};
    size_t test_data_size_{};
    std::string train_data_type_;
    std::string test_data_type_;
    std::string file_path_;
    std::string metric_;

    std::vector<SparseVector> sparse_train_;
    std::vector<SparseVector> sparse_test_;

    std::string vector_type_ = DENSE_VECTORS;
};
}  // namespace vsag::eval
