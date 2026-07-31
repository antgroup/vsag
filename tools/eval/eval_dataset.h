
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

#include <memory>
#include <unordered_map>
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
    EvalDataset() = default;
    EvalDataset(const EvalDataset&) = delete;
    EvalDataset&
    operator=(const EvalDataset&) = delete;
    EvalDataset(EvalDataset&&) = delete;
    EvalDataset&
    operator=(EvalDataset&&) = delete;

    static EvalDatasetPtr
    Load(const std::string& filename);

    static EvalDatasetPtr
    FromDatasets(const vsag::DatasetPtr& base,
                 const vsag::DatasetPtr& queries,
                 const vsag::DatasetPtr& ground_truth,
                 const std::string& metric_type);

    static EvalDatasetPtr
    FromSearchDatasets(const vsag::DatasetPtr& queries, const vsag::DatasetPtr& ground_truth);

    static void
    Save(const EvalDatasetPtr& dataset, const std::string& filename);

public:
    [[nodiscard]] const void*
    GetTrain() const {
        if (base_dataset_ != nullptr) {
            return base_dataset_->GetFloat32Vectors();
        }
        if (vector_type_ == DENSE_VECTORS) {
            return train_.get();
        }
        if (vector_type_ == SPARSE_VECTORS) {
            return sparse_train_.data();
        }
        return nullptr;
    }

    [[nodiscard]] const void*
    GetTest() const {
        if (query_dataset_ != nullptr) {
            return query_dataset_->GetFloat32Vectors();
        }
        if (vector_type_ == DENSE_VECTORS) {
            return test_.get();
        }
        if (vector_type_ == SPARSE_VECTORS) {
            return sparse_test_.data();
        }
        return nullptr;
    }

    [[nodiscard]] const vsag::MultiVector*
    GetMultiTrainVectors() const {
        return multi_train_vectors_.data();
    }

    [[nodiscard]] const vsag::MultiVector*
    GetMultiTestVectors() const {
        return multi_test_vectors_.data();
    }

    [[nodiscard]] const uint32_t*
    GetTrainVectorCounts() const {
        return train_vector_counts_.get();
    }

    [[nodiscard]] const uint32_t*
    GetTestVectorCounts() const {
        return test_vector_counts_.get();
    }

    [[nodiscard]] int64_t
    GetMultiVectorDim() const {
        return multi_vector_dim_;
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
            return static_cast<const char*>(GetTrain()) + id * dim_ * train_data_size_;
        }
        if (vector_type_ == SPARSE_VECTORS) {
            return sparse_train_.data() + id;
        }
        return nullptr;
    }

    [[nodiscard]] const void*
    GetOneTest(int64_t id) const {
        if (vector_type_ == DENSE_VECTORS) {
            return static_cast<const char*>(GetTest()) + id * dim_ * test_data_size_;
        }
        if (vector_type_ == SPARSE_VECTORS) {
            return sparse_test_.data() + id;
        }
        return nullptr;
    }

    [[nodiscard]] int64_t
    GetNearestNeighbor(int64_t i) const {
        const auto* neighbors = GetNeighbors(i);
        return neighbors == nullptr ? -1 : neighbors[0];
    }

    [[nodiscard]] const int64_t*
    GetNeighbors(int64_t i) const {
        if (ground_truth_dataset_ != nullptr) {
            return ground_truth_dataset_->GetIds() + i * neighbors_shape_.second;
        }
        return neighbors_ == nullptr ? nullptr : neighbors_.get() + i * neighbors_shape_.second;
    }

    [[nodiscard]] const float*
    GetDistances(int64_t i) const {
        if (ground_truth_dataset_ != nullptr) {
            const auto* distances = ground_truth_dataset_->GetDistances();
            return distances == nullptr ? nullptr : distances + i * neighbors_shape_.second;
        }
        return distances_ == nullptr ? nullptr : distances_.get() + i * neighbors_shape_.second;
    }

    [[nodiscard]] const int64_t*
    GetTrainIds() const {
        if (base_dataset_ != nullptr && base_dataset_->GetIds() != nullptr) {
            return base_dataset_->GetIds();
        }
        return identity_train_ids_.get();
    }

    [[nodiscard]] const std::string*
    GetTrainPaths() const {
        return base_dataset_ == nullptr ? nullptr : base_dataset_->GetPaths();
    }

    [[nodiscard]] const std::string*
    GetTestPaths() const {
        return query_dataset_ == nullptr ? nullptr : query_dataset_->GetPaths();
    }

    [[nodiscard]] const void*
    GetOneTrainById(int64_t id) const {
        if (train_ids_are_identity_) {
            if (id < 0 || id >= number_of_base_) {
                return nullptr;
            }
            return GetOneTrain(id);
        }
        const auto found = train_id_to_row_.find(id);
        return found == train_id_to_row_.end() ? nullptr : GetOneTrain(found->second);
    }

    [[nodiscard]] int64_t
    GetNumberOfBase() const {
        return number_of_base_;
    }

    [[nodiscard]] int64_t
    GetNumberOfQuery() const {
        return number_of_query_;
    }

    [[nodiscard]] uint64_t
    GetGroundTruthK() const {
        return neighbors_shape_.second > 0 ? static_cast<uint64_t>(neighbors_shape_.second) : 0;
    }

    [[nodiscard]] const std::string&
    GetMetric() const {
        return metric_;
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

    // Per-record byte offsets into the on-disk sparse byte streams. For a
    // sparse split with N records, GetSparseTrainOffsets() returns a vector
    // of size N+1 where offsets[i] is the start of record i and offsets[N]
    // equals the total byte stream length, enabling O(1) record-i lookup.
    [[nodiscard]] const std::vector<uint64_t>&
    GetSparseTrainOffsets() const {
        return sparse_train_offsets_;
    }
    [[nodiscard]] const std::vector<uint64_t>&
    GetSparseTestOffsets() const {
        return sparse_test_offsets_;
    }
    [[nodiscard]] const std::vector<uint64_t>&
    GetTrainTokenSequenceOffsets() const {
        return train_token_seq_offsets_;
    }
    [[nodiscard]] const std::vector<uint64_t>&
    GetTestTokenSequenceOffsets() const {
        return test_token_seq_offsets_;
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
            delete[] i.token_sequence_;
        }
        for (auto& i : sparse_test_) {
            delete[] i.ids_;
            delete[] i.vals_;
            delete[] i.token_sequence_;
        }
        for (auto& mv : multi_train_vectors_) {
            delete[] mv.vectors_;
        }
        for (auto& mv : multi_test_vectors_) {
            delete[] mv.vectors_;
        }
    }

private:
    void
    InitializeTrainIds(const int64_t* ids);

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
        const int ndims = dataspace.getSimpleExtentNdims();
        if (ndims != 1 && ndims != 2) {
            throw std::runtime_error("unsupported dataset rank: " + std::to_string(ndims));
        }
        hsize_t dims_out[2] = {0, 0};
        dataspace.getSimpleExtentDims(dims_out, NULL);
        if (ndims == 1) {
            return std::make_pair<int64_t, int64_t>(dims_out[0], 0);
        }
        return std::make_pair<int64_t, int64_t>(dims_out[0], dims_out[1]);
    }

    static std::string
    to_string(const shape_t& shape) {
        return "[" + std::to_string(shape.first) + "," + std::to_string(shape.second) + "]";
    }

private:
    vsag::DistanceFuncType distance_func_;

protected:
    std::shared_ptr<char[]> train_;
    std::shared_ptr<char[]> test_;
    std::shared_ptr<int64_t[]> neighbors_;
    std::shared_ptr<float[]> distances_;
    std::shared_ptr<int64_t[]> identity_train_ids_;
    bool train_ids_are_identity_{false};
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

    vsag::DatasetPtr base_dataset_;
    vsag::DatasetPtr query_dataset_;
    vsag::DatasetPtr ground_truth_dataset_;
    std::unordered_map<int64_t, int64_t> train_id_to_row_;

    std::vector<SparseVector> sparse_train_;
    std::vector<SparseVector> sparse_test_;

    // Per-record byte offsets within /train and /test sparse byte streams.
    // Always length N+1 (Q+1 for test), with offsets_.back() == byte stream
    // total size. Built on load (from the optional /train_offsets &
    // /test_offsets datasets, or recomputed if absent) and used for O(1)
    // random access to the i-th sparse vector record on disk.
    std::vector<uint64_t> sparse_train_offsets_;
    std::vector<uint64_t> sparse_test_offsets_;

    // Per-record byte offsets within /train_token_sequences &
    // /test_token_sequences. Only populated when the corresponding token
    // sequence dataset is present.
    std::vector<uint64_t> train_token_seq_offsets_;
    std::vector<uint64_t> test_token_seq_offsets_;

    std::vector<vsag::MultiVector> multi_train_vectors_;
    std::vector<vsag::MultiVector> multi_test_vectors_;
    std::shared_ptr<uint32_t[]> train_vector_counts_;
    std::shared_ptr<uint32_t[]> test_vector_counts_;
    int64_t multi_vector_dim_{0};

    std::string vector_type_ = DENSE_VECTORS;
};
}  // namespace vsag::eval
