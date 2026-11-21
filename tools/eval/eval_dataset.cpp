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

#include "eval_dataset.h"

#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>

namespace vsag::eval {

namespace {

EvalDataset::Loader&
registered_loader() {
    static EvalDataset::Loader loader;
    return loader;
}

std::mutex&
registered_loader_mutex() {
    static std::mutex mutex;
    return mutex;
}

}  // namespace

void
EvalDataset::RegisterLoader(Loader loader) {
    std::lock_guard<std::mutex> lock(registered_loader_mutex());
    registered_loader() = std::move(loader);
}

EvalDatasetPtr
EvalDataset::Load(const std::string& filename) {
    Loader loader;
    {
        std::lock_guard<std::mutex> lock(registered_loader_mutex());
        loader = registered_loader();
    }
    if (not loader) {
        throw std::runtime_error("no eval dataset loader is available in this build: " + filename);
    }
    return loader(filename);
}

void
EvalDataset::InitializeTrainIds(const int64_t* ids) {
    train_ids_are_identity_ = true;
    train_id_to_row_.clear();
    if (ids == nullptr) {
        return;
    }

    for (int64_t i = 0; i < number_of_base_; ++i) {
        if (ids[i] != i) {
            train_ids_are_identity_ = false;
            break;
        }
    }
    if (train_ids_are_identity_) {
        return;
    }

    train_id_to_row_.reserve(static_cast<uint64_t>(number_of_base_));
    for (int64_t i = 0; i < number_of_base_; ++i) {
        if (!train_id_to_row_.emplace(ids[i], i).second) {
            throw std::invalid_argument("base dataset contains duplicate ids");
        }
    }
}

EvalDatasetPtr
EvalDataset::FromDatasets(const vsag::DatasetPtr& base,
                          const vsag::DatasetPtr& queries,
                          const vsag::DatasetPtr& ground_truth,
                          const std::string& metric_type) {
    if (base == nullptr || queries == nullptr) {
        throw std::invalid_argument("base and queries datasets are required");
    }
    if (base->GetNumElements() <= 0 || queries->GetNumElements() <= 0) {
        throw std::invalid_argument("base and queries datasets must not be empty");
    }
    if (base->GetDim() <= 0 || base->GetDim() != queries->GetDim()) {
        throw std::invalid_argument("base and queries datasets must have the same positive dim");
    }
    if (base->GetFloat32Vectors() == nullptr || queries->GetFloat32Vectors() == nullptr) {
        throw std::invalid_argument("in-memory eval supports only float32 dense vectors");
    }
    if (ground_truth != nullptr &&
        (ground_truth->GetNumElements() != queries->GetNumElements() ||
         ground_truth->GetDim() <= 0 || ground_truth->GetIds() == nullptr)) {
        throw std::invalid_argument(
            "ground_truth must contain one non-empty id row for every query");
    }

    auto dataset = std::make_shared<EvalDataset>();
    dataset->base_dataset_ = base;
    dataset->query_dataset_ = queries;
    dataset->ground_truth_dataset_ = ground_truth;
    dataset->vector_type_ = DENSE_VECTORS;
    dataset->train_data_type_ = vsag::DATATYPE_FLOAT32;
    dataset->test_data_type_ = vsag::DATATYPE_FLOAT32;
    dataset->train_data_size_ = sizeof(float);
    dataset->test_data_size_ = sizeof(float);
    dataset->number_of_base_ = base->GetNumElements();
    dataset->number_of_query_ = queries->GetNumElements();
    dataset->dim_ = base->GetDim();
    dataset->train_shape_ = {dataset->number_of_base_, dataset->dim_};
    dataset->test_shape_ = {dataset->number_of_query_, dataset->dim_};
    dataset->neighbors_shape_ =
        ground_truth == nullptr ? shape_t{0, 0}
                                : shape_t{ground_truth->GetNumElements(), ground_truth->GetDim()};
    dataset->file_path_ = "<memory>";

    if (metric_type == "l2") {
        dataset->metric_ = "euclidean";
        dataset->distance_func_ =
            [](const void* left, const void* right, const void* dim) -> float {
            return std::sqrt(vsag::L2Sqr(left, right, dim));
        };
    } else if (metric_type == "ip") {
        dataset->metric_ = "ip";
        dataset->distance_func_ = vsag::InnerProductDistance;
    } else if (metric_type == "cosine") {
        dataset->metric_ = "angular";
        dataset->distance_func_ =
            [](const void* left, const void* right, const void* dim) -> float {
            return 1 - vsag::InnerProduct(left, right, dim) /
                           std::sqrt(vsag::InnerProduct(left, left, dim) *
                                     vsag::InnerProduct(right, right, dim));
        };
    } else {
        throw std::invalid_argument("unsupported in-memory eval metric_type: " + metric_type);
    }

    dataset->InitializeTrainIds(base->GetIds());
    if (ground_truth != nullptr) {
        if (ground_truth->GetNumElements() >
            std::numeric_limits<int64_t>::max() / ground_truth->GetDim()) {
            throw std::invalid_argument("ground_truth contains too many ids");
        }
        const auto ground_truth_count = ground_truth->GetNumElements() * ground_truth->GetDim();
        for (int64_t i = 0; i < ground_truth_count; ++i) {
            if (dataset->GetOneTrainById(ground_truth->GetIds()[i]) == nullptr) {
                throw std::invalid_argument("ground_truth contains an id not present in base");
            }
        }
    }
    return dataset;
}

EvalDatasetPtr
EvalDataset::FromSearchDatasets(const vsag::DatasetPtr& queries,
                                const vsag::DatasetPtr& ground_truth) {
    if (queries == nullptr || queries->GetNumElements() <= 0) {
        throw std::invalid_argument("queries dataset is required and must not be empty");
    }
    if (queries->GetDim() <= 0 || queries->GetFloat32Vectors() == nullptr) {
        throw std::invalid_argument(
            "in-memory search evaluation requires positive-dimensional float32 queries");
    }
    if (ground_truth != nullptr &&
        (ground_truth->GetNumElements() != queries->GetNumElements() ||
         ground_truth->GetDim() <= 0 || ground_truth->GetIds() == nullptr)) {
        throw std::invalid_argument(
            "ground_truth must contain one non-empty id row for every query");
    }

    auto dataset = std::make_shared<EvalDataset>();
    dataset->query_dataset_ = queries;
    dataset->ground_truth_dataset_ = ground_truth;
    dataset->vector_type_ = DENSE_VECTORS;
    dataset->train_data_type_ = vsag::DATATYPE_FLOAT32;
    dataset->test_data_type_ = vsag::DATATYPE_FLOAT32;
    dataset->train_data_size_ = sizeof(float);
    dataset->test_data_size_ = sizeof(float);
    dataset->number_of_query_ = queries->GetNumElements();
    dataset->dim_ = queries->GetDim();
    dataset->train_shape_ = {0, dataset->dim_};
    dataset->test_shape_ = {dataset->number_of_query_, dataset->dim_};
    dataset->neighbors_shape_ =
        ground_truth == nullptr ? shape_t{0, 0}
                                : shape_t{ground_truth->GetNumElements(), ground_truth->GetDim()};
    dataset->file_path_ = "<memory>";
    return dataset;
}

}  // namespace vsag::eval
