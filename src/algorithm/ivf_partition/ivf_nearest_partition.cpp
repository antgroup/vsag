
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

#include "ivf_nearest_partition.h"

#include <fmt/format-inl.h>

#include "algorithm/brute_force.h"
#include "algorithm/brute_force_parameter.h"
#include "impl/kmeans_cluster.h"
#include "inner_string_params.h"
#include "safe_allocator.h"
#include "utils/util_functions.h"

namespace vsag {

static constexpr const char* SEARCH_PARAM_TEMPLATE_STR = R"(
{{
    "hnsw": {{
        "ef_search": {}
    }}
}}
)";

IVFNearestPartition::IVFNearestPartition(BucketIdType bucket_count,
                                         const IndexCommonParam& common_param,
                                         IVFNearestPartitionTrainerType trainer_type)
    : IVFPartitionStrategy(common_param, bucket_count), trainer_type_(trainer_type) {
    this->factory_router_index(common_param);
}

void
IVFNearestPartition::Train(const DatasetPtr dataset) {
    auto dim = this->dim_;
    auto centroids = Dataset::Make();
    Vector<float> data(bucket_count_ * dim, allocator_);
    Vector<LabelType> ids(this->bucket_count_, allocator_);
    std::iota(ids.begin(), ids.end(), 0);
    centroids->Ids(ids.data())
        ->Dim(dim)
        ->Float32Vectors(data.data())
        ->NumElements(this->bucket_count_)
        ->Owner(false);

    if (trainer_type_ == IVFNearestPartitionTrainerType::KMeansTrainer) {
        KMeansCluster cls(static_cast<int32_t>(dim), this->allocator_);
        cls.Run(this->bucket_count_, dataset->GetFloat32Vectors(), dataset->GetNumElements());
        memcpy(data.data(), cls.k_centroids_, dim * this->bucket_count_ * sizeof(float));
    } else if (trainer_type_ == IVFNearestPartitionTrainerType::RandomTrainer) {
        auto selected = select_k_numbers(dataset->GetNumElements(), this->bucket_count_);
        for (int i = 0; i < bucket_count_; ++i) {
            memcpy(data.data() + i * dim,
                   dataset->GetFloat32Vectors() + selected[i] * dim,
                   dim * this->bucket_count_ * sizeof(float));
        }
    }

    auto build_result = this->route_index_ptr_->Build(centroids);
    this->is_trained_ = true;
}

Vector<BucketIdType>
IVFNearestPartition::ClassifyDatas(const void* datas,
                                   int64_t count,
                                   BucketIdType buckets_per_data) {
    Vector<BucketIdType> result(buckets_per_data * count, this->allocator_);
    for (int64_t i = 0; i < count; ++i) {
        auto query = Dataset::Make();
        query->Dim(this->dim_)
            ->Float32Vectors(reinterpret_cast<const float*>(datas) + i * this->dim_)
            ->NumElements(1)
            ->Owner(false);
        auto search_param = fmt::format(
            SEARCH_PARAM_TEMPLATE_STR, std::max(10L, static_cast<int64_t>(buckets_per_data * 1.2)));
        FilterPtr filter = nullptr;
        auto search_result =
            this->route_index_ptr_->KnnSearch(query, buckets_per_data, search_param, filter);
        const auto* result_ids = search_result->GetIds();

        for (int64_t j = 0; j < buckets_per_data; ++j) {
            result[i * buckets_per_data + j] = static_cast<BucketIdType>(result_ids[j]);
        }
    }
    return result;
}
void
IVFNearestPartition::Serialize(StreamWriter& writer) {
    IVFPartitionStrategy::Serialize(writer);
    this->route_index_ptr_->Serialize(writer);
}
void
IVFNearestPartition::Deserialize(StreamReader& reader) {
    IVFPartitionStrategy::Deserialize(reader);
    this->route_index_ptr_->Deserialize(reader);
}
void
IVFNearestPartition::factory_router_index(const IndexCommonParam& common_param) {
    auto param_ptr = std::make_shared<BruteForceParameter>();
    param_ptr->flatten_param = std::make_shared<FlattenDataCellParameter>();
    JsonType memory_json = {
        {"type", IO_TYPE_VALUE_BLOCK_MEMORY_IO},
    };
    param_ptr->flatten_param->io_parameter = IOParameter::GetIOParameterByJson(memory_json);
    JsonType quantizer_json = {
        {"type", QUANTIZATION_TYPE_VALUE_FP32},
    };
    param_ptr->flatten_param->quantizer_parameter =
        QuantizerParameter::GetQuantizerParameterByJson(quantizer_json);

    this->route_index_ptr_ = std::make_shared<BruteForce>(param_ptr, common_param);
}
}  // namespace vsag
