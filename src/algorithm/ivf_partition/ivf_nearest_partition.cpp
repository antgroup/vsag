
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

#include <fmt/format.h>

#include <atomic>
#include <cstdlib>

#include "algorithm/hgraph.h"
#include "algorithm/inner_index_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/cluster/kmeans_cluster.h"
#include "inner_string_params.h"
#include "utils/util_functions.h"

namespace vsag {

static constexpr const char* SEARCH_PARAM_TEMPLATE_STR = R"(
{{
    "hgraph": {{
        "ef_search": {}
    }}
}}
)";

IVFNearestPartition::IVFNearestPartition(BucketIdType bucket_count,
                                         const IndexCommonParam& common_param,
                                         IVFPartitionStrategyParametersPtr param)
    : IVFPartitionStrategy(common_param, bucket_count),
      ivf_partition_strategy_param_(std::move(param)) {
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

    if (ivf_partition_strategy_param_->partition_train_type ==
        IVFNearestPartitionTrainerType::KMeansTrainer) {
        constexpr int32_t kmeans_iter_count = 25;
        KMeansCluster cls(static_cast<int32_t>(dim), this->allocator_);
        cls.Run(this->bucket_count_,
                dataset->GetFloat32Vectors(),
                dataset->GetNumElements(),
                kmeans_iter_count);
        memcpy(data.data(), cls.k_centroids_, dim * this->bucket_count_ * sizeof(float));
    } else if (ivf_partition_strategy_param_->partition_train_type ==
               IVFNearestPartitionTrainerType::RandomTrainer) {
        auto selected = select_k_numbers(dataset->GetNumElements(), this->bucket_count_);
        for (int i = 0; i < bucket_count_; ++i) {
            memcpy(data.data() + i * dim,
                   dataset->GetFloat32Vectors() + selected[i] * dim,
                   dim * sizeof(float));
        }
    }
    if (metric_type_ == MetricType::METRIC_TYPE_COSINE) {
        for (int i = 0; i < bucket_count_; ++i) {
            Normalize(data.data() + i * dim_, data.data() + i * dim_, dim_);
        }
    }

    auto build_result = this->route_index_ptr_->Build(centroids);
    this->is_trained_ = true;
}

Vector<BucketIdType>
IVFNearestPartition::ClassifyDatas(const void* datas,
                                   int64_t count,
                                   BucketIdType buckets_per_data,
                                   Statistics& stats) const {
    std::mutex dist_cmp_reduce_mutex;
    uint32_t dist_cmp = 0;
    Vector<BucketIdType> result(buckets_per_data * count, -1, this->allocator_);
    auto task = [&](int64_t i) {
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

        for (int64_t j = 0; j < search_result->GetDim(); ++j) {
            result[i * buckets_per_data + j] = static_cast<BucketIdType>(result_ids[j]);
        }

        std::scoped_lock lock(dist_cmp_reduce_mutex);
        // the return value of GetStatistics always has the same length as the input keys, and
        // atoi("") returns a `0`.
        dist_cmp += std::atoi(search_result->GetStatistics({"dist_cmp"})[0].c_str());
    };
    if (thread_pool_ == nullptr) {
        for (int64_t i = 0; i < count; ++i) {
            task(i);
        }
    } else {
        Vector<std::future<void>> futures(allocator_);
        for (int64_t i = 0; i < count; ++i) {
            futures.push_back(thread_pool_->GeneralEnqueue(task, i));
        }
        for (auto& item : futures) {
            item.get();
        }
    }

    stats.dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);

    return std::move(result);
}

void
IVFNearestPartition::Serialize(StreamWriter& writer) {
    IVFPartitionStrategy::Serialize(writer);
    this->route_index_ptr_->Serialize(writer);
}
void
IVFNearestPartition::Deserialize(lvalue_or_rvalue<StreamReader> reader) {
    IVFPartitionStrategy::Deserialize(reader);
    this->route_index_ptr_->Deserialize(reader);
}
void
IVFNearestPartition::factory_router_index(const IndexCommonParam& common_param) {
    ParamPtr param_ptr;
    JsonType hgraph_json;
    hgraph_json["base_quantization_type"].SetString("fp32");
    hgraph_json["max_degree"].SetInt(64);
    hgraph_json["ef_construction"].SetInt(300);

    param_ptr = HGraph::CheckAndMappingExternalParam(hgraph_json, common_param);
    this->route_index_ptr_ = std::make_shared<HGraph>(param_ptr, common_param);
}
void
IVFNearestPartition::GetCentroid(BucketIdType bucket_id, Vector<float>& centroid) {
    if (!is_trained_ || bucket_id >= bucket_count_) {
        throw std::runtime_error("Invalid bucket_id or partition not trained");
    }
    this->route_index_ptr_->GetCodeByInnerId(bucket_id, (uint8_t*)centroid.data());
}
}  // namespace vsag
