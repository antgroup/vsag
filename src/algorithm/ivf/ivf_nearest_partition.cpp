
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
#include <limits>

#include "algorithm/hgraph/hgraph.h"
#include "algorithm/inner_index_interface.h"
#include "container_types.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/blas/blas_function.h"
#include "impl/cluster/kmeans_cluster.h"
#include "inner_string_params.h"
#include "query_context.h"
#include "simd/fp32_simd.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

namespace {

constexpr uint64_t K_CENTROID_SCAN_LAYOUT_MARKER = 0x565341475343414EULL;  // "VSAGSCAN"

constexpr BucketIdType INVALID_BUCKET_ID = static_cast<BucketIdType>(-1);

// Max-heap used to keep the k smallest routing keys: the largest retained key is at top() and
// is discarded when a better candidate arrives. bucket_id provides a deterministic tie-break.
using ScanRouteHeap = std::priority_queue<std::pair<float, InnerIdType>,
                                          Vector<std::pair<float, InnerIdType>>,
                                          std::less<>>;

// C = A * B^T (same shape as GNOIMIPartition's batched centroid routing).
void
compute_centroid_dots(const float* A, const float* B, float* C, int64_t M, int64_t N, int64_t K) {
    BlasFunction::Sgemm(BlasFunction::ColMajor,
                        BlasFunction::Trans,
                        BlasFunction::NoTrans,
                        static_cast<int32_t>(N),
                        static_cast<int32_t>(M),
                        static_cast<int32_t>(K),
                        1.0F,
                        B,
                        static_cast<int32_t>(K),
                        A,
                        static_cast<int32_t>(K),
                        0.0F,
                        C,
                        static_cast<int32_t>(N));
}

}  // namespace

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
      ivf_partition_strategy_param_(std::move(param)),
      use_route_graph_(this->ivf_partition_strategy_param_->use_route_graph),
      common_param_(common_param) {
    if (this->use_route_graph_) {
        this->factory_router_index(common_param);
    }
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
        KMeansCluster cls(static_cast<int32_t>(dim), this->allocator_, this->thread_pool_);
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

    if (this->use_route_graph_) {
        const auto failed_ids = this->route_index_ptr_->Build(centroids);
        if (not failed_ids.empty()) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "failed to build all IVF routing centroids");
        }
    } else {
        // Persist the trained centroids (normalized for cosine) so the scan routing path can
        // assign buckets without creating a routing HGraph.
        this->centroids_.resize(this->bucket_count_ * this->dim_);
        memcpy(
            this->centroids_.data(), data.data(), this->bucket_count_ * this->dim_ * sizeof(float));
        this->norms_.resize(this->bucket_count_);
        if (metric_type_ == MetricType::METRIC_TYPE_L2SQR) {
            for (int i = 0; i < bucket_count_; ++i) {
                this->norms_[i] = FP32ComputeIP(this->centroids_.data() + i * dim_,
                                                this->centroids_.data() + i * dim_,
                                                dim_) /
                                  2.0F;
            }
        } else {
            // IP/COSINE distances are defined as `1 - dot`; with normalized cosine centroids
            // the constant term is 1.0 for every centroid.
            std::fill(this->norms_.begin(), this->norms_.end(), 1.0F);
        }
    }
    this->is_trained_ = true;
}

Vector<BucketIdType>
IVFNearestPartition::ClassifyDatas(const void* datas,
                                   int64_t count,
                                   BucketIdType buckets_per_data,
                                   QueryContext* ctx) const {
    if (not this->use_route_graph_) {
        return this->classify_datas_by_scan(datas, count, buckets_per_data, ctx);
    }
    std::mutex dist_cmp_reduce_mutex;
    uint32_t dist_cmp = 0;
    Vector<BucketIdType> result(buckets_per_data * count, INVALID_BUCKET_ID, this->allocator_);
    auto task = [&](int64_t i) {
        auto query = Dataset::Make();
        query->Dim(this->dim_)
            ->Float32Vectors(reinterpret_cast<const float*>(datas) + i * this->dim_)
            ->NumElements(1)
            ->Owner(false);
        auto search_param =
            fmt::format(SEARCH_PARAM_TEMPLATE_STR,
                        std::max<int64_t>(10, static_cast<int64_t>(buckets_per_data * 1.2)));
        FilterPtr filter = nullptr;
        auto search_result =
            this->route_index_ptr_->KnnSearch(query, buckets_per_data, search_param, filter);
        const auto* result_ids = search_result->GetIds();

        for (int64_t j = 0; j < search_result->GetDim(); ++j) {
            result[i * buckets_per_data + j] = static_cast<BucketIdType>(result_ids[j]);
        }

        if (ctx != nullptr and ctx->stats != nullptr) {
            // GetStatistics re-parses the routing statistics JSON and re-dumps it to strings,
            // which is allocation-heavy; it must not run inside the reduce lock. Only the two
            // integer accumulations below are serialized. The return value always has the same
            // length as the input keys, and atoi("") returns a `0`.
            auto route_stats = search_result->GetStatistics({"dist_cmp", "distance_evaluations"});
            const uint32_t local_dist_cmp = std::atoi(route_stats[0].c_str());
            uint64_t eval_count = 0;
            if (route_stats.size() > 1) {
                eval_count = std::strtoull(route_stats[1].c_str(), nullptr, 10);
            }
            {
                std::scoped_lock lock(dist_cmp_reduce_mutex);
                dist_cmp += local_dist_cmp;
            }
            ctx->stats->AddDistance(SearchStatistics::DistancePhase::ROUTING,
                                    DistanceEvaluationBackend::FP32,
                                    eval_count);
        }
    };
    if (thread_pool_ == nullptr or count == 1) {
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

    if (ctx != nullptr and ctx->stats != nullptr) {
        ctx->stats->dist_cmp.fetch_add(dist_cmp, std::memory_order_relaxed);
    }

    return std::move(result);
}

void
IVFNearestPartition::Serialize(StreamWriter& writer) {
    IVFPartitionStrategy::Serialize(writer);
    if (this->use_route_graph_) {
        // Keep the route-graph layout byte-for-byte compatible with the legacy format.
        this->route_index_ptr_->Serialize(writer);
    } else {
        StreamWriter::WriteObj(writer, K_CENTROID_SCAN_LAYOUT_MARKER);
        StreamWriter::WriteVector(writer, this->centroids_);
        StreamWriter::WriteVector(writer, this->norms_);
    }
}
void
IVFNearestPartition::Deserialize(LvalueOrRvalue<StreamReader> reader) {
    IVFPartitionStrategy::Deserialize(reader);
    // Peek the layout marker without consuming bytes so legacy (marker-less) route-graph
    // indices remain loadable with any reader configuration.
    const auto cursor = reader->GetCursor();
    uint64_t layout = 0;
    StreamReader::ReadObj(reader, layout);
    reader->Seek(cursor);
    if (layout == K_CENTROID_SCAN_LAYOUT_MARKER) {
        StreamReader::ReadObj(reader, layout);
        StreamReader::ReadVector(reader, this->centroids_);
        StreamReader::ReadVector(reader, this->norms_);
        const bool invalid_trained_layout =
            this->is_trained_ and (this->centroids_.size() != this->bucket_count_ * this->dim_ or
                                   this->norms_.size() != this->bucket_count_);
        const bool invalid_untrained_layout =
            not this->is_trained_ and (not this->centroids_.empty() or not this->norms_.empty());
        if (invalid_trained_layout or invalid_untrained_layout) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                "invalid IVF centroid scan routing layout");
        }
        this->route_index_ptr_.reset();
        this->use_route_graph_ = false;
    } else {
        // Route-graph layout (the legacy marker-less format). Lazily create the routing HGraph
        // when the reader was configured with use_route_graph=false.
        if (this->route_index_ptr_ == nullptr) {
            this->factory_router_index(this->common_param_);
        }
        this->route_index_ptr_->Deserialize(reader);
        this->centroids_.clear();
        this->norms_.clear();
        this->use_route_graph_ = true;
    }
}
void
IVFNearestPartition::factory_router_index(const IndexCommonParam& common_param) {
    ParamPtr param_ptr;
    JsonType hgraph_json;
    hgraph_json["base_quantization_type"].SetString("fp32");
    hgraph_json["max_degree"].SetInt(ivf_partition_strategy_param_->route_max_degree);
    hgraph_json["ef_construction"].SetInt(ivf_partition_strategy_param_->route_ef_construction);

    param_ptr = HGraph::CheckAndMappingExternalParam(hgraph_json, common_param);
    this->route_index_ptr_ = std::make_shared<HGraph>(param_ptr, common_param);
}
void
IVFNearestPartition::GetCentroid(BucketIdType bucket_id, Vector<float>& centroid) {
    if (!is_trained_) {
        throw VsagException(ErrorType::WRONG_STATUS, "Partition not trained");
    }
    if (bucket_id >= bucket_count_) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "Invalid bucket_id");
    }
    if (this->use_route_graph_) {
        this->route_index_ptr_->GetCodeByInnerId(bucket_id, (uint8_t*)centroid.data());
    } else {
        memcpy(centroid.data(),
               this->centroids_.data() + bucket_id * this->dim_,
               this->dim_ * sizeof(float));
    }
}

[[nodiscard]] uint64_t
IVFNearestPartition::GetMemoryUsage() const {
    auto memory = static_cast<uint64_t>(sizeof(IVFNearestPartition));
    if (this->use_route_graph_) {
        memory += this->route_index_ptr_->GetMemoryUsage();
    }
    memory += this->centroids_.size() * sizeof(float);
    memory += this->norms_.size() * sizeof(float);
    return memory;
}
Vector<BucketIdType>
IVFNearestPartition::classify_datas_by_scan(const void* datas,
                                            int64_t count,
                                            BucketIdType buckets_per_data,
                                            QueryContext* ctx) const {
    Vector<BucketIdType> result(buckets_per_data * count, INVALID_BUCKET_ID, this->allocator_);
    // Centroids only exist after Train(). An untrained index must not route: leave every slot
    // at INVALID_BUCKET_ID (mirrors the untrained route-graph behavior).
    if (not this->is_trained_ or this->bucket_count_ == 0 or count == 0) {
        return result;
    }

    const auto k = std::min<BucketIdType>(buckets_per_data, bucket_count_);
    if (k == 0) {
        return result;
    }

    Vector<float> norm_vectors(allocator_);
    const auto* query_data = static_cast<const float*>(datas);
    if (metric_type_ == MetricType::METRIC_TYPE_COSINE) {
        norm_vectors.resize(count * dim_);
        for (int64_t i = 0; i < count; ++i) {
            Normalize(query_data + i * dim_, norm_vectors.data() + i * dim_, dim_);
        }
        query_data = norm_vectors.data();
    }

    // Batch dot products query x centroids^T via Sgemm (same shape as GNOIMIPartition).
    Vector<float> dots(count * bucket_count_, allocator_);
    compute_centroid_dots(
        query_data, this->centroids_.data(), dots.data(), count, bucket_count_, dim_);

    auto task = [&](int64_t i) {
        Vector<std::pair<float, InnerIdType>> heap_storage(this->allocator_);
        heap_storage.reserve(k);
        ScanRouteHeap heap(std::less<>{}, std::move(heap_storage));
        const auto* dots_i = dots.data() + i * bucket_count_;
        for (BucketIdType b = 0; b < bucket_count_; ++b) {
            const std::pair<float, InnerIdType> candidate{this->norms_[b] - dots_i[b],
                                                          static_cast<InnerIdType>(b)};
            if (heap.size() < k || candidate < heap.top()) {
                heap.push(candidate);
            }
            if (heap.size() > k) {
                heap.pop();
            }
        }
        for (BucketIdType j = k; j > 0; --j) {
            result[i * buckets_per_data + j - 1] = static_cast<BucketIdType>(heap.top().second);
            heap.pop();
        }
    };
    if (thread_pool_ == nullptr or count == 1) {
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

    if (ctx != nullptr and ctx->stats != nullptr) {
        const auto distance_evaluations =
            static_cast<uint64_t>(count) * static_cast<uint64_t>(bucket_count_);
        const auto dist_cmp_increment = static_cast<uint32_t>(
            std::min<uint64_t>(distance_evaluations, std::numeric_limits<uint32_t>::max()));
        ctx->stats->dist_cmp.fetch_add(dist_cmp_increment, std::memory_order_relaxed);
        ctx->stats->AddDistance(SearchStatistics::DistancePhase::ROUTING,
                                DistanceEvaluationBackend::FP32,
                                distance_evaluations);
    }
    return result;
}

}  // namespace vsag
