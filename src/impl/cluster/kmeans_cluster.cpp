
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

#include "kmeans_cluster.h"

#include <omp.h>

#include <random>

#include "algorithm/inner_index_interface.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/blas/blas_function.h"
#include "nearest_centroid_assign.h"
#include "simd/fp32_simd.h"
#include "utils/util_functions.h"

namespace vsag {
namespace {
constexpr uint64_t QUERY_BS = 65536ULL;
}  // namespace

KMeansCluster::KMeansCluster(int32_t dim, Allocator* allocator, SafeThreadPoolPtr thread_pool)
    : dim_(dim), allocator_(allocator), thread_pool_(std::move(thread_pool)) {
    if (thread_pool_ == nullptr) {
        this->thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
    }
}

KMeansCluster::~KMeansCluster() {
    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
}

Vector<int>
KMeansCluster::Run(uint32_t k,
                   const float* datas,
                   uint64_t count,
                   int iter,
                   double* err,
                   bool use_mse_for_convergence,
                   float threshold,
                   KMeansInitMethod init_method,
                   std::optional<uint32_t> random_seed,
                   bool deterministic_reduction) {
    if (k == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "k must be positive");
    }
    if (count == 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "count must be positive");
    }
    if (datas == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "datas cannot be null");
    }
    if (k > count) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "k cannot be larger than count");
    }

    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
    uint64_t size = static_cast<uint64_t>(k) * static_cast<uint64_t>(dim_) * sizeof(float);
    k_centroids_ = static_cast<float*>(allocator_->Allocate(size));

    std::random_device rd;
    std::mt19937 gen(random_seed.has_value() ? *random_seed : rd());

    if (init_method == KMeansInitMethod::KMEANS_PLUS_PLUS) {
        select_initial_centroids_kmeans_plus_plus(datas, count, k, gen);
    } else {
        select_initial_centroids_random(datas, count, k, gen);
    }

    double total_err = std::numeric_limits<double>::max();
    double last_err = std::numeric_limits<double>::max();
    Vector<int32_t> labels(count, -1, this->allocator_);
    std::vector<std::future<void>> futures;

    logger::trace("KMeansCluster::Run k: {}, count: {}, iter: {}", k, count, iter);
    if (k < THRESHOLD_FOR_HGRAPH) {
        logger::trace("KMeansCluster::Run use blas");
    } else {
        logger::trace("KMeansCluster::Run use hgraph");
    }

    for (int it = 0; it < iter; ++it) {
        if (k < THRESHOLD_FOR_HGRAPH) {
            total_err = NearestCentroidAssign(k_centroids_,
                                              k,
                                              datas,
                                              count,
                                              static_cast<uint64_t>(dim_),
                                              thread_pool_,
                                              allocator_,
                                              labels.data());
        } else {
            total_err = this->find_nearest_one_with_hgraph(datas, count, k, labels);
        }
        constexpr uint64_t bs = 1024;

        Vector<int> counts(k, 0, allocator_);
        Vector<float> new_centroids(static_cast<uint64_t>(k) * dim_, 0.0F, allocator_);
        const uint64_t block_count = (count + bs - 1) / bs;
        const uint64_t centroid_values = static_cast<uint64_t>(k) * dim_;
        Vector<int> block_counts(allocator_);
        Vector<float> block_centroids(allocator_);
        if (deterministic_reduction) {
            block_counts.resize(block_count * k, 0);
            block_centroids.resize(block_count * centroid_values, 0.0F);
        }
        std::mutex merge_mutex;

        auto update_centroids_func = [&](uint64_t block_id, uint64_t start, uint64_t end) {
            omp_set_num_threads(1);
            Vector<int> local_counts(k, 0, allocator_);
            Vector<float> local_centroids(centroid_values, 0.0F, allocator_);

            for (uint64_t i = start; i < end; ++i) {
                int32_t label = labels[i];
                if (label >= 0 && label < static_cast<int32_t>(k)) {
                    local_counts[label]++;
                    BlasFunction::Saxpy(
                        dim_,
                        1.0F,
                        datas + i * dim_,
                        1,
                        local_centroids.data() + label * static_cast<uint64_t>(dim_),
                        1);
                }
            }

            if (deterministic_reduction) {
                std::copy(
                    local_counts.begin(), local_counts.end(), block_counts.data() + block_id * k);
                std::copy(local_centroids.begin(),
                          local_centroids.end(),
                          block_centroids.data() + block_id * centroid_values);
            } else {
                std::lock_guard<std::mutex> lock(merge_mutex);
                for (uint32_t j = 0; j < k; ++j) {
                    if (local_counts[j] > 0) {
                        counts[j] += local_counts[j];
                        BlasFunction::Saxpy(
                            dim_,
                            1.0F,
                            local_centroids.data() + j * static_cast<uint64_t>(dim_),
                            1,
                            new_centroids.data() + j * static_cast<uint64_t>(dim_),
                            1);
                    }
                }
            }
        };
        for (uint64_t i = 0; i < count; i += bs) {
            futures.emplace_back(thread_pool_->GeneralEnqueue(
                update_centroids_func, i / bs, i, std::min(i + bs, count)));
        }
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();
        if (deterministic_reduction) {
            for (uint64_t block_id = 0; block_id < block_count; ++block_id) {
                for (uint32_t j = 0; j < k; ++j) {
                    const auto count_offset = block_id * k + j;
                    if (block_counts[count_offset] > 0) {
                        counts[j] += block_counts[count_offset];
                        BlasFunction::Saxpy(dim_,
                                            1.0F,
                                            block_centroids.data() + block_id * centroid_values +
                                                static_cast<uint64_t>(j) * dim_,
                                            1,
                                            new_centroids.data() + static_cast<uint64_t>(j) * dim_,
                                            1);
                    }
                }
            }
        }

        std::uniform_int_distribution<uint64_t> dis(0, count - 1);
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                BlasFunction::Sscal(dim_,
                                    1.0F / static_cast<float>(counts[j]),
                                    new_centroids.data() + j * static_cast<uint64_t>(dim_),
                                    1);
                std::copy(new_centroids.data() + j * static_cast<uint64_t>(dim_),
                          new_centroids.data() + (j + 1) * static_cast<uint64_t>(dim_),
                          k_centroids_ + j * static_cast<uint64_t>(dim_));
            } else {
                auto index = dis(gen);
                for (int s = 0; s < dim_; ++s) {
                    k_centroids_[j * dim_ + s] = datas[index * dim_ + s];
                }
            }
        }

        logger::trace("[{}] KMeansCluster::Run iter: {}/{} finished, cur loss is {}",
                      get_current_time(),
                      static_cast<int>(it),
                      static_cast<int>(iter),
                      static_cast<double>(total_err));
        if (it > 0 && use_mse_for_convergence &&
            std::fabs(last_err - total_err) / static_cast<double>(count) < threshold) {
            break;
        }

        last_err = total_err;
    }
    if (err != nullptr) {
        *err = total_err;
    }
    return labels;
}

double
KMeansCluster::find_nearest_one_with_hgraph(const float* query,
                                            const uint64_t query_count,
                                            const uint64_t k,
                                            Vector<int32_t>& labels) {
    if (k_centroids_ == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "k_centroids_ is nullptr");
    }
    double error = 0.0;
    std::mutex error_mutex;

    IndexCommonParam param;
    param.dim_ = dim_;
    param.allocator_ = std::make_shared<SafeAllocator>(this->allocator_);
    param.thread_pool_ = this->thread_pool_;
    param.metric_ = MetricType::METRIC_TYPE_L2SQR;
    auto max_degree = std::max(32, dim_ / 8);

    auto hgraph =
        InnerIndexInterface::FastCreateIndex(fmt::format("hgraph|{}|fp32", max_degree), param);
    auto base = Dataset::Make();
    Vector<int64_t> ids(k, allocator_);
    std::iota(ids.begin(), ids.end(), 0);
    base->Dim(dim_)
        ->NumElements(static_cast<int64_t>(k))
        ->Float32Vectors(this->k_centroids_)
        ->Ids(ids.data())
        ->Owner(false);
    hgraph->Build(base);
    hgraph->SetImmutable();
    FilterPtr filter = nullptr;
    constexpr const char* search_param = R"({"hgraph":{"ef_search":10}})";
    auto func = [&](const uint64_t begin, const uint64_t end) -> void {
        double thread_local_error = 0.0;
        for (uint64_t j = begin; j < end; ++j) {
            auto q = Dataset::Make();
            q->Owner(false)
                ->Float32Vectors(query + j * this->dim_)
                ->NumElements(1)
                ->Dim(this->dim_);
            auto ret = hgraph->KnnSearch(q, 1, search_param, filter);
            labels[j] = static_cast<int32_t>(ret->GetIds()[0]);
            thread_local_error += static_cast<double>(ret->GetDistances()[0]);
        }
        {
            std::lock_guard<std::mutex> lock(error_mutex);
            error += thread_local_error;
        }
    };
    std::vector<std::future<void>> futures;
    for (uint64_t i = 0; i < query_count; i += QUERY_BS) {
        futures.emplace_back(
            thread_pool_->GeneralEnqueue(func, i, std::min(i + QUERY_BS, query_count)));
    }
    for (auto& future : futures) {
        future.wait();
    }
    return error / static_cast<float>(query_count);
}

void
// NOLINTNEXTLINE(readability-make-member-function-const)
KMeansCluster::select_initial_centroids_random(const float* datas,
                                               uint64_t count,
                                               uint32_t k,
                                               std::mt19937& gen) {
    std::uniform_int_distribution<uint64_t> dis(0, count - 1);
    for (uint32_t i = 0; i < k; ++i) {
        auto index = dis(gen);
        for (int32_t j = 0; j < dim_; ++j) {
            k_centroids_[i * dim_ + j] = datas[index * dim_ + j];
        }
    }
}

void
KMeansCluster::select_initial_centroids_kmeans_plus_plus(const float* datas,
                                                         uint64_t count,
                                                         uint32_t k,
                                                         std::mt19937& gen) {
    std::uniform_int_distribution<uint64_t> first_dis(0, count - 1);
    uint64_t first_idx = first_dis(gen);
    for (int32_t j = 0; j < dim_; ++j) {
        k_centroids_[j] = datas[first_idx * dim_ + j];
    }

    Vector<float> min_distances(count, std::numeric_limits<float>::max(), allocator_);

    for (uint32_t c = 1; c < k; ++c) {
        const float* centroid =
            k_centroids_ + static_cast<uint64_t>(c - 1) * static_cast<uint64_t>(dim_);

        for (uint64_t i = 0; i < count; ++i) {
            float dist = FP32ComputeL2Sqr(datas + i * dim_, centroid, dim_);
            min_distances[i] = std::min(min_distances[i], dist);
        }

        double total_weight = 0.0;
        for (uint64_t i = 0; i < count; ++i) {
            total_weight += min_distances[i];
        }

        if (total_weight <= 0.0) {
            std::uniform_int_distribution<uint64_t> dis(0, count - 1);
            uint64_t idx = dis(gen);
            for (int32_t j = 0; j < dim_; ++j) {
                k_centroids_[c * dim_ + j] = datas[idx * dim_ + j];
            }
            continue;
        }

        std::uniform_real_distribution<double> prob_dis(0.0, total_weight);
        double threshold = prob_dis(gen);
        double cumulative = 0.0;
        uint64_t selected_idx = count - 1;

        for (uint64_t i = 0; i < count; ++i) {
            cumulative += min_distances[i];
            if (cumulative >= threshold) {
                selected_idx = i;
                break;
            }
        }

        for (int32_t j = 0; j < dim_; ++j) {
            k_centroids_[c * dim_ + j] = datas[selected_idx * dim_ + j];
        }
    }
}

}  // namespace vsag
