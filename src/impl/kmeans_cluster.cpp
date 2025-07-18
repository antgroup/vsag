
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

#include <cblas.h>
#include <omp.h>

#include <random>

#include "algorithm/inner_index_interface.h"
#include "byte_buffer.h"
#include "diskann_logger.h"
#include "impl/allocator/safe_allocator.h"
#include "simd/fp32_simd.h"
#include "utils/util_functions.h"

namespace vsag {
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
                   float threshold) {
    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
    uint64_t size = static_cast<uint64_t>(k) * static_cast<uint64_t>(dim_) * sizeof(float);
    k_centroids_ = static_cast<float*>(allocator_->Allocate(size));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, count - 1);
    for (int i = 0; i < k; ++i) {
        auto index = dis(gen);
        for (int j = 0; j < dim_; ++j) {
            k_centroids_[i * dim_ + j] = datas[index * dim_ + j];
        }
    }

    double total_err = std::numeric_limits<double>::max();
    double last_err = std::numeric_limits<double>::max();
    Vector<int32_t> labels(count, -1, this->allocator_);
    std::vector<std::mutex> mutexes(k);
    std::vector<std::future<void>> futures;
    ByteBuffer y_sqr_buffer(static_cast<uint64_t>(k) * sizeof(float), allocator_);
    ByteBuffer distances_buffer(static_cast<uint64_t>(k) * QUERY_BS * sizeof(float), allocator_);
    auto* y_sqr = reinterpret_cast<float*>(y_sqr_buffer.data);
    auto* distances = reinterpret_cast<float*>(distances_buffer.data);

    logger::trace("KMeansCluster::Run k: {}, count: {}, iter: {}", k, count, iter);
    if (k < THRESHOLD_FOR_HGRAPH) {
        logger::trace("KMeansCluster::Run use blas");
    } else {
        logger::trace("KMeansCluster::Run use hgraph");
    }

    for (int it = 0; it < iter; ++it) {
        logger::trace("[{}] KMeansCluster::Run iter: {}/{}, cur loss is {}",
                      get_current_time(),
                      it,
                      iter,
                      total_err);
        if (k < THRESHOLD_FOR_HGRAPH) {
            total_err = this->find_nearest_one_with_blas(datas, count, k, y_sqr, distances, labels);
        } else {
            total_err = this->find_nearest_one_with_hgraph(datas, count, k, labels);
        }
        constexpr uint64_t bs = 1024;

        Vector<int> counts(k, 0, allocator_);
        Vector<float> new_centroids(static_cast<uint64_t>(k) * dim_, 0.0F, allocator_);

        auto update_centroids_func = [&](uint64_t start, uint64_t end) {
            omp_set_num_threads(1);
            for (uint64_t i = start; i < end; ++i) {
                uint32_t label = labels[i];
                {
                    std::lock_guard<std::mutex> lock(mutexes[label]);
                    counts[label]++;
                    cblas_saxpy(dim_,
                                1.0F,
                                datas + i * dim_,
                                1,
                                new_centroids.data() + label * static_cast<uint64_t>(dim_),
                                1);
                }
            }
        };
        for (uint64_t i = 0; i < count; i += bs) {
            futures.emplace_back(
                thread_pool_->GeneralEnqueue(update_centroids_func, i, std::min(i + bs, count)));
        }
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();

        if (it > 0 && use_mse_for_convergence &&
            std::fabs(last_err - total_err) / static_cast<double>(count) < threshold) {
            break;
        }

        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                cblas_sscal(dim_,
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
        last_err = total_err;
    }
    if (err != nullptr) {
        *err = total_err;
    }
    return labels;
}

double
KMeansCluster::find_nearest_one_with_blas(const float* query,
                                          const uint64_t query_count,
                                          const uint64_t k,
                                          float* y_sqr,
                                          float* distances,
                                          Vector<int32_t>& labels) {
    double error = 0.0;
    std::mutex error_mutex;
    if (k_centroids_ == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "k_centroids_ is nullptr");
    }

    auto& thread_pool = this->thread_pool_;
    auto bs = 1024;
    std::vector<std::future<void>> futures;

    auto wait_futures_and_clear = [&]() {
        for (auto& future : futures) {
            future.wait();
        }
        futures.clear();
    };

    auto compute_ip_func = [&](uint64_t start, uint64_t end) -> void {
        for (uint64_t i = start; i < end; ++i) {
            y_sqr[i] = FP32ComputeIP(k_centroids_ + i * dim_, k_centroids_ + i * dim_, dim_);
        }
    };
    for (uint64_t i = 0; i < static_cast<uint64_t>(k); i += bs) {
        futures.emplace_back(thread_pool->GeneralEnqueue(
            compute_ip_func, i, std::min(i + bs, static_cast<uint64_t>(k))));
    }
    wait_futures_and_clear();

    for (uint64_t i = 0; i < query_count; i += QUERY_BS) {
        auto end = std::min(i + QUERY_BS, query_count);
        auto cur_query_count = end - i;
        auto* cur_label = labels.data() + i;

        cblas_sgemm(CblasColMajor,
                    CblasTrans,
                    CblasNoTrans,
                    static_cast<blasint>(k),
                    static_cast<blasint>(cur_query_count),
                    dim_,
                    -2.0F,
                    k_centroids_,
                    dim_,
                    query + i * dim_,
                    dim_,
                    0.0F,
                    distances,
                    static_cast<blasint>(k));

        auto assign_labels_func = [&](uint64_t start, uint64_t end) -> void {
            omp_set_num_threads(1);
            double thread_local_error = 0.0;
            for (uint64_t i = start; i < end; ++i) {
                cblas_saxpy(static_cast<blasint>(k), 1.0, y_sqr, 1, distances + i * k, 1);
                auto* min_elem = std::min_element(distances + i * k, distances + i * k + k);
                auto x_sqr = FP32ComputeIP(query + i * dim_, query + i * dim_, dim_);
                auto min_index = std::distance(distances + i * k, min_elem);
                thread_local_error += static_cast<double>(*min_elem + x_sqr);
                if (min_index != cur_label[i]) {
                    cur_label[i] = static_cast<int>(min_index);
                }
            }
            {
                std::lock_guard<std::mutex> lock(error_mutex);
                error += thread_local_error;
            }
        };
        for (uint64_t j = 0; j < cur_query_count; j += bs) {
            futures.emplace_back(thread_pool->GeneralEnqueue(
                assign_labels_func, j, std::min(j + bs, cur_query_count)));
        }
        wait_futures_and_clear();
    }
    return error / static_cast<float>(query_count);
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

    auto hgraph = InnerIndexInterface::FastCreateIndex("hgraph|32|fp32", param);
    auto base = Dataset::Make();
    Vector<int64_t> ids(k, allocator_);
    std::iota(ids.begin(), ids.end(), 0);
    base->Dim(dim_)
        ->NumElements(static_cast<int64_t>(k))
        ->Float32Vectors(this->k_centroids_)
        ->Ids(ids.data())
        ->Owner(false);
    hgraph->Build(base);
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

}  // namespace vsag
