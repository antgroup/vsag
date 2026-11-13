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

#include "e_means_cluster.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "impl/blas/blas_function.h"
#include "nearest_centroid_assign.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

constexpr double INITIAL_COUNT = 1e-10;
constexpr double STEP_FLOOR_RATIO = 0.1;
constexpr uint64_t REVIVAL_PERIOD = 5;
constexpr double REVIVAL_THRESHOLD = 0.1;
constexpr float SPLIT_EPS = 1.0F / 1024.0F;
constexpr double PI = 3.14159265358979323846;

// Table 2 of the paper (fitted on DEEP); batch law for n <= 2M.
constexpr double BATCH_C = 0.035;
constexpr double BATCH_K = 0.13;
constexpr double BATCH_N = 0.88;
constexpr double BATCH_T = 0.81;
constexpr uint64_t LARGE_N = 2'000'000ULL;
constexpr uint64_t LARGE_N_BATCH_DIVISOR = 8;
constexpr double STEP_C = 0.824;
constexpr double STEP_K = -0.46;
constexpr double STEP_B = 0.90;
constexpr double STEP_N = -0.45;
constexpr double STEP_T = -0.11;

/// Splits donor clusters into dead ones, as Faiss does: the dead cluster takes
/// half of a donor's mass, with the two centroids perturbed in opposite
/// directions.
void
ReviveDeadClusters(Vector<double>& counts,
                   Vector<float>& sums,
                   uint32_t k,
                   uint64_t dim,
                   Allocator* allocator,
                   std::mt19937& gen) {
    const double mean = std::accumulate(counts.begin(), counts.end(), 0.0) / k;
    Vector<double> weights(k, 0.0, allocator);
    for (uint32_t j = 0; j < k; ++j) {
        if (counts[j] >= REVIVAL_THRESHOLD * mean) {
            continue;
        }
        for (uint32_t p = 0; p < k; ++p) {
            weights[p] = std::max(counts[p] - 1.0, 0.0);
        }
        if (std::accumulate(weights.begin(), weights.end(), 0.0) <= 0.0) {
            return;
        }
        std::discrete_distribution<uint32_t> pick(weights.begin(), weights.end());
        const uint32_t donor = pick(gen);
        if (donor == j) {
            continue;
        }
        const double half = counts[donor] / 2.0;
        for (uint64_t i = 0; i < dim; ++i) {
            const float c = static_cast<float>(sums[donor * dim + i] / counts[donor]);
            const float up = (i % 2 == 0) ? 1.0F + SPLIT_EPS : 1.0F - SPLIT_EPS;
            sums[j * dim + i] = static_cast<float>(half) * c * up;
            sums[donor * dim + i] = static_cast<float>(half) * c * (2.0F - up);
        }
        counts[j] = half;
        counts[donor] = half;
    }
}

}  // namespace

EMeansCluster::EMeansCluster(int32_t dim, Allocator* allocator, SafeThreadPoolPtr thread_pool)
    : allocator_(allocator), thread_pool_(std::move(thread_pool)), dim_(dim) {
    if (thread_pool_ == nullptr) {
        thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
    }
}

EMeansCluster::~EMeansCluster() {
    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
}

uint64_t
EMeansCluster::DefaultBatchSize(uint64_t count, uint32_t k, int epochs) {
    double b = BATCH_C * std::pow(static_cast<double>(k), BATCH_K) *
               std::pow(static_cast<double>(count), BATCH_N) *
               std::pow(static_cast<double>(epochs), BATCH_T);
    if (count > LARGE_N) {
        b /= static_cast<double>(LARGE_N_BATCH_DIVISOR);
    }
    const double rounded = std::exp2(std::round(std::log2(std::max(b, 1.0))));
    return std::min(static_cast<uint64_t>(rounded), count);
}

double
EMeansCluster::DefaultPeakStep(uint64_t count, uint32_t k, int epochs, uint64_t batch_size) {
    const double a = STEP_C * std::pow(static_cast<double>(k), STEP_K) *
                     std::pow(static_cast<double>(batch_size), STEP_B) *
                     std::pow(static_cast<double>(count), STEP_N) *
                     std::pow(static_cast<double>(epochs), STEP_T);
    return std::min(a, 1.0);
}

Vector<int>
EMeansCluster::Run(uint32_t k,
                   const float* datas,
                   uint64_t count,
                   int epochs,
                   double* err,
                   std::optional<uint32_t> random_seed,
                   uint64_t batch_size,
                   double peak_step) {
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
    if (epochs <= 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "epochs must be positive");
    }
    if (peak_step > 1.0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "peak_step must be at most 1");
    }

    const uint64_t dim = static_cast<uint64_t>(dim_);
    const uint64_t kk = k;
    if (batch_size == 0) {
        batch_size = DefaultBatchSize(count, k, epochs);
    }
    batch_size = std::min(batch_size, count);
    if (peak_step <= 0.0) {
        peak_step = DefaultPeakStep(count, k, epochs, batch_size);
    }
    const uint64_t batches_per_epoch = count / batch_size;
    const uint64_t total_updates = static_cast<uint64_t>(epochs) * batches_per_epoch;

    std::random_device rd;
    std::mt19937 gen(random_seed.has_value() ? *random_seed : rd());

    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
    k_centroids_ = static_cast<float*>(allocator_->Allocate(kk * dim * sizeof(float)));

    Vector<uint32_t> order(count, 0, allocator_);
    std::iota(order.begin(), order.end(), 0U);
    Vector<double> counts(k, INITIAL_COUNT, allocator_);
    Vector<float> sums(kk * dim, 0.0F, allocator_);
    {
        // The first k entries of a shuffle are k distinct random points.
        std::shuffle(order.begin(), order.end(), gen);
        for (uint64_t j = 0; j < kk; ++j) {
            for (uint64_t i = 0; i < dim; ++i) {
                sums[j * dim + i] =
                    static_cast<float>(INITIAL_COUNT) * datas[uint64_t(order[j]) * dim + i];
            }
        }
    }

    Vector<float> batch(batch_size * dim, 0.0F, allocator_);
    Vector<int32_t> batch_labels(batch_size, 0, allocator_);
    Vector<float> batch_sums(kk * dim, 0.0F, allocator_);
    Vector<double> batch_counts(k, 0.0, allocator_);
    Vector<int32_t> last_labels(count, -1, allocator_);

    auto read_centroids = [&]() {
        for (uint64_t j = 0; j < kk; ++j) {
            const float inv = static_cast<float>(1.0 / counts[j]);
            for (uint64_t i = 0; i < dim; ++i) {
                k_centroids_[j * dim + i] = sums[j * dim + i] * inv;
            }
        }
    };

    uint64_t t = 0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), gen);
        for (uint64_t b = 0; b < batches_per_epoch; ++b, ++t) {
            const uint32_t* ids = order.data() + b * batch_size;
            for (uint64_t i = 0; i < batch_size; ++i) {
                std::copy_n(datas + uint64_t(ids[i]) * dim, dim, batch.data() + i * dim);
            }

            const double cosine =
                0.5 *
                (1.0 + std::cos(PI * static_cast<double>(t) / static_cast<double>(total_updates)));
            const double alpha = peak_step * (STEP_FLOOR_RATIO + (1.0 - STEP_FLOOR_RATIO) * cosine);
            const double gamma = 1.0 - alpha;

            read_centroids();
            NearestCentroidAssign(k_centroids_,
                                  kk,
                                  batch.data(),
                                  batch_size,
                                  dim,
                                  thread_pool_,
                                  allocator_,
                                  batch_labels.data());

            std::fill(batch_sums.begin(), batch_sums.end(), 0.0F);
            std::fill(batch_counts.begin(), batch_counts.end(), 0.0);
            const bool last_epoch = epoch == epochs - 1;
            for (uint64_t i = 0; i < batch_size; ++i) {
                const uint64_t label = static_cast<uint64_t>(batch_labels[i]);
                batch_counts[label] += 1.0;
                BlasFunction::Saxpy(
                    dim_, 1.0F, batch.data() + i * dim, 1, batch_sums.data() + label * dim, 1);
                if (last_epoch) {
                    last_labels[ids[i]] = batch_labels[i];
                }
            }

            for (uint64_t j = 0; j < kk; ++j) {
                counts[j] = gamma * counts[j] + alpha * batch_counts[j];
            }
            BlasFunction::Sscal(
                static_cast<int32_t>(kk * dim), static_cast<float>(gamma), sums.data(), 1);
            BlasFunction::Saxpy(static_cast<int32_t>(kk * dim),
                                static_cast<float>(alpha),
                                batch_sums.data(),
                                1,
                                sums.data(),
                                1);
            if ((t + 1) % REVIVAL_PERIOD == 0) {
                ReviveDeadClusters(counts, sums, k, dim, allocator_, gen);
            }
        }
    }

    // Last-epoch-mean finalization: the mean of the points assigned in the last
    // epoch; clusters it did not reach keep their EMA centroid.
    read_centroids();
    Vector<double> final_counts(k, 0.0, allocator_);
    Vector<float> final_sums(kk * dim, 0.0F, allocator_);
    for (uint64_t i = 0; i < count; ++i) {
        if (last_labels[i] < 0) {
            continue;
        }
        const uint64_t label = static_cast<uint64_t>(last_labels[i]);
        final_counts[label] += 1.0;
        BlasFunction::Saxpy(dim_, 1.0F, datas + i * dim, 1, final_sums.data() + label * dim, 1);
    }
    for (uint64_t j = 0; j < kk; ++j) {
        if (final_counts[j] <= 0.0) {
            continue;
        }
        const float inv = static_cast<float>(1.0 / final_counts[j]);
        for (uint64_t i = 0; i < dim; ++i) {
            k_centroids_[j * dim + i] = final_sums[j * dim + i] * inv;
        }
    }

    Vector<int> labels(count, 0, allocator_);
    Vector<int32_t> assigned(count, -1, allocator_);
    const double mean_err = NearestCentroidAssign(
        k_centroids_, kk, datas, count, dim, thread_pool_, allocator_, assigned.data());
    for (uint64_t i = 0; i < count; ++i) {
        labels[i] = assigned[i];
    }
    if (err != nullptr) {
        *err = mean_err;
    }
    return labels;
}

}  // namespace vsag
