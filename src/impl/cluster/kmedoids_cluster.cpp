
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

#include "kmedoids_cluster.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "nearest_centroid_assign.h"
#include "simd/fp32_simd.h"
#include "vsag_exception.h"

namespace vsag {
KMedoidsCluster::KMedoidsCluster(int32_t dim, Allocator* allocator, SafeThreadPoolPtr thread_pool)
    : allocator_(allocator),
      thread_pool_(std::move(thread_pool)),
      dim_(dim),
      medoid_ids_(allocator) {
    if (thread_pool_ == nullptr) {
        thread_pool_ = SafeThreadPool::FactoryDefaultThreadPool();
    }
}

KMedoidsCluster::~KMedoidsCluster() {
    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
}

Vector<int>
KMedoidsCluster::Run(uint32_t k,
                     const float* datas,
                     uint64_t count,
                     int iter,
                     double* err,
                     float threshold,
                     KMeansInitMethod init_method,
                     uint64_t seed) {
    if (allocator_ == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "allocator cannot be null");
    }
    if (dim_ <= 0) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "dim must be positive");
    }
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
    const auto centroid_size = static_cast<uint64_t>(k) * static_cast<uint64_t>(dim_);
    k_centroids_ = static_cast<float*>(allocator_->Allocate(centroid_size * sizeof(float)));
    medoid_ids_ = Vector<int64_t>(k, -1, allocator_);

    std::mt19937 gen;
    if (seed == 0) {
        std::random_device rd;
        gen.seed(rd());
    } else {
        std::seed_seq seed_sequence{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
        gen.seed(seed_sequence);
    }
    if (init_method == KMeansInitMethod::KMEANS_PLUS_PLUS) {
        select_initial_medoids_kmeans_plus_plus(datas, count, k, gen);
    } else {
        select_initial_medoids_random(datas, count, k, gen);
    }

    Vector<int32_t> labels(count, -1, allocator_);
    double total_err = NearestCentroidAssign(k_centroids_,
                                             k,
                                             datas,
                                             count,
                                             static_cast<uint64_t>(dim_),
                                             thread_pool_,
                                             allocator_,
                                             labels.data());
    double last_err = total_err;

    for (int it = 0; it < iter; ++it) {
        const bool changed = update_medoids(datas, count, labels);
        if (!changed) {
            break;
        }
        for (uint32_t c = 0; c < k; ++c) {
            const auto dst_offset = static_cast<uint64_t>(c) * static_cast<uint64_t>(dim_);
            const auto src_offset =
                static_cast<uint64_t>(medoid_ids_[c]) * static_cast<uint64_t>(dim_);
            std::memcpy(k_centroids_ + dst_offset,
                        datas + src_offset,
                        static_cast<uint64_t>(dim_) * sizeof(float));
        }
        const double new_err = NearestCentroidAssign(k_centroids_,
                                                     k,
                                                     datas,
                                                     count,
                                                     static_cast<uint64_t>(dim_),
                                                     thread_pool_,
                                                     allocator_,
                                                     labels.data());
        if (it > 0 && std::fabs(last_err - new_err) < threshold) {
            total_err = new_err;
            break;
        }
        last_err = new_err;
        total_err = new_err;
    }

    if (err != nullptr) {
        *err = total_err;
    }
    return labels;
}

void
KMedoidsCluster::select_initial_medoids_random(const float* datas,
                                               uint64_t count,
                                               uint32_t k,
                                               std::mt19937& gen) {
    std::vector<uint64_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen);
    for (uint32_t c = 0; c < k; ++c) {
        const auto index = indices[c];
        medoid_ids_[c] = static_cast<int64_t>(index);
        std::memcpy(k_centroids_ + static_cast<uint64_t>(c) * static_cast<uint64_t>(dim_),
                    datas + index * static_cast<uint64_t>(dim_),
                    static_cast<uint64_t>(dim_) * sizeof(float));
    }
}

void
KMedoidsCluster::select_initial_medoids_kmeans_plus_plus(const float* datas,
                                                         uint64_t count,
                                                         uint32_t k,
                                                         std::mt19937& gen) {
    std::uniform_int_distribution<uint64_t> first_dis(0, count - 1);
    const auto first_idx = first_dis(gen);
    std::vector<uint8_t> selected(count, 0);
    selected[first_idx] = 1;
    medoid_ids_[0] = static_cast<int64_t>(first_idx);
    std::memcpy(k_centroids_,
                datas + first_idx * static_cast<uint64_t>(dim_),
                static_cast<uint64_t>(dim_) * sizeof(float));

    Vector<float> min_distances(count, std::numeric_limits<float>::max(), allocator_);
    for (uint32_t c = 1; c < k; ++c) {
        const float* centroid =
            k_centroids_ + static_cast<uint64_t>(c - 1) * static_cast<uint64_t>(dim_);
        for (uint64_t i = 0; i < count; ++i) {
            const float dist = FP32ComputeL2Sqr(
                datas + i * static_cast<uint64_t>(dim_), centroid, static_cast<uint64_t>(dim_));
            min_distances[i] = std::min(min_distances[i], dist);
        }
        double total_weight = 0.0;
        for (uint64_t i = 0; i < count; ++i) {
            if (selected[i] == 0) {
                total_weight += min_distances[i];
            }
        }

        uint64_t selected_idx = count - 1;
        if (total_weight <= 0.0) {
            std::vector<uint64_t> candidates;
            candidates.reserve(count - c);
            for (uint64_t i = 0; i < count; ++i) {
                if (selected[i] == 0) {
                    candidates.push_back(i);
                }
            }
            std::uniform_int_distribution<uint64_t> dis(0, candidates.size() - 1);
            selected_idx = candidates[dis(gen)];
        } else {
            std::uniform_real_distribution<double> prob_dis(0.0, total_weight);
            const double threshold = prob_dis(gen);
            double cumulative = 0.0;
            for (uint64_t i = 0; i < count; ++i) {
                if (selected[i] == 0) {
                    cumulative += min_distances[i];
                    if (cumulative >= threshold) {
                        selected_idx = i;
                        break;
                    }
                }
            }
        }
        selected[selected_idx] = 1;
        medoid_ids_[c] = static_cast<int64_t>(selected_idx);
        std::memcpy(k_centroids_ + static_cast<uint64_t>(c) * static_cast<uint64_t>(dim_),
                    datas + selected_idx * static_cast<uint64_t>(dim_),
                    static_cast<uint64_t>(dim_) * sizeof(float));
    }
}

bool
KMedoidsCluster::update_medoids(const float* datas, uint64_t count, const Vector<int32_t>& labels) {
    const auto k = static_cast<uint64_t>(medoid_ids_.size());
    Vector<uint64_t> bucket_sizes(k + 1, 0, allocator_);
    for (uint64_t i = 0; i < count; ++i) {
        const auto label = labels[i];
        if (label >= 0 && label < static_cast<int32_t>(k)) {
            ++bucket_sizes[static_cast<uint64_t>(label) + 1];
        }
    }
    for (uint64_t c = 0; c < k; ++c) {
        bucket_sizes[c + 1] += bucket_sizes[c];
    }
    Vector<uint64_t> cursors(k, 0, allocator_);
    for (uint64_t c = 0; c < k; ++c) {
        cursors[c] = bucket_sizes[c];
    }
    Vector<uint64_t> members(count, 0, allocator_);
    for (uint64_t i = 0; i < count; ++i) {
        const auto label = labels[i];
        if (label >= 0 && label < static_cast<int32_t>(k)) {
            members[cursors[static_cast<uint64_t>(label)]++] = i;
        }
    }

    Vector<int64_t> new_ids(k, -1, allocator_);
    Vector<uint8_t> used_indices(count, 0, allocator_);
    for (uint64_t c = 0; c < k; ++c) {
        const auto medoid = medoid_ids_[c];
        if (medoid >= 0 && static_cast<uint64_t>(medoid) < count) {
            used_indices[static_cast<uint64_t>(medoid)] = 1;
        }
    }
    for (uint64_t c = 0; c < k; ++c) {
        if (bucket_sizes[c] != bucket_sizes[c + 1]) {
            continue;
        }
        for (uint64_t index = 0; index < count; ++index) {
            if (used_indices[index] == 0) {
                new_ids[c] = static_cast<int64_t>(index);
                used_indices[index] = 1;
                break;
            }
        }
        if (new_ids[c] < 0) {
            new_ids[c] = medoid_ids_[c];
        }
    }
    const uint64_t batch_size = CLUSTER_BATCH;
    std::vector<std::future<void>> futures;
    for (uint64_t start = 0; start < k; start += batch_size) {
        const auto end = std::min(start + batch_size, k);
        futures.emplace_back(thread_pool_->GeneralEnqueue([&, start, end]() {
            Vector<double> sums(static_cast<uint64_t>(dim_), 0.0, allocator_);
            Vector<float> mean(static_cast<uint64_t>(dim_), 0.0F, allocator_);
            for (uint64_t c = start; c < end; ++c) {
                const auto member_start = bucket_sizes[c];
                const auto member_end = bucket_sizes[c + 1];
                if (member_start == member_end) {
                    continue;
                }
                std::fill(sums.begin(), sums.end(), 0.0);
                const auto member_count = member_end - member_start;
                for (uint64_t p = member_start; p < member_end; ++p) {
                    const auto index = members[p];
                    const auto* data = datas + index * static_cast<uint64_t>(dim_);
                    for (int32_t d = 0; d < dim_; ++d) {
                        sums[static_cast<uint64_t>(d)] += data[d];
                    }
                }
                for (int32_t d = 0; d < dim_; ++d) {
                    mean[static_cast<uint64_t>(d)] = static_cast<float>(
                        sums[static_cast<uint64_t>(d)] / static_cast<double>(member_count));
                }
                uint64_t best_index = members[member_start];
                float best_distance =
                    FP32ComputeL2Sqr(datas + best_index * static_cast<uint64_t>(dim_),
                                     mean.data(),
                                     static_cast<uint64_t>(dim_));
                for (uint64_t p = member_start + 1; p < member_end; ++p) {
                    const auto index = members[p];
                    const auto distance =
                        FP32ComputeL2Sqr(datas + index * static_cast<uint64_t>(dim_),
                                         mean.data(),
                                         static_cast<uint64_t>(dim_));
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_index = index;
                    }
                }
                new_ids[c] = static_cast<int64_t>(best_index);
            }
        }));
    }
    for (auto& future : futures) {
        future.wait();
    }

    bool changed = false;
    for (uint64_t c = 0; c < k; ++c) {
        if (new_ids[c] != medoid_ids_[c]) {
            changed = true;
        }
        medoid_ids_[c] = new_ids[c];
    }
    return changed;
}

}  // namespace vsag
