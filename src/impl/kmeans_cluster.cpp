
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

#include <random>

#include "byte_buffer.h"
#include "simd/fp32_simd.h"

namespace vsag {

KMeansCluster::KMeansCluster(int32_t dim, Allocator* allocator) : dim_(dim), allocator_(allocator) {
}

KMeansCluster::~KMeansCluster() {
    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
}

Vector<int>
KMeansCluster::Run(uint32_t k, const float* datas, uint64_t count, int iter) {
    // Allocate space for centroids
    if (k_centroids_ != nullptr) {
        allocator_->Deallocate(k_centroids_);
        k_centroids_ = nullptr;
    }
    uint64_t size = static_cast<uint64_t>(k) * static_cast<uint64_t>(dim_) * sizeof(float);
    k_centroids_ = static_cast<float*>(allocator_->Allocate(size));

    // Initialize centroids randomly
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, count - 1);
    for (int i = 0; i < k; ++i) {
        auto index = dis(gen);
        for (int j = 0; j < dim_; ++j) {
            k_centroids_[i * dim_ + j] = datas[index * dim_ + j];
        }
    }

    ByteBuffer y_sqr_buffer(static_cast<uint64_t>(k) * sizeof(float), allocator_);
    ByteBuffer distances_buffer(static_cast<uint64_t>(k) * count * sizeof(float), allocator_);
    auto* y_sqr = reinterpret_cast<float*>(y_sqr_buffer.data);
    auto* distances = reinterpret_cast<float*>(distances_buffer.data);

    Vector<int> labels(count, -1, this->allocator_);
    bool have_empty = false;
    for (int it = 0; it < iter; ++it) {
        bool has_converged = true;

        for (int64_t i = 0; i < k; ++i) {
            y_sqr[i] = FP32ComputeIP(k_centroids_ + i * dim_, k_centroids_ + i * dim_, dim_);
        }

        cblas_sgemm(CblasColMajor,
                    CblasTrans,
                    CblasNoTrans,
                    static_cast<blasint>(k),
                    static_cast<blasint>(count),
                    dim_,
                    -2.0F,
                    k_centroids_,
                    dim_,
                    datas,
                    dim_,
                    0.0F,
                    distances,
                    static_cast<blasint>(k));

        for (uint64_t i = 0; i < count; ++i) {
            cblas_saxpy(static_cast<blasint>(k), 1.0, y_sqr, 1, distances + i * k, 1);
            auto* min_elem = std::min_element(distances + i * k, distances + i * k + k);
            auto min_index = std::distance(distances + i * k, min_elem);
            if (min_index != labels[i]) {
                labels[i] = static_cast<int>(min_index);
                has_converged = false;
            }
        }

        if (has_converged and not have_empty) {
            break;
        }

        // Update centroids
        Vector<int> counts(k, 0, allocator_);
        Vector<float> new_centroids(static_cast<uint64_t>(k) * dim_, 0.0F, allocator_);
        have_empty = false;
        for (uint64_t i = 0; i < count; ++i) {
            uint32_t label = labels[i];
            counts[label]++;
            cblas_saxpy(dim_,
                        1.0F,
                        datas + i * dim_,
                        1,
                        new_centroids.data() + label * static_cast<uint64_t>(dim_),
                        1);
        }

        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                cblas_sscal(dim_,
                            1.0F / static_cast<float>(counts[j]),
                            new_centroids.data() + j * static_cast<uint64_t>(dim_),
                            1);
                // Copy new centroids to k_centroids_
                std::copy(new_centroids.data() + j * static_cast<uint64_t>(dim_),
                          new_centroids.data() + (j + 1) * static_cast<uint64_t>(dim_),
                          k_centroids_ + j * static_cast<uint64_t>(dim_));
            } else {
                have_empty = true;
                auto index = dis(gen);
                for (int s = 0; s < dim_; ++s) {
                    k_centroids_[j * dim_ + s] = datas[index * dim_ + s];
                }
            }
        }
    }
    return labels;
}

}  // namespace vsag
