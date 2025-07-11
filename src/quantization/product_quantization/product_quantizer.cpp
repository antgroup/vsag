
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

#include "product_quantizer.h"

#include <cblas.h>

namespace vsag {

template <MetricType metric>
void
ProductQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                           Computer<ProductQuantizer>& computer) const {
    try {
        const float* cur_query = query;
        Vector<float> norm_vec(this->allocator_);
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            norm_vec.resize(this->dim_);
            Normalize(query, norm_vec.data(), this->dim_);
            cur_query = norm_vec.data();
        }
        auto* lookup_table = reinterpret_cast<float*>(
            this->allocator_->Allocate(this->pq_dim_ * CENTROIDS_PER_SUBSPACE * sizeof(float)));

        for (int i = 0; i < pq_dim_; ++i) {
            const auto* per_query = cur_query + i * subspace_dim_;
            const auto* per_code_book = get_codebook_data(i, 0);
            auto* per_result = lookup_table + i * CENTROIDS_PER_SUBSPACE;
            if constexpr (metric == MetricType::METRIC_TYPE_IP or
                          metric == MetricType::METRIC_TYPE_COSINE) {
                cblas_sgemv(CblasRowMajor,
                            CblasNoTrans,
                            CENTROIDS_PER_SUBSPACE,
                            subspace_dim_,
                            1.0F,
                            per_code_book,
                            subspace_dim_,
                            per_query,
                            1,
                            0.0F,
                            per_result,
                            1);
            } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
                // TODO(LHT): use blas opt
                for (int64_t j = 0; j < CENTROIDS_PER_SUBSPACE; ++j) {
                    per_result[j] = FP32ComputeL2Sqr(
                        per_query, per_code_book + j * subspace_dim_, subspace_dim_);
                }
            }
        }

        computer.buf_ = reinterpret_cast<uint8_t*>(lookup_table);

    } catch (const std::bad_alloc& e) {
        if (computer.buf_ != nullptr) {
            this->allocator_->Deallocate(computer.buf_);
        }
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
}

template class ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>;
template class ProductQuantizer<MetricType::METRIC_TYPE_COSINE>;
template class ProductQuantizer<MetricType::METRIC_TYPE_IP>;

}  // namespace vsag