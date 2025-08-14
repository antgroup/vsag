
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

#include "impl/kmeans_cluster.h"
#include "prefetch.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"

namespace vsag {

template <MetricType metric>
ProductQuantizer<metric>::ProductQuantizer(int dim, int64_t pq_dim, Allocator* allocator)
    : Quantizer<ProductQuantizer<metric>>(dim, allocator),
      pq_dim_(pq_dim),
      codebooks_(allocator),
      reverse_codebooks_(allocator) {
    if (dim % pq_dim != 0) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("pq_dim({}) does not divide evenly into dim({})", pq_dim, dim));
    }
    this->code_size_ = this->pq_dim_;
    this->query_code_size_ = this->pq_dim_ * CENTROIDS_PER_SUBSPACE * sizeof(float);
    this->metric_ = metric;
    this->subspace_dim_ = this->dim_ / pq_dim;
    codebooks_.resize(this->dim_ * CENTROIDS_PER_SUBSPACE);
    reverse_codebooks_.resize(this->dim_ * CENTROIDS_PER_SUBSPACE);
}

template <MetricType metric>
ProductQuantizer<metric>::ProductQuantizer(const ProductQuantizerParamPtr& param,
                                           const IndexCommonParam& common_param)
    : ProductQuantizer<metric>(common_param.dim_, param->pq_dim_, common_param.allocator_.get()) {
}

template <MetricType metric>
ProductQuantizer<metric>::ProductQuantizer(const QuantizerParamPtr& param,
                                           const IndexCommonParam& common_param)
    : ProductQuantizer<metric>(std::dynamic_pointer_cast<ProductQuantizerParameter>(param),
                               common_param) {
}

template <MetricType metric>
bool
ProductQuantizer<metric>::TrainImpl(const vsag::DataType* data, uint64_t count) {
    if (this->is_trained_) {
        return true;
    }
    count = std::min(count, 65536UL);
    Vector<float> slice(this->allocator_);
    slice.resize(count * subspace_dim_);
    Vector<float> norm_data(this->allocator_);
    const float* train_data = data;
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        norm_data.resize(count * this->dim_);
        for (int64_t i = 0; i < count; ++i) {
            Normalize(data + i * this->dim_, norm_data.data() + i * this->dim_, this->dim_);
        }
        train_data = norm_data.data();
    }

    for (int64_t i = 0; i < pq_dim_; ++i) {
        for (int64_t j = 0; j < count; ++j) {
            memcpy(slice.data() + j * subspace_dim_,
                   train_data + j * this->dim_ + i * subspace_dim_,
                   subspace_dim_ * sizeof(float));
        }
        KMeansCluster cluster(subspace_dim_, this->allocator_);
        cluster.Run(CENTROIDS_PER_SUBSPACE, slice.data(), count);
        memcpy(this->codebooks_.data() + i * CENTROIDS_PER_SUBSPACE * subspace_dim_,
               cluster.k_centroids_,
               CENTROIDS_PER_SUBSPACE * subspace_dim_ * sizeof(float));
    }
    this->transpose_codebooks();

    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
bool
ProductQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) {
    const DataType* cur = data;
    Vector<float> tmp(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        tmp.resize(this->dim_);
        Normalize(data, tmp.data(), this->dim_);
        cur = tmp.data();
    }
    for (int i = 0; i < pq_dim_; ++i) {
        // TODO(LHT): use blas
        float nearest_dis = std::numeric_limits<float>::max();
        uint8_t nearest_id = 0;
        const float* query = cur + i * subspace_dim_;
        const float* base = this->codebooks_.data() + i * subspace_dim_ * CENTROIDS_PER_SUBSPACE;
        for (int j = 0; j < CENTROIDS_PER_SUBSPACE; ++j) {
            float dist = FP32ComputeL2Sqr(query, base + j * subspace_dim_, subspace_dim_);
            if (dist < nearest_dis) {
                nearest_dis = dist;
                nearest_id = static_cast<uint8_t>(j);
            }
        }
        codes[i] = nearest_id;
    }
    return true;
}

template <MetricType metric>
bool
ProductQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
ProductQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
bool
ProductQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (int i = 0; i < pq_dim_; ++i) {
        auto idx = codes[i];
        memcpy(data + i * subspace_dim_,
               this->get_codebook_data(i, idx),
               subspace_dim_ * sizeof(float));
    }
    return true;
}

template <MetricType metric>
inline float
ProductQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    float dist = 0.0F;
    for (int i = 0; i < pq_dim_; ++i) {
        const auto* vec1 = get_codebook_data(i, codes1[i]);
        const auto* vec2 = get_codebook_data(i, codes2[i]);
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            dist += FP32ComputeL2Sqr(vec1, vec2, subspace_dim_);
        } else if constexpr (metric == MetricType::METRIC_TYPE_IP or
                             metric == MetricType::METRIC_TYPE_COSINE) {
            dist += FP32ComputeIP(vec1, vec2, subspace_dim_);
        }
    }
    return dist;
}

template <MetricType metric>
void
ProductQuantizer<metric>::ComputeDistImpl(Computer<ProductQuantizer>& computer,
                                          const uint8_t* codes,
                                          float* dists) const {
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    float dist = 0.0F;
    int64_t i = 0;
    for (; i + 4 < pq_dim_; i += 4) {
        float dism = 0;
        dism = lut[*codes++];
        lut += CENTROIDS_PER_SUBSPACE;
        dism += lut[*codes++];
        lut += CENTROIDS_PER_SUBSPACE;
        dism += lut[*codes++];
        lut += CENTROIDS_PER_SUBSPACE;
        dism += lut[*codes++];
        lut += CENTROIDS_PER_SUBSPACE;
        dist += dism;
    }
    for (; i < pq_dim_; ++i) {
        dist += lut[*codes++];
        lut += CENTROIDS_PER_SUBSPACE;
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE or
                  metric == MetricType::METRIC_TYPE_IP) {
        dists[0] = 1.0F - dist;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists[0] = dist;
    }
}

template <MetricType metric>
void
ProductQuantizer<metric>::ComputeDistsBatch4Impl(Computer<ProductQuantizer<metric>>& computer,
                                                 const uint8_t* codes1,
                                                 const uint8_t* codes2,
                                                 const uint8_t* codes3,
                                                 const uint8_t* codes4,
                                                 float& dists1,
                                                 float& dists2,
                                                 float& dists3,
                                                 float& dists4) const {
    auto* lut = reinterpret_cast<float*>(computer.buf_);

    float d0 = 0.0F;
    float d1 = 0.0F;
    float d2 = 0.0F;
    float d3 = 0.0F;

    int64_t i = 0;

    // Main loop: process 4 PQ dimensions per iteration
    for (; i + 3 < pq_dim_; i += 4) {
        const float* l0 = lut + (i + 0) * CENTROIDS_PER_SUBSPACE;
        const float* l1 = lut + (i + 1) * CENTROIDS_PER_SUBSPACE;
        const float* l2 = lut + (i + 2) * CENTROIDS_PER_SUBSPACE;
        const float* l3 = lut + (i + 3) * CENTROIDS_PER_SUBSPACE;

        d0 += l0[codes1[i + 0]];
        d1 += l0[codes2[i + 0]];
        d2 += l0[codes3[i + 0]];
        d3 += l0[codes4[i + 0]];

        d0 += l1[codes1[i + 1]];
        d1 += l1[codes2[i + 1]];
        d2 += l1[codes3[i + 1]];
        d3 += l1[codes4[i + 1]];

        d0 += l2[codes1[i + 2]];
        d1 += l2[codes2[i + 2]];
        d2 += l2[codes3[i + 2]];
        d3 += l2[codes4[i + 2]];

        d0 += l3[codes1[i + 3]];
        d1 += l3[codes2[i + 3]];
        d2 += l3[codes3[i + 3]];
        d3 += l3[codes4[i + 3]];
    }

    // Tail loop: handle remaining dimensions
    for (; i < pq_dim_; ++i) {
        const float* li = lut + i * CENTROIDS_PER_SUBSPACE;

        d0 += li[codes1[i]];
        d1 += li[codes2[i]];
        d2 += li[codes3[i]];
        d3 += li[codes4[i]];
    }

    // Apply final distance transformation based on metric type
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE ||
                  metric == MetricType::METRIC_TYPE_IP) {
        dists1 = 1.0F - d0;
        dists2 = 1.0F - d1;
        dists3 = 1.0F - d2;
        dists4 = 1.0F - d3;
    } else if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists1 = d0;
        dists2 = d1;
        dists3 = d2;
        dists4 = d3;
    }
}

template <MetricType metric>
void
ProductQuantizer<metric>::ScanBatchDistImpl(Computer<ProductQuantizer<metric>>& computer,
                                            uint64_t count,
                                            const uint8_t* codes,
                                            float* dists) const {
    // TODO(LHT): Optimize batch for simd
    for (uint64_t i = 0; i < count; ++i) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <MetricType metric>
void
ProductQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, this->pq_dim_);
    StreamWriter::WriteObj(writer, this->subspace_dim_);
    StreamWriter::WriteVector(writer, this->codebooks_);
}

template <MetricType metric>
void
ProductQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->pq_dim_);
    StreamReader::ReadObj(reader, this->subspace_dim_);
    StreamReader::ReadVector(reader, this->codebooks_);
    this->transpose_codebooks();
}

template <MetricType metric>
void
ProductQuantizer<metric>::ReleaseComputerImpl(Computer<ProductQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <MetricType metric>
void
ProductQuantizer<metric>::transpose_codebooks() {
    for (int64_t i = 0; i < this->pq_dim_; ++i) {
        for (int64_t j = 0; j < CENTROIDS_PER_SUBSPACE; ++j) {
            memcpy(this->reverse_codebooks_.data() + j * this->dim_ + i * subspace_dim_,
                   this->codebooks_.data() + i * CENTROIDS_PER_SUBSPACE * subspace_dim_ +
                       j * subspace_dim_,
                   subspace_dim_ * sizeof(float));
        }
    }
}

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
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
        auto* lookup_table = reinterpret_cast<float*>(computer.buf_);

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

    } catch (const std::bad_alloc& e) {
        if (computer.buf_ != nullptr) {
            this->allocator_->Deallocate(computer.buf_);
        }
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
}

TEMPLATE_QUANTIZER(ProductQuantizer)
}  // namespace vsag