
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

#pragma once

#include <cblas.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>

#include "impl/kmeans_cluster.h"
#include "index/index_common_param.h"
#include "inner_string_params.h"
#include "prefetch.h"
#include "product_quantizer_parameter.h"
#include "quantization/quantizer.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class ProductQuantizer : public Quantizer<ProductQuantizer<metric>> {
public:
    explicit ProductQuantizer(int dim, int64_t pq_dim, Allocator* allocator);

    ProductQuantizer(const ProductQuantizerParamPtr& param, const IndexCommonParam& common_param);

    ProductQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~ProductQuantizer() = default;

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes);

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count);

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data);

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count);

    inline float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    inline void
    ProcessQueryImpl(const DataType* query, Computer<ProductQuantizer>& computer) const;

    inline void
    ComputeDistImpl(Computer<ProductQuantizer>& computer, const uint8_t* codes, float* dists) const;

    inline void
    ScanBatchDistImpl(Computer<ProductQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    inline void
    SerializeImpl(StreamWriter& writer);

    inline void
    DeserializeImpl(StreamReader& reader);

    inline void
    ReleaseComputerImpl(Computer<ProductQuantizer<metric>>& computer) const;

    [[nodiscard]] inline std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_PQ;
    }

private:
    [[nodiscard]] inline const float*
    get_codebook_data(int64_t subspace_idx, int64_t centroid_num) const {
        return this->codebooks_.data() + subspace_idx * subspace_dim_ * CENTROIDS_PER_SUBSPACE +
               centroid_num * subspace_dim_;
    }

public:
    constexpr static int64_t PQ_BITS = 8L;
    constexpr static int64_t CENTROIDS_PER_SUBSPACE = 256L;

public:
    int64_t pq_dim_{1};
    int64_t subspace_dim_{1};  // equal to dim/pq_dim_;

    Vector<float> codebooks_;
};

template <MetricType Metric>
ProductQuantizer<Metric>::ProductQuantizer(int dim, int64_t pq_dim, Allocator* allocator)
    : Quantizer<ProductQuantizer<Metric>>(dim, allocator), pq_dim_(pq_dim), codebooks_(allocator) {
    if (dim % pq_dim != 0) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("pq_dim({}) does not divide evenly into dim({})", pq_dim, dim));
    }
    this->code_size_ = this->pq_dim_;
    this->subspace_dim_ = this->dim_ / pq_dim;
    codebooks_.resize(this->dim_ * CENTROIDS_PER_SUBSPACE);
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

template <MetricType metric>
void
ProductQuantizer<metric>::ComputeDistImpl(Computer<ProductQuantizer>& computer,
                                          const uint8_t* codes,
                                          float* dists) const {
    auto* lut = reinterpret_cast<float*>(computer.buf_);
    dists[0] = 0.0F;
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
        dists[0] += dism;
    }
    for (; i < pq_dim_; ++i) {
        dists[0] += lut[*codes++];
        lut += CENTROIDS_PER_SUBSPACE;
    }
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE or
                  metric == MetricType::METRIC_TYPE_IP) {
        dists[0] = 1.0F - dists[0];
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
}

template <MetricType metric>
void
ProductQuantizer<metric>::ReleaseComputerImpl(Computer<ProductQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

}  // namespace vsag
