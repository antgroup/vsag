
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

#include "pq_fastscan_quantizer.h"

#include <cblas.h>

#include "impl/kmeans_cluster.h"
#include "index/index_common_param.h"
#include "pq_fastscan_quantizer_parameter.h"
#include "prefetch.h"
#include "quantization/quantizer.h"
#include "quantization/scalar_quantization/scalar_quantization_trainer.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"
#include "simd/pqfs_simd.h"

namespace vsag {

template <MetricType metric>
PQFastScanQuantizer<metric>::PQFastScanQuantizer(int dim, int64_t pq_dim, Allocator* allocator)
    : Quantizer<PQFastScanQuantizer<metric>>(dim, allocator),
      pq_dim_(pq_dim),
      codebooks_(allocator) {
    if (dim % pq_dim != 0) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("pq_dim({}) does not divide evenly into dim({})", pq_dim, dim));
    }
    this->code_size_ = (this->pq_dim_ + 1) / 2;
    this->subspace_dim_ = this->dim_ / pq_dim;
    this->metric_ = metric;
    codebooks_.resize(this->dim_ * CENTROIDS_PER_SUBSPACE);
    this->query_code_size_ =
        this->pq_dim_ * CENTROIDS_PER_SUBSPACE * sizeof(uint8_t) + 2 * sizeof(float);
}

template <MetricType metric>
PQFastScanQuantizer<metric>::PQFastScanQuantizer(const PQFastScanQuantizerParamPtr& param,
                                                 const IndexCommonParam& common_param)
    : PQFastScanQuantizer<metric>(
          common_param.dim_, param->pq_dim_, common_param.allocator_.get()) {
}

template <MetricType metric>
PQFastScanQuantizer<metric>::PQFastScanQuantizer(const QuantizerParamPtr& param,
                                                 const IndexCommonParam& common_param)
    : PQFastScanQuantizer<metric>(std::dynamic_pointer_cast<PQFastScanQuantizerParameter>(param),
                                  common_param) {
}

template <MetricType metric>
bool
PQFastScanQuantizer<metric>::TrainImpl(const vsag::DataType* data, uint64_t count) {
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
PQFastScanQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) {
    const DataType* cur = data;
    Vector<float> tmp(this->allocator_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        tmp.resize(this->dim_);
        Normalize(data, tmp.data(), this->dim_);
        cur = tmp.data();
    }
    memset(codes, 0, this->code_size_);
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
        if (i % 2 == 1) {
            nearest_id <<= 4L;
        }
        codes[i / 2] |= nearest_id;
    }
    return true;
}

template <MetricType metric>
bool
PQFastScanQuantizer<metric>::EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <MetricType metric>
bool
PQFastScanQuantizer<metric>::DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
        this->DecodeOneImpl(codes + i * this->code_size_, data + i * this->dim_);
    }
    return true;
}

template <MetricType metric>
bool
PQFastScanQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    for (int i = 0; i < pq_dim_; ++i) {
        auto idx = codes[i / 2];
        if (i % 2 == 0) {
            idx &= 0x0F;
        } else {
            idx >>= 4L;
        }
        memcpy(data + i * subspace_dim_,
               this->get_codebook_data(i, idx),
               subspace_dim_ * sizeof(float));
    }
    return true;
}

template <MetricType metric>
inline float
PQFastScanQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    throw VsagException(ErrorType::INTERNAL_ERROR, "PQFastScan doesn't support ComputeCodes");
}

template <MetricType metric>
void
PQFastScanQuantizer<metric>::ComputeDistImpl(Computer<PQFastScanQuantizer>& computer,
                                             const uint8_t* codes,
                                             float* dists) const {
    throw VsagException(ErrorType::INTERNAL_ERROR,
                        "PQFastScan doesn't support ComputeDist, only support ComputeBatchDist");
}

template <MetricType metric>
void
PQFastScanQuantizer<metric>::ScanBatchDistImpl(Computer<PQFastScanQuantizer<metric>>& computer,
                                               uint64_t count,
                                               const uint8_t* codes,
                                               float* dists) const {
    auto* sq_info =
        reinterpret_cast<float*>(computer.buf_ + this->pq_dim_ * CENTROIDS_PER_SUBSPACE);
    auto diff = sq_info[0];
    auto lower = sq_info[1];
    auto map_int32_to_float = [&](const int32_t* from, float* to, int64_t map_count) {
        for (int j = 0; j < map_count; ++j) {
            to[j] = static_cast<float>(from[j]) / 255.0F * diff + lower;
            if constexpr (metric == MetricType::METRIC_TYPE_COSINE or
                          metric == MetricType::METRIC_TYPE_IP) {
                to[j] = 1.0F - to[j];
            }
        }
    };

    uint64_t block_count = count / BLOCK_SIZE_PACKAGE;
    Vector<int32_t> tmp_dist(BLOCK_SIZE_PACKAGE, 0, this->allocator_);
    for (int64_t i = 0; i < block_count; ++i) {
        PQFastScanLookUp32(computer.buf_, codes, this->pq_dim_, tmp_dist.data());
        map_int32_to_float(tmp_dist.data(), dists, BLOCK_SIZE_PACKAGE);
        codes += BLOCK_SIZE_PACKAGE * this->code_size_;
        dists += BLOCK_SIZE_PACKAGE;
        memset(tmp_dist.data(), 0, BLOCK_SIZE_PACKAGE * sizeof(int32_t));
    }

    if (count > block_count * BLOCK_SIZE_PACKAGE) {
        PQFastScanLookUp32(computer.buf_, codes, this->pq_dim_, tmp_dist.data());
        map_int32_to_float(tmp_dist.data(), dists, count - block_count * BLOCK_SIZE_PACKAGE);
    }
}

template <MetricType metric>
void
PQFastScanQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, this->pq_dim_);
    StreamWriter::WriteObj(writer, this->subspace_dim_);
    StreamWriter::WriteVector(writer, this->codebooks_);
}

template <MetricType metric>
void
PQFastScanQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->pq_dim_);
    StreamReader::ReadObj(reader, this->subspace_dim_);
    StreamReader::ReadVector(reader, this->codebooks_);
}

template <MetricType metric>
void
PQFastScanQuantizer<metric>::ReleaseComputerImpl(
    Computer<PQFastScanQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <MetricType metric>
void
PQFastScanQuantizer<metric>::Package32(const uint8_t* codes,
                                       uint8_t* packaged_codes,
                                       int64_t valid_size) const {
    constexpr int32_t mapper[32] = {0, 16, 8,  24, 1, 17, 9,  25, 2, 18, 10, 26, 3, 19, 11, 27,
                                    4, 20, 12, 28, 5, 21, 13, 29, 6, 22, 14, 30, 7, 23, 15, 31};
    if (valid_size == -1) {
        valid_size = BLOCK_SIZE_PACKAGE;
    }

    auto get_code = [&](int64_t vector_index, int64_t space_index) -> uint8_t {
        if (vector_index >= valid_size) {
            return 0;
        }
        uint8_t code = codes[vector_index * this->code_size_ + space_index / 2];
        if (space_index % 2 == 0) {
            return code & 0x0F;
        }
        return code >> 4L;
    };
    memset(packaged_codes, 0, this->code_size_ * BLOCK_SIZE_PACKAGE);
    for (int i = 0; i < this->pq_dim_; ++i) {
        for (int j = 0; j < BLOCK_SIZE_PACKAGE; ++j) {
            auto code = get_code(mapper[j], i);
            if (j % 2 == 1) {
                code <<= 4L;
            }
            packaged_codes[i * BLOCK_SIZE_PACKAGE / 2 + j / 2] |= code;
        }
    }
}

template <MetricType metric>
void
PQFastScanQuantizer<metric>::Unpack32(const uint8_t* packaged_codes, uint8_t* codes) const {
    constexpr int32_t mapper[32] = {0, 16, 8,  24, 1, 17, 9,  25, 2, 18, 10, 26, 3, 19, 11, 27,
                                    4, 20, 12, 28, 5, 21, 13, 29, 6, 22, 14, 30, 7, 23, 15, 31};

    for (int64_t i = 0; i < this->pq_dim_; ++i) {
        for (int64_t j = 0; j < BLOCK_SIZE_PACKAGE; ++j) {
            int64_t block_base = i * (BLOCK_SIZE_PACKAGE / 2) + (j / 2);
            uint8_t byte = packaged_codes[block_base];

            uint8_t code;
            if (j % 2 == 0) {
                code = byte & 0x0F;
            } else {
                code = byte >> 4;
            }
            int64_t vector_index = mapper[j];

            int64_t code_offset = vector_index * this->code_size_ + (i / 2);

            if (i % 2 == 0) {
                codes[code_offset] = (codes[code_offset] & 0xF0) | code;
            } else {
                codes[code_offset] = (codes[code_offset] & 0x0F) | (code << 4);
            }
        }
    }
}

template <MetricType metric>
void
PQFastScanQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                              Computer<PQFastScanQuantizer>& computer) const {
    try {
        const float* cur_query = query;
        Vector<float> norm_vec(this->allocator_);
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            norm_vec.resize(this->dim_);
            Normalize(query, norm_vec.data(), this->dim_);
            cur_query = norm_vec.data();
        }
        Vector<float> lookup_table(this->pq_dim_ * CENTROIDS_PER_SUBSPACE, this->allocator_);
        if (computer.buf_ == nullptr) {
            computer.buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
        for (int i = 0; i < pq_dim_; ++i) {
            const auto* per_query = cur_query + i * subspace_dim_;
            const auto* per_code_book = get_codebook_data(i, 0);
            auto* per_result = lookup_table.data() + i * CENTROIDS_PER_SUBSPACE;
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

        ScalarQuantizationTrainer trainer(CENTROIDS_PER_SUBSPACE, 8);
        float upper;
        float lower;
        trainer.TrainUniform(
            lookup_table.data(), pq_dim_, upper, lower, false, SQTrainMode::CLASSIC);
        auto diff = upper - lower;
        int64_t j = 0;
        for (; j < this->pq_dim_ * CENTROIDS_PER_SUBSPACE; ++j) {
            computer.buf_[j] = (lookup_table[j] - lower) / diff * 255;
        }
        auto* sq_info = reinterpret_cast<float*>(computer.buf_ + j);
        sq_info[0] = diff;
        sq_info[1] = lower;
    } catch (const std::bad_alloc& e) {
        if (computer.buf_ != nullptr) {
            this->allocator_->Deallocate(computer.buf_);
        }
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }
}

TEMPLATE_QUANTIZER(PQFastScanQuantizer);
}  // namespace vsag
