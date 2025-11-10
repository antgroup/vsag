
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

#include <sstream>
#include <string>

#include "algorithm/ivf_partition/ivf_nearest_partition.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "residual_quantizer_parameter.h"

namespace vsag {

// =============================================================================
// Base Code Structure Layout
// -----------------------------------------------------------------------------
// Memory layout of encoded data:
//   - Quantized vector codes
//   - Squared residual and linear correction term
//   - Assigned centroid identifier
//
// +-------------------------+---------------------+--------------+------------------------+
// | Field                   | Data Type           | Size (bytes) | Offset (bytes)         |
// +-------------------------+---------------------+--------------+------------------------+
// | code (quantizer)        | uint8_t[m]          | m            | 0                      |
// | (x - c3)^2 + 2c(x - c3) | float               | 4            | m                      |
// | centroid_id (e.g., c3)  | uint32_t            | 4            | m + 4                  |
// +-------------------------+---------------------+--------------+------------------------+
//
// Notes:
//   m = quantizer_->code_size_
//   x = original vector, c3 = centroid 3
// =============================================================================

// =============================================================================
// Query Code Structure Layout
// -----------------------------------------------------------------------------
// Memory layout of encoded query data, consisting of:
//   - Quantized vector codes
//   - Squared distances from the query to multiple centroids (e.g., c1, c2, ...)
//
// +-------------------------+---------------------+--------------+------------------------+
// | Field                   | Data Type           | Size (bytes) | Offset (bytes)         |
// +-------------------------+---------------------+--------------+------------------------+
// | code (quantizer)        | uint8_t[m]          | m            | 0                      |
// | (q - c1)^2              | float               | 4            | m                      |
// | (q - c2)^2              | float               | 4            | m + 4                  |
// | ...                     | float               | 4 each       | m + 4 * k (k >= 1)     |
// | q^2                     | float               | 4            | m + 4 * n              |
// +-------------------------+---------------------+--------------+------------------------+
//
// Notes:
//   m = quantizer_->code_size_
//   q  = query vector
//   n = centroids_count_
//   ck = k-th centroid (e.g., c1, c2, ..., cn)
// =============================================================================

template <typename QuantTmpl, MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class ResidualQuantizer : public Quantizer<ResidualQuantizer<QuantTmpl, metric>> {
public:
    explicit ResidualQuantizer(const ResidualQuantizerParamPtr& param,
                               const IndexCommonParam& common_param);

    explicit ResidualQuantizer(const QuantizerParamPtr& param,
                               const IndexCommonParam& common_param);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const DataType* data, uint8_t* codes, uint64_t count) const;

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data) {
        return false;
    }

    bool
    DecodeBatchImpl(const uint8_t* codes, DataType* data, uint64_t count) {
        return false;
    }

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    void
    ProcessQueryImpl(const DataType* query,
                     Computer<ResidualQuantizer<QuantTmpl, metric>>& computer) const;

    void
    ComputeDistImpl(Computer<ResidualQuantizer<QuantTmpl, metric>>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    ScanBatchDistImpl(Computer<ResidualQuantizer<QuantTmpl, metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ReleaseComputerImpl(Computer<ResidualQuantizer<QuantTmpl, metric>>& computer) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_RQ;
    }

public:
    uint32_t centroids_count_{0};

    Vector<float> centroids_norm_;

    IVFPartitionStrategyPtr partition_strategy_{nullptr};

    uint32_t res_norm_offset_{0};

    uint32_t align_size_{0};

    std::shared_ptr<QuantTmpl> quantizer_;
};

template <typename QuantTmpl, MetricType metric>
ResidualQuantizer<QuantTmpl, metric>::ResidualQuantizer(const QuantizerParamPtr& param,
                                                        const IndexCommonParam& common_param)
    : ResidualQuantizer<QuantTmpl, metric>(
          std::dynamic_pointer_cast<ResidualQuantizerParameter>(param), common_param) {
}

template <typename QuantTmpl, MetricType metric>
ResidualQuantizer<QuantTmpl, metric>::ResidualQuantizer(const ResidualQuantizerParamPtr& param,
                                                        const IndexCommonParam& common_param)
    : Quantizer<ResidualQuantizer<QuantTmpl, metric>>(common_param.dim_,
                                                      common_param.allocator_.get()),
      centroids_count_(param->centroids_count_),
      centroids_norm_(param->centroids_count_, common_param.allocator_.get()) {
    /*
     * USAGE:
     * 1. when centroids_count == 0, centroids_storage != nullptr: use outside centroids (e.g., QG)
     * 2. when centroids_count != 0, centroids_storage == nullptr: use ivf-partition centroids (e.g., kmeans + Graph)
     * 3. when both is ok, use runtime InnerTransformParamPtr& param in Transform to control
     * 4. when both is default value, use 2. with centroids_count = DEFAULT_CENTROIDS_COUNT
     *
     */

    // 1. init partition
    JsonType json;
    json[IVF_PARTITION_STRATEGY_TYPE_KEY].SetString(IVF_PARTITION_STRATEGY_TYPE_NEAREST);
    json[IVF_TRAIN_TYPE_KEY].SetString(IVF_TRAIN_TYPE_KMEANS);
    auto ivf_partition_strategy_parameter = std::make_shared<IVFPartitionStrategyParameters>();
    ivf_partition_strategy_parameter->FromJson(json);
    this->partition_strategy_ = std::make_shared<IVFNearestPartition>(
        centroids_count_, common_param, ivf_partition_strategy_parameter);

    // 2. init quantizer (note that the metric of quantizer should be IP)
    auto detailed_quantizer_param =
        QuantizerParameter::GetQuantizerParameterByJson(param->base_quantizer_json_);
    this->quantizer_ = std::make_shared<QuantTmpl>(detailed_quantizer_param, common_param);

    // 3. compute norm offset
    align_size_ = sizeof(float);
    this->res_norm_offset_ =
        (this->quantizer_->GetCodeSize() + align_size_ - 1) / align_size_ * align_size_;

    // 4. compute code size
    this->code_size_ = this->res_norm_offset_ + sizeof(float) * 2;
    this->query_code_size_ =
        this->res_norm_offset_ + sizeof(float) * centroids_count_ + sizeof(float);
}

template <typename QuantTmpl, MetricType metric>
bool
ResidualQuantizer<QuantTmpl, metric>::TrainImpl(const DataType* data, uint64_t count) {
    // 1. sample and train clusters
    count = std::min(count, (uint64_t)TQ_MAX_TRAIN_COUNT);
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(data)->NumElements(count)->Owner(false);
    this->partition_strategy_->Train(dataset);

    // 2. compute norms
    Vector<float> centroid_vec(this->dim_, 0, this->allocator_);
    for (auto i = 0; i < centroids_count_; i++) {
        this->partition_strategy_->GetCentroid(i, centroid_vec);
        centroids_norm_[i] = FP32ComputeIP(centroid_vec.data(), centroid_vec.data(), this->dim_);
    }

    // 3. train quantizer
    this->is_trained_ = quantizer_->Train(data, count);
    return this->is_trained_;
}

template <typename QuantTmpl, MetricType metric>
bool
ResidualQuantizer<QuantTmpl, metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    // 1. get centroid
    Vector<float> centroid_vec(this->dim_, 0, this->allocator_);
    uint32_t centroid_id = this->partition_strategy_->ClassifyDatas(data, 1, 1)[0];
    this->partition_strategy_->GetCentroid(centroid_id, centroid_vec);

    // 2. compute residual part and norm
    Vector<float> data_buffer(this->dim_, 0, this->allocator_);  // x - c
    for (int i = 0; i < this->dim_; i++) {
        data_buffer[i] = data[i] - centroid_vec[i];
    }
    float n1 = FP32ComputeIP(data_buffer.data(), data_buffer.data(), this->dim_);  // (x - c)^2
    float n2 = 2 * FP32ComputeIP(data_buffer.data(),
                                 (const float*)(centroid_vec.data()),
                                 this->dim_);  // 2c * (x - c)

    // 3. store norm data
    *(float*)(codes + res_norm_offset_) = n1 + n2;
    *(uint32_t*)(codes + res_norm_offset_ + sizeof(float)) = centroid_id;

    // 4. execute quantize
    return quantizer_->EncodeOne(data_buffer.data(), codes);
};

template <typename QuantTmpl, MetricType metric>
void
ResidualQuantizer<QuantTmpl, metric>::ProcessQueryImpl(
    const vsag::DataType* query, Computer<ResidualQuantizer>& computer) const {
    // 0. allocate
    try {
        computer.inner_computer_->buf_ =
            reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
    } catch (const std::bad_alloc& e) {
        computer.inner_computer_->buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }

    // 1. pre-compute all |q - c|^2 and store them into meta
    Vector<float> centroid_vec(this->dim_, 0, this->allocator_);
    for (auto i = 0; i < centroids_count_; i++) {
        this->partition_strategy_->GetCentroid(i, centroid_vec);
        auto norm = FP32ComputeL2Sqr(query, (const float*)(centroid_vec.data()), this->dim_);
        *(float*)(computer.inner_computer_->buf_ + res_norm_offset_ + i * sizeof(float)) = norm;
    }
    *(float*)(computer.inner_computer_->buf_ + this->query_code_size_ - sizeof(float)) =
        FP32ComputeL2Sqr(query, query, this->dim_);

    // 2. execute quantize
    // note that only when computer.buf_ == nullptr, quantizer_ will allocate data to buf_
    quantizer_->ProcessQuery(query, *computer.inner_computer_);
};

template <typename QuantTmpl, MetricType metric>
void
ResidualQuantizer<QuantTmpl, metric>::ComputeDistImpl(Computer<ResidualQuantizer>& computer,
                                                      const uint8_t* codes,
                                                      float* dists) const {
    auto n1_n2 = *(float*)(codes + res_norm_offset_);
    auto centroid_id = *(uint32_t*)(codes + res_norm_offset_ + sizeof(float));
    auto n_3 =
        *(float*)(computer.inner_computer_->buf_ + res_norm_offset_ + centroid_id * sizeof(float));
    auto quantize_dist = quantizer_->ComputeDist(*(computer.inner_computer_), codes);

    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        dists[0] = n1_n2 + n_3 - 2 * quantize_dist;
    } else {
        auto q_sqr =
            *(float*)(computer.inner_computer_->buf_ + this->query_code_size_ - sizeof(float));
        auto c_sqr = centroids_norm_[centroid_id];
        auto qc = (n_3 - q_sqr - c_sqr) / 2.0;

        dists[0] = qc + quantize_dist;
    }
};

template <typename QuantTmpl, MetricType metric>
float
ResidualQuantizer<QuantTmpl, metric>::ComputeImpl(const uint8_t* codes1,
                                                  const uint8_t* codes2) const {
    throw std::runtime_error("not support compute dist between codes");
}

template <typename QuantTmpl, MetricType metric>
void
ResidualQuantizer<QuantTmpl, metric>::ScanBatchDistImpl(
    Computer<ResidualQuantizer<QuantTmpl, metric>>& computer,
    uint64_t count,
    const uint8_t* codes,
    float* dists) const {
    for (uint64_t i = 0; i < count; ++i) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <typename QuantTmpl, MetricType metric>
void
ResidualQuantizer<QuantTmpl, metric>::ReleaseComputerImpl(
    Computer<ResidualQuantizer<QuantTmpl, metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <typename QuantTmpl, MetricType metric>
bool
ResidualQuantizer<QuantTmpl, metric>::EncodeBatchImpl(const DataType* data,
                                                      uint8_t* codes,
                                                      uint64_t count) const {
    for (uint64_t i = 0; i < count; ++i) {
        EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

template <typename QuantTmpl, MetricType metric>
void
ResidualQuantizer<QuantTmpl, metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteVector(writer, this->centroids_norm_);

    this->partition_strategy_->Serialize(writer);

    this->quantizer_->Serialize(writer);
}

template <typename QuantTmpl, MetricType metric>
void
ResidualQuantizer<QuantTmpl, metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->centroids_norm_);

    this->partition_strategy_->Deserialize(reader);

    this->quantizer_->Deserialize(reader);
}

}  // namespace vsag
