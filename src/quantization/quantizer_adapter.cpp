
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

#include "quantization/quantizer_adapter.h"

#include <cmath>
#include <memory>
#include <type_traits>

#include "quantization/computer.h"
#include "quantization/fp32_quantizer.h"
#include "quantization/quantizer.h"
#include "quantization/quantizer_headers.h"
#include "quantization/scalar_quantization/half_precision_quantizer.h"
#include "quantization/scalar_quantization/local_scalar_quantizer.h"
#include "quantization/scalar_quantization/scalar_quantizer.h"
#include "simd/bf16_simd.h"
#include "simd/fp16_simd.h"
#include "utils/half_input.h"

namespace vsag {

using generic::BF16ToFloat;
using generic::FloatToBF16;
using generic::FloatToFP16;
using half_input::FP16ToFloat;

namespace {
template <typename QuantT>
struct HalfInputTraits {
    static constexpr bool ENABLED = false;
};

template <typename Format, MetricType metric>
struct HalfInputTraits<HalfPrecisionQuantizer<Format, metric>> {
    static constexpr bool ENABLED = true;
    static constexpr DataTypes DATA_TYPE =
        std::is_same_v<Format, FP16Format> ? DataTypes::DATA_TYPE_FP16 : DataTypes::DATA_TYPE_BF16;
};
}  // namespace

template <typename QuantT, typename DataT>
QuantizerAdapter<QuantT, DataT>::QuantizerAdapter(const QuantizerParamPtr& param,
                                                  const IndexCommonParam& common_param)
    : Quantizer<QuantizerAdapter<QuantT, DataT>>(common_param.dim_, common_param.allocator_.get()),
      data_type_(common_param.data_type_) {
    auto inner_common_param = common_param;
    inner_common_param.data_type_ = DataTypes::DATA_TYPE_FLOAT;
    this->inner_quantizer_ = std::make_shared<QuantT>(param, inner_common_param);
    this->code_size_ = this->inner_quantizer_->GetCodeSize();
    this->query_code_size_ = this->inner_quantizer_->GetQueryCodeSize();
    this->metric_ = common_param.metric_;
    this->hold_molds_ = this->inner_quantizer_->HoldMolds();
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::TrainImpl(const float* data, uint64_t count) {
    if constexpr (std::is_same_v<DataT, int8_t>) {
        const auto* data_int8 = reinterpret_cast<const int8_t*>(data);
        Vector<float> vec(this->dim_ * count, this->allocator_);
        for (int64_t i = 0; i < this->dim_ * count; ++i) {
            vec[i] = static_cast<float>(data_int8[i]);
        }
        this->is_trained_ = this->inner_quantizer_->ReTrain(vec.data(), count);
        return this->is_trained_;
    } else if constexpr (std::is_same_v<DataT, uint16_t>) {
        const auto* data_fp16 = reinterpret_cast<const uint16_t*>(data);
        Vector<float> vec(this->dim_ * count, this->allocator_);
        for (int64_t i = 0; i < this->dim_ * count; ++i) {
            if (data_type_ == DataTypes::DATA_TYPE_FP16) {
                vec[i] = FP16ToFloat(data_fp16[i]);
            } else {
                vec[i] = BF16ToFloat(data_fp16[i]);
            }
        }
        this->is_trained_ = this->inner_quantizer_->ReTrain(vec.data(), count);
        return this->is_trained_;
    } else {
        static_assert(std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint16_t>,
                      "QuantizerAdapter::TrainImpl only supports int8_t and uint16_t data types");
        return false;
    }
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::EncodeOneImpl(const float* data, uint8_t* codes) {
    static_assert(std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint16_t>,
                  "QuantizerAdapter::EncodeOneImpl only supports int8_t and uint16_t data types");
    if constexpr (std::is_same_v<DataT, int8_t>) {
        const auto* data_int8 = reinterpret_cast<const int8_t*>(data);
        Vector<float> vec(this->dim_, this->allocator_);
        for (int64_t i = 0; i < this->dim_; i++) {
            vec[i] = static_cast<float>(data_int8[i]);
        }
        return this->inner_quantizer_->EncodeOneImpl(vec.data(), codes);
    } else if constexpr (std::is_same_v<DataT, uint16_t>) {
        const auto* data_fp16 = reinterpret_cast<const uint16_t*>(data);
        if constexpr (HalfInputTraits<QuantT>::ENABLED) {
            if (data_type_ == HalfInputTraits<QuantT>::DATA_TYPE) {
                return this->inner_quantizer_->EncodeOneHalf(data_fp16, codes);
            }
        }
        Vector<float> vec(this->dim_, this->allocator_);
        for (int64_t i = 0; i < this->dim_; i++) {
            if (data_type_ == DataTypes::DATA_TYPE_FP16) {
                vec[i] = FP16ToFloat(data_fp16[i]);
            } else {
                vec[i] = BF16ToFloat(data_fp16[i]);
            }
        }
        return this->inner_quantizer_->EncodeOneImpl(vec.data(), codes);
    }
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::EncodeBatchImpl(const float* data,
                                                 uint8_t* codes,
                                                 uint64_t count) {
    static_assert(std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint16_t>,
                  "QuantizerAdapter::EncodeBatchImpl only supports int8_t and uint16_t data types");
    if constexpr (std::is_same_v<DataT, int8_t>) {
        const auto* data_int8 = reinterpret_cast<const int8_t*>(data);
        Vector<float> vec(this->dim_ * count, this->allocator_);
        for (int64_t i = 0; i < this->dim_ * count; ++i) {
            vec[i] = static_cast<float>(data_int8[i]);
        }
        return this->inner_quantizer_->EncodeBatchImpl(vec.data(), codes, count);
    } else if constexpr (std::is_same_v<DataT, uint16_t>) {
        const auto* data_fp16 = reinterpret_cast<const uint16_t*>(data);
        if constexpr (HalfInputTraits<QuantT>::ENABLED) {
            if (data_type_ == HalfInputTraits<QuantT>::DATA_TYPE) {
                return this->inner_quantizer_->EncodeBatchHalf(data_fp16, codes, count);
            }
        }
        Vector<float> vec(this->dim_ * count, this->allocator_);
        for (int64_t i = 0; i < this->dim_ * count; ++i) {
            if (data_type_ == DataTypes::DATA_TYPE_FP16) {
                vec[i] = FP16ToFloat(data_fp16[i]);
            } else {
                vec[i] = BF16ToFloat(data_fp16[i]);
            }
        }
        return this->inner_quantizer_->EncodeBatchImpl(vec.data(), codes, count);
    }
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::DecodeOneImpl(const uint8_t* codes, float* data) {
    static_assert(std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint16_t>,
                  "QuantizerAdapter::DecodeOneImpl only supports int8_t and uint16_t data types");
    if constexpr (std::is_same_v<DataT, int8_t>) {
        Vector<float> vec(this->dim_, this->allocator_);
        if (!this->inner_quantizer_->DecodeOneImpl(codes, vec.data())) {
            return false;
        }
        for (int64_t i = 0; i < this->dim_; i++) {
            reinterpret_cast<DataT*>(data)[i] = static_cast<DataT>(std::round(vec[i]));
        }
        return true;
    } else if constexpr (std::is_same_v<DataT, uint16_t>) {
        Vector<float> vec(this->dim_, this->allocator_);
        if (!this->inner_quantizer_->DecodeOneImpl(codes, vec.data())) {
            return false;
        }
        auto* data_fp16 = reinterpret_cast<uint16_t*>(data);
        for (int64_t i = 0; i < this->dim_; i++) {
            if (data_type_ == DataTypes::DATA_TYPE_FP16) {
                data_fp16[i] = FloatToFP16(vec[i]);
            } else {
                data_fp16[i] = FloatToBF16(vec[i]);
            }
        }
        return true;
    }
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::DecodeBatchImpl(const uint8_t* codes,
                                                 float* data,
                                                 uint64_t count) {
    static_assert(std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint16_t>,
                  "QuantizerAdapter::DecodeBatchImpl only supports int8_t and uint16_t data types");
    if constexpr (std::is_same_v<DataT, int8_t>) {
        Vector<float> vec(this->dim_ * count, this->allocator_);
        if (!this->inner_quantizer_->DecodeBatchImpl(codes, vec.data(), count)) {
            return false;
        }
        for (int64_t i = 0; i < this->dim_ * count; i++) {
            reinterpret_cast<DataT*>(data)[i] = static_cast<DataT>(std::round(vec[i]));
        }
        return true;
    } else if constexpr (std::is_same_v<DataT, uint16_t>) {
        Vector<float> vec(this->dim_ * count, this->allocator_);
        if (!this->inner_quantizer_->DecodeBatchImpl(codes, vec.data(), count)) {
            return false;
        }
        auto* data_fp16 = reinterpret_cast<uint16_t*>(data);
        for (int64_t i = 0; i < this->dim_ * count; i++) {
            if (data_type_ == DataTypes::DATA_TYPE_FP16) {
                data_fp16[i] = FloatToFP16(vec[i]);
            } else {
                data_fp16[i] = FloatToBF16(vec[i]);
            }
        }
        return true;
    }
}
template <typename QuantT, typename DataT>
float
QuantizerAdapter<QuantT, DataT>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    return this->inner_quantizer_->ComputeImpl(codes1, codes2);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::SerializeImpl(StreamWriter& writer) {
    this->inner_quantizer_->SerializeImpl(writer);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::DeserializeImpl(StreamReader& reader) {
    // The adapter serializes the shared quantizer header only once. Restore
    // its training state before the inner payload validates trained models.
    this->inner_quantizer_->is_trained_ = this->is_trained_;
    this->inner_quantizer_->DeserializeImpl(reader);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ProcessQueryImpl(
    const float* query, Computer<QuantizerAdapter<QuantT, DataT>>& computer) const {
    static_assert(
        std::is_same_v<DataT, int8_t> || std::is_same_v<DataT, uint16_t>,
        "QuantizerAdapter::ProcessQueryImpl only supports int8_t and uint16_t data types");
    auto& inner_computer = get_inner_computer(computer);
    if constexpr (std::is_same_v<DataT, int8_t>) {
        const auto* query_int8 = reinterpret_cast<const int8_t*>(query);
        Vector<float> vec(this->dim_, this->allocator_);
        for (int64_t i = 0; i < this->dim_; i++) {
            vec[i] = static_cast<float>(query_int8[i]);
        }
        this->inner_quantizer_->ProcessQueryImpl(vec.data(), inner_computer);
    } else if constexpr (std::is_same_v<DataT, uint16_t>) {
        const auto* query_int16 = reinterpret_cast<const uint16_t*>(query);
        if constexpr (HalfInputTraits<QuantT>::ENABLED) {
            if (data_type_ == HalfInputTraits<QuantT>::DATA_TYPE) {
                this->inner_quantizer_->ProcessQueryHalf(query_int16, inner_computer);
                return;
            }
        }
        Vector<float> vec(this->dim_, this->allocator_);
        for (int64_t i = 0; i < this->dim_; i++) {
            if (data_type_ == DataTypes::DATA_TYPE_FP16) {
                vec[i] = FP16ToFloat(query_int16[i]);
            } else {
                vec[i] = BF16ToFloat(query_int16[i]);
            }
        }
        this->inner_quantizer_->ProcessQueryImpl(vec.data(), inner_computer);
    }
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ComputeDistImpl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer, const uint8_t* codes, float* dists) const {
    auto& inner_computer = get_inner_computer(computer);
    this->inner_quantizer_->ComputeDistImpl(inner_computer, codes, dists);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ComputeDistsBatch4Impl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer,
    const uint8_t* codes1,
    const uint8_t* codes2,
    const uint8_t* codes3,
    const uint8_t* codes4,
    float& dists1,
    float& dists2,
    float& dists3,
    float& dists4) const {
    auto& inner_computer = get_inner_computer(computer);
    this->inner_quantizer_->ComputeDistsBatch4(
        inner_computer, codes1, codes2, codes3, codes4, dists1, dists2, dists3, dists4);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ScanRowBatchDistImpl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer,
    uint64_t count,
    const uint8_t* codes,
    float* dists) const {
    this->inner_quantizer_->ScanRowBatchDists(get_inner_computer(computer), count, codes, dists);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ScanBatchDistImpl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer,
    uint64_t count,
    const uint8_t* codes,
    float* dists) const {
    auto& inner_computer = get_inner_computer(computer);
    this->inner_quantizer_->ScanBatchDists(inner_computer, count, codes, dists);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::ReleaseComputerImpl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer) const {
    computer.inner_computer_.reset();
    computer.allocator_->Deallocate(computer.buf_);
    computer.buf_ = nullptr;
}

template <typename QuantT, typename DataT>
Computer<QuantT>&
QuantizerAdapter<QuantT, DataT>::get_inner_computer(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer) const {
    if (computer.inner_computer_ == nullptr) {
        // Deliberately heap-own this cached, small control object: its typed
        // destructor releases native state. Query buffers still use the index
        // allocator; the control object is not included in its memory accounting.
        computer.inner_computer_ =
            std::make_unique<Computer<QuantT>>(this->inner_quantizer_.get(), computer.allocator_);
    }
    return *static_cast<Computer<QuantT>*>(computer.inner_computer_.get());
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::ComputeDistWithThresholdImpl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer,
    const uint8_t* codes,
    float threshold,
    float* dists) const {
    return this->inner_quantizer_->ComputeDistWithThreshold(
        get_inner_computer(computer), codes, threshold, dists);
}

template <typename QuantT, typename DataT>
bool
QuantizerAdapter<QuantT, DataT>::ComputeDistWithLowerBoundImpl(
    Computer<QuantizerAdapter<QuantT, DataT>>& computer,
    const uint8_t* codes,
    float* dists,
    float* lower_bound) const {
    return this->inner_quantizer_->ComputeDistWithLowerBound(
        get_inner_computer(computer), codes, dists, lower_bound);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::Package32(const uint8_t* codes,
                                           uint8_t* packaged_codes,
                                           int64_t valid_size) const {
    this->inner_quantizer_->Package32(codes, packaged_codes, valid_size);
}

template <typename QuantT, typename DataT>
void
QuantizerAdapter<QuantT, DataT>::Unpack32(const uint8_t* packaged_codes, uint8_t* codes) const {
    this->inner_quantizer_->Unpack32(packaged_codes, codes);
}

TEMPLATE_QUANTIZER_ADAPTER(ProductQuantizer, int8_t);
TEMPLATE_QUANTIZER_ADAPTER(ProductQuantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(FP16Quantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(BF16Quantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(SQ8Quantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(FP32Quantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(SQ4Quantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(SQ4UniformQuantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(SQ8UniformQuantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(PQFastScanQuantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(RaBitQuantizer, uint16_t);
TEMPLATE_QUANTIZER_ADAPTER(SAQQuantizer, uint16_t);
template class QuantizerAdapter<SQ8LocalQuantizer, uint16_t>;
template class QuantizerAdapter<SQ4LocalQuantizer, uint16_t>;

// Only the same-metric combinations selected by the factory are instantiated.
template class QuantizerAdapter<
    TransformQuantizer<SQ8Quantizer<MetricType::METRIC_TYPE_L2SQR>, MetricType::METRIC_TYPE_L2SQR>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SQ8Quantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<TransformQuantizer<SQ8Quantizer<MetricType::METRIC_TYPE_COSINE>,
                                                   MetricType::METRIC_TYPE_COSINE>,
                                uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<FP32Quantizer<MetricType::METRIC_TYPE_L2SQR>, MetricType::METRIC_TYPE_L2SQR>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<FP32Quantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<TransformQuantizer<FP32Quantizer<MetricType::METRIC_TYPE_COSINE>,
                                                   MetricType::METRIC_TYPE_COSINE>,
                                uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SQ4Quantizer<MetricType::METRIC_TYPE_L2SQR>, MetricType::METRIC_TYPE_L2SQR>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SQ4Quantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<TransformQuantizer<SQ4Quantizer<MetricType::METRIC_TYPE_COSINE>,
                                                   MetricType::METRIC_TYPE_COSINE>,
                                uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SQ4UniformQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                       MetricType::METRIC_TYPE_L2SQR>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SQ4UniformQuantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SQ4UniformQuantizer<MetricType::METRIC_TYPE_COSINE>,
                       MetricType::METRIC_TYPE_COSINE>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SQ8UniformQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                       MetricType::METRIC_TYPE_L2SQR>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SQ8UniformQuantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SQ8UniformQuantizer<MetricType::METRIC_TYPE_COSINE>,
                       MetricType::METRIC_TYPE_COSINE>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<BF16Quantizer<MetricType::METRIC_TYPE_L2SQR>, MetricType::METRIC_TYPE_L2SQR>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<BF16Quantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<TransformQuantizer<BF16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                                                   MetricType::METRIC_TYPE_COSINE>,
                                uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<FP16Quantizer<MetricType::METRIC_TYPE_L2SQR>, MetricType::METRIC_TYPE_L2SQR>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<FP16Quantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<TransformQuantizer<FP16Quantizer<MetricType::METRIC_TYPE_COSINE>,
                                                   MetricType::METRIC_TYPE_COSINE>,
                                uint16_t>;
template class QuantizerAdapter<TransformQuantizer<ProductQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                                   MetricType::METRIC_TYPE_L2SQR>,
                                uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<ProductQuantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<TransformQuantizer<ProductQuantizer<MetricType::METRIC_TYPE_COSINE>,
                                                   MetricType::METRIC_TYPE_COSINE>,
                                uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<PQFastScanQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                       MetricType::METRIC_TYPE_L2SQR>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<PQFastScanQuantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<PQFastScanQuantizer<MetricType::METRIC_TYPE_COSINE>,
                       MetricType::METRIC_TYPE_COSINE>,
    uint16_t>;
template class QuantizerAdapter<TransformQuantizer<RaBitQuantizer<MetricType::METRIC_TYPE_L2SQR>,
                                                   MetricType::METRIC_TYPE_L2SQR>,
                                uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<RaBitQuantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<TransformQuantizer<RaBitQuantizer<MetricType::METRIC_TYPE_COSINE>,
                                                   MetricType::METRIC_TYPE_COSINE>,
                                uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SAQQuantizer<MetricType::METRIC_TYPE_L2SQR>, MetricType::METRIC_TYPE_L2SQR>,
    uint16_t>;
template class QuantizerAdapter<
    TransformQuantizer<SAQQuantizer<MetricType::METRIC_TYPE_IP>, MetricType::METRIC_TYPE_IP>,
    uint16_t>;
template class QuantizerAdapter<TransformQuantizer<SAQQuantizer<MetricType::METRIC_TYPE_COSINE>,
                                                   MetricType::METRIC_TYPE_COSINE>,
                                uint16_t>;
}  // namespace vsag
