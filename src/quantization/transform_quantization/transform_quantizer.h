
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

#include "impl/transform/transformer_headers.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "transform_quantizer_parameter.h"

namespace vsag {

/***
 * @brief Transform Quantizer applies transformation chain before base quantization.
 *
 * code layout:
 * +----------------+----------+----------+-----+----------+
 * | base-code      | meta[0]  | meta[1]  | ... | meta[N]  |
 * | [aligned size] | [var]    | [var]    |     | [var]    |
 * +----------------+----------+----------+-----+----------+
 *
 * query layout:
 * +----------------+----------+----------+-----+----------+
 * | base-code      | meta[0]  | meta[1]  | ... | meta[N]  |
 * | [aligned size] | [var]    | [var]    |     | [var]    |
 * +----------------+----------+----------+-----+----------+
 *
 * - base-code: quantized code from inner quantizer (aligned)
 * - meta[i]: metadata from i-th transformer in chain
 * - Transformers: PCA, FHT, ROM, MRLE
 * - Distance is recovered through chain of transformers
 */
template <typename QuantTmpl, MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class TransformQuantizer : public Quantizer<TransformQuantizer<QuantTmpl, metric>> {
public:
    explicit TransformQuantizer(const TransformQuantizerParamPtr& param,
                                const IndexCommonParam& common_param);

    explicit TransformQuantizer(const QuantizerParamPtr& param,
                                const IndexCommonParam& common_param);

    bool
    TrainImpl(const float* data, uint64_t count);

    bool
    EncodeOneImpl(const float* data, uint8_t* codes) const;

    bool
    EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count) const;

    bool
    DecodeOneImpl(const uint8_t* codes, float* data) {
        return false;
    }

    bool
    DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count) {
        return false;
    }

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    void
    ProcessQueryImpl(const float* query,
                     Computer<TransformQuantizer<QuantTmpl, metric>>& computer) const;

    void
    ComputeDistImpl(Computer<TransformQuantizer<QuantTmpl, metric>>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    ScanBatchDistImpl(Computer<TransformQuantizer<QuantTmpl, metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    void
    ReleaseComputerImpl(Computer<TransformQuantizer<QuantTmpl, metric>>& computer) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_TQ;
    }

    [[nodiscard]] uint64_t
    GetTransformedDim() const {
        return quantizer_->GetDim();
    }

    void
    TransformBaseVector(const float* input, Vector<float>& output) const;

    bool
    EncodeOneWithScratch(const float* data,
                         uint8_t* codes,
                         Vector<float>& primary_scratch,
                         Vector<float>& secondary_scratch) const;

public:
    VectorTransformerPtr
    MakeTransformerInstance(std::string transform_str,
                            const VectorTransformerParameter& param) const;

    const float*
    ExecuteChainTransform(const float* input,
                          const uint32_t* meta_offsets,
                          uint8_t* codes,
                          Vector<float>& primary_scratch,
                          Vector<float>& secondary_scratch) const;

    float
    ExecuteChainDistanceRecovery(float quantize_dist,
                                 const uint32_t* meta_offsets_1,
                                 const uint32_t* meta_offsets_2,
                                 const uint8_t* codes_1,
                                 const uint8_t* codes_2) const;

public:
    Vector<uint32_t> base_meta_offsets_;   // note that code(quantizer) offset is always 0
    Vector<uint32_t> query_meta_offsets_;  // note that code(quantizer) offset is always 0

    uint32_t align_size_{0};

    std::shared_ptr<QuantTmpl> quantizer_;

    std::vector<VectorTransformerPtr> transform_chain_;
};

template <typename QuantTmpl, MetricType metric>
TransformQuantizer<QuantTmpl, metric>::TransformQuantizer(const QuantizerParamPtr& param,
                                                          const IndexCommonParam& common_param)
    : TransformQuantizer<QuantTmpl, metric>(
          std::dynamic_pointer_cast<TransformQuantizerParameter>(param), common_param) {
}

template <typename QuantTmpl, MetricType metric>
TransformQuantizer<QuantTmpl, metric>::TransformQuantizer(const TransformQuantizerParamPtr& param,
                                                          const IndexCommonParam& common_param)
    : Quantizer<TransformQuantizer<QuantTmpl, metric>>(common_param.dim_,
                                                       common_param.allocator_.get()),
      base_meta_offsets_(common_param.allocator_.get()),
      query_meta_offsets_(common_param.allocator_.get()) {
    // 1. init transform chain
    VectorTransformerParameter transformer_param;
    transformer_param.FromJson(param->base_quantizer_json_);
    transformer_param.input_dim_ = this->dim_;
    for (const auto& transform_str : param->tq_chain_) {
        transform_chain_.emplace_back(MakeTransformerInstance(transform_str, transformer_param));
        if (transform_chain_.back()->GetType() == VectorTransformerType::MRLE and
            transform_chain_.size() > 1) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("MRLE must be first if exists"));
        }
        transformer_param.input_dim_ = transform_chain_.back()->GetOutputDim();
    }

    // 2. init quantizer
    IndexCommonParam copy_common_param = common_param;
    copy_common_param.dim_ = transformer_param.input_dim_;
    auto detailed_quantizer_param =
        QuantizerParameter::GetQuantizerParameterByJson(param->base_quantizer_json_);
    this->quantizer_ = std::make_shared<QuantTmpl>(detailed_quantizer_param, copy_common_param);

    // 3. compute align_size
    align_size_ = sizeof(float);
    for (const auto& vector_transformer : this->transform_chain_) {
        auto meta_size = vector_transformer->GetMetaSize();
        auto align_size = vector_transformer->GetAlignSize();
        align_size_ = std::max(align_size_, align_size);
    }

    // 4. compute code_size
    this->code_size_ = ((quantizer_->GetCodeSize() + align_size_ - 1) / align_size_) * align_size_;
    this->query_code_size_ =
        ((quantizer_->GetQueryCodeSize() + align_size_ - 1) / align_size_) * align_size_;

    for (const auto& vector_transformer : this->transform_chain_) {
        base_meta_offsets_.push_back(this->code_size_);
        query_meta_offsets_.push_back(this->query_code_size_);
        auto aligned_meta_size = vector_transformer->GetMetaSize(align_size_);
        this->code_size_ += aligned_meta_size;
        this->query_code_size_ += aligned_meta_size;
    }
}

template <typename QuantTmpl, MetricType metric>
VectorTransformerPtr
TransformQuantizer<QuantTmpl, metric>::MakeTransformerInstance(
    std::string transform_str, const VectorTransformerParameter& param) const {
    uint32_t input_dim = param.input_dim_;
    uint32_t output_dim = input_dim;

    if (transform_str == TRANSFORMER_TYPE_VALUE_PCA) {
        if (param.pca_dim_ != 0) {
            output_dim = param.pca_dim_;
        }
        return std::make_shared<PCATransformer>(this->allocator_, input_dim, output_dim);
    }

    if (transform_str == TRANSFORMER_TYPE_VALUE_FHT) {
        return std::make_shared<FhtKacRotator>(this->allocator_, input_dim);
    }

    if (transform_str == TRANSFORMER_TYPE_VALUE_ROM) {
        return std::make_shared<RandomOrthogonalMatrix>(this->allocator_, input_dim, output_dim);
    }

    if (transform_str == TRANSFORMER_TYPE_VALUE_MRLE) {
        if (param.mrle_dim_ != 0) {
            output_dim = param.mrle_dim_;
            if (output_dim > input_dim) {
                throw VsagException(ErrorType::INVALID_ARGUMENT,
                                    fmt::format("mrle dim must be less than input dim"));
            }
        }
        return std::make_shared<MRLETransformer<metric>>(this->allocator_, input_dim, output_dim);
    }

    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("invalid transformer name {}", transform_str));
};

template <typename QuantTmpl, MetricType metric>
bool
TransformQuantizer<QuantTmpl, metric>::TrainImpl(const float* data, uint64_t count) {
    // 1. train transformer based on original data
    for (const auto& vector_transformer : this->transform_chain_) {
        vector_transformer->Train(data, count);
    }

    // 2. execute transform on original data
    const uint64_t transformed_dim = this->GetTransformedDim();
    Vector<float> transformed_data(transformed_dim * count, 0, this->allocator_);
    Vector<float> primary_scratch(this->allocator_);
    Vector<float> secondary_scratch(this->allocator_);
    for (uint64_t i = 0; i < count; ++i) {
        const auto* transformed = ExecuteChainTransform(
            data + i * this->dim_, nullptr, nullptr, primary_scratch, secondary_scratch);
        memcpy(transformed_data.data() + i * transformed_dim,
               transformed,
               transformed_dim * sizeof(float));
    }

    // 3. train quantizer based on transformed data
    this->is_trained_ = quantizer_->Train(transformed_data.data(), count);
    return this->is_trained_;
}

template <typename QuantTmpl, MetricType metric>
const float*
TransformQuantizer<QuantTmpl, metric>::ExecuteChainTransform(
    const float* input,
    const uint32_t* meta_offsets,
    uint8_t* codes,
    Vector<float>& primary_scratch,
    Vector<float>& secondary_scratch) const {
    if (primary_scratch.size() < this->dim_) {
        primary_scratch.resize(this->dim_, 0.0F);
    }

    const float* current = input;
    float* next = primary_scratch.data();
    for (uint32_t i = 0; i < this->transform_chain_.size(); i++) {
        const auto& vector_transformer = this->transform_chain_[i];
        uint8_t* meta = nullptr;
        if (codes != nullptr && meta_offsets != nullptr && vector_transformer->GetMetaSize() > 0) {
            meta = codes + meta_offsets[i];
        }
        vector_transformer->Transform(current, next, meta);
        current = next;

        if (i + 1 < this->transform_chain_.size()) {
            if (next == primary_scratch.data()) {
                if (secondary_scratch.size() < this->dim_) {
                    secondary_scratch.resize(this->dim_, 0.0F);
                }
                next = secondary_scratch.data();
            } else {
                next = primary_scratch.data();
            }
        }
    }
    return current;
}

template <typename QuantTmpl, MetricType metric>
void
TransformQuantizer<QuantTmpl, metric>::TransformBaseVector(const float* input,
                                                           Vector<float>& output) const {
    Vector<float> secondary_scratch(this->allocator_);
    const auto* transformed =
        ExecuteChainTransform(input, nullptr, nullptr, output, secondary_scratch);
    if (transformed == output.data()) {
        output.resize(this->GetTransformedDim());
    } else {
        output.assign(transformed, transformed + this->GetTransformedDim());
    }
}

template <typename QuantTmpl, MetricType metric>
bool
TransformQuantizer<QuantTmpl, metric>::EncodeOneWithScratch(
    const float* data,
    uint8_t* codes,
    Vector<float>& primary_scratch,
    Vector<float>& secondary_scratch) const {
    const auto* transformed = ExecuteChainTransform(
        data, base_meta_offsets_.data(), codes, primary_scratch, secondary_scratch);
    return quantizer_->EncodeOne(transformed, codes);
}

template <typename QuantTmpl, MetricType metric>
bool
TransformQuantizer<QuantTmpl, metric>::EncodeOneImpl(const float* data, uint8_t* codes) const {
    Vector<float> primary_scratch(this->allocator_);
    Vector<float> secondary_scratch(this->allocator_);
    return EncodeOneWithScratch(data, codes, primary_scratch, secondary_scratch);
}

template <typename QuantTmpl, MetricType metric>
void
TransformQuantizer<QuantTmpl, metric>::ProcessQueryImpl(
    const float* query, Computer<TransformQuantizer>& computer) const {
    // 0. allocate
    try {
        if (computer.inner_computer_->buf_ == nullptr) {
            computer.inner_computer_->buf_ =
                reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
        }
    } catch (const std::bad_alloc& e) {
        computer.inner_computer_->buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }

    // 1. execute transform
    const auto* transformed = ExecuteChainTransform(query,
                                                    query_meta_offsets_.data(),
                                                    computer.inner_computer_->buf_,
                                                    computer.primary_scratch_,
                                                    computer.secondary_scratch_);

    // 2. execute quantize
    // note that only when computer.inner_computer_->buf_ == nullptr, quantizer_ will allocate its query buffer
    quantizer_->ProcessQuery(transformed, *computer.inner_computer_);
};

template <typename QuantTmpl, MetricType metric>
float
TransformQuantizer<QuantTmpl, metric>::ExecuteChainDistanceRecovery(float quantize_dist,
                                                                    const uint32_t* meta_offsets_1,
                                                                    const uint32_t* meta_offsets_2,
                                                                    const uint8_t* codes_1,
                                                                    const uint8_t* codes_2) const {
    auto dist = quantize_dist;
    for (uint32_t i = 0; i < this->transform_chain_.size(); i++) {
        const auto& vector_transformer = this->transform_chain_[i];
        const auto* meta_1 = codes_1 + meta_offsets_1[i];
        const auto* meta_2 = codes_2 + meta_offsets_2[i];

        dist = vector_transformer->RecoveryDistance(dist, meta_1, meta_2);
    }

    return dist;
}

template <typename QuantTmpl, MetricType metric>
void
TransformQuantizer<QuantTmpl, metric>::ComputeDistImpl(Computer<TransformQuantizer>& computer,
                                                       const uint8_t* codes,
                                                       float* dists) const {
    const auto* meta_offset_1 = query_meta_offsets_.data();
    const auto* meta_offset_2 = base_meta_offsets_.data();

    const auto* codes_1 = computer.inner_computer_->buf_;
    const auto* codes_2 = codes;

    auto quantize_dist = quantizer_->ComputeDist(*(computer.inner_computer_), codes);
    dists[0] =
        ExecuteChainDistanceRecovery(quantize_dist, meta_offset_1, meta_offset_2, codes_1, codes_2);
};

template <typename QuantTmpl, MetricType metric>
float
TransformQuantizer<QuantTmpl, metric>::ComputeImpl(const uint8_t* codes1,
                                                   const uint8_t* codes2) const {
    const auto* meta_offset = base_meta_offsets_.data();

    auto quantize_dist = quantizer_->Compute(codes1, codes2);
    auto dist =
        ExecuteChainDistanceRecovery(quantize_dist, meta_offset, meta_offset, codes1, codes2);

    return dist;
}

template <typename QuantTmpl, MetricType metric>
void
TransformQuantizer<QuantTmpl, metric>::ScanBatchDistImpl(
    Computer<TransformQuantizer<QuantTmpl, metric>>& computer,
    uint64_t count,
    const uint8_t* codes,
    float* dists) const {
    for (uint64_t i = 0; i < count; ++i) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <typename QuantTmpl, MetricType metric>
void
TransformQuantizer<QuantTmpl, metric>::ReleaseComputerImpl(
    Computer<TransformQuantizer<QuantTmpl, metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <typename QuantTmpl, MetricType metric>
bool
TransformQuantizer<QuantTmpl, metric>::EncodeBatchImpl(const float* data,
                                                       uint8_t* codes,
                                                       uint64_t count) const {
    Vector<float> primary_scratch(this->allocator_);
    Vector<float> secondary_scratch(this->allocator_);
    for (uint64_t i = 0; i < count; ++i) {
        if (not EncodeOneWithScratch(data + i * this->dim_,
                                     codes + i * this->code_size_,
                                     primary_scratch,
                                     secondary_scratch)) {
            return false;
        }
    }
    return true;
}

template <typename QuantTmpl, MetricType metric>
void
TransformQuantizer<QuantTmpl, metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteVector(writer, this->base_meta_offsets_);
    StreamWriter::WriteVector(writer, this->query_meta_offsets_);
    StreamWriter::WriteObj(writer, this->align_size_);

    for (const auto& transformer : this->transform_chain_) {
        transformer->Serialize(writer);
    }

    this->quantizer_->Serialize(writer);
}

template <typename QuantTmpl, MetricType metric>
void
TransformQuantizer<QuantTmpl, metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->base_meta_offsets_);
    StreamReader::ReadVector(reader, this->query_meta_offsets_);
    StreamReader::ReadObj(reader, this->align_size_);

    for (const auto& transformer : this->transform_chain_) {
        transformer->Deserialize(reader);
    }

    this->quantizer_->Deserialize(reader);
}

}  // namespace vsag
