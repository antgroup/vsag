
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

#include "transform_quantizer.h"

#include "impl/transform/transformer_headers.h"
#include "typing.h"
#include "utils/util_functions.h"

namespace vsag {

template <MetricType metric>
TransformQuantizer<metric>::TransformQuantizer(const QuantizerParamPtr& param,
                                               const IndexCommonParam& common_param)
    : TransformQuantizer<metric>(std::dynamic_pointer_cast<TransformQuantizerParameter>(param),
                                 common_param) {
}

template <MetricType metric>
TransformQuantizer<metric>::TransformQuantizer(const TransformQuantizerParamPtr& param,
                                               const IndexCommonParam& common_param)
    : Quantizer<TransformQuantizer<metric>>(common_param.dim_, common_param.allocator_.get()),
      base_meta_offsets_(common_param.allocator_.get()),
      query_meta_offsets_(common_param.allocator_.get()) {
    // 1. init quantizer
    auto detailed_quantizer_param =
        QuantizerParameter::GetQuantizerParameterByJson(param->base_quantizer_json_);
    this->quantizer_ = this->MakeInstance(detailed_quantizer_param, common_param);

    // 2. init transform chain
    VectorTransformerParameter transformer_param;
    transformer_param.FromJson(param->base_quantizer_json_);
    transformer_param.input_dim_ = this->dim_;
    for (const auto& transform_str : param->tq_chain_) {
        transform_chain_.emplace_back(MakeTransformerInstance(transform_str, transformer_param));
        transformer_param.input_dim_ = transform_chain_.back()->GetOutputDim();
    }

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

template <MetricType metric>
VectorTransformerPtr
TransformQuantizer<metric>::MakeTransformerInstance(std::string transform_str,
                                                    const VectorTransformerParameter& param) const {
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

    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("invalid transformer name {}", transform_str));
};

template <MetricType metric>
bool
TransformQuantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    // 1. train transformer based on original data
    for (const auto& vector_transformer : this->transform_chain_) {
        vector_transformer->Train(data, count);
    }

    // 2. execute transform on original data
    Vector<DataType> transformed_data(this->dim_ * count, 0, this->allocator_);
    Vector<uint8_t> tmp_codes(this->code_size_, 0, this->allocator_);
    transformed_data.assign(data, data + count * this->dim_);
    for (auto i = 0; i < count; i++) {
        ExecuteChainTransform(
            transformed_data.data() + i * this->dim_, base_meta_offsets_.data(), tmp_codes.data());
    }

    // 3. train quantizer based on transformed data
    return quantizer_->Train(transformed_data.data(), count);
}

template <MetricType metric>
void
TransformQuantizer<metric>::ExecuteChainTransform(DataType* prev_data,
                                                  const uint32_t* meta_offsets,
                                                  uint8_t* codes) const {
    Vector<DataType> next_data(this->dim_, 0, this->allocator_);

    for (uint32_t i = 0; i < this->transform_chain_.size(); i++) {
        auto vector_transformer = this->transform_chain_[i];
        auto meta_offset = meta_offsets[i];

        auto meta = vector_transformer->Transform(prev_data, next_data.data());
        meta->EncodeMeta(codes + meta_offset);

        memcpy(prev_data, next_data.data(), this->dim_ * sizeof(DataType));
    }
}

template <MetricType metric>
bool
TransformQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    // 1. execute transform
    Vector<DataType> data_buffer(this->code_size_, 0, this->allocator_);
    data_buffer.assign(data, data + this->dim_);
    ExecuteChainTransform(data_buffer.data(), base_meta_offsets_.data(), codes);

    // 2. execute quantize
    return quantizer_->EncodeOne(data_buffer.data(), codes);
};

template <MetricType metric>
void
TransformQuantizer<metric>::ProcessQueryImpl(const vsag::DataType* query,
                                             Computer<TransformQuantizer>& computer) const {
    // 0. allocate
    try {
        computer.buf_ =
            reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
    } catch (const std::bad_alloc& e) {
        computer.buf_ = nullptr;
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "bad alloc when init computer buf");
    }

    // 1. execute transform
    Vector<DataType> data_buffer(this->code_size_, 0, this->allocator_);
    data_buffer.assign(query, query + this->dim_);
    ExecuteChainTransform(data_buffer.data(), query_meta_offsets_.data(), computer.buf_);

    // 2. execute quantize
    // note that only when computer.buf_ == nullptr, quantizer_ will allocate data to buf_
    quantizer_->ProcessQuery(query, computer.inner_computer_);
};

template <MetricType metric>
float
TransformQuantizer<metric>::ExecuteChainDistanceRecovery(float quantize_dist,
                                                         const uint32_t* meta_offsets_1,
                                                         const uint32_t* meta_offsets_2,
                                                         const uint8_t* codes_1,
                                                         const uint8_t* codes_2) const {
    auto dist = quantize_dist;
    for (uint32_t i = 0; i < this->transform_chain_.size(); i++) {
        auto vector_transformer = this->transform_chain_[i];
        const auto* meta_1 = codes_1 + meta_offsets_1[i];
        const auto* meta_2 = codes_2 + meta_offsets_2[i];

        dist = vector_transformer->RecoveryDistance(dist, meta_1, meta_2);
    }

    return dist;
}

template <MetricType metric>
void
TransformQuantizer<metric>::ComputeDistImpl(Computer<TransformQuantizer>& computer,
                                            const uint8_t* codes,
                                            float* dists) const {
    const auto* meta_offset_1 = query_meta_offsets_.data();
    const auto* meta_offset_2 = base_meta_offsets_.data();

    const auto* codes_1 = computer.buf_;
    const auto* codes_2 = codes;

    auto quantize_dist = quantizer_->ComputeDist(computer.inner_computer_, codes);
    dists[0] =
        ExecuteChainDistanceRecovery(quantize_dist, meta_offset_1, meta_offset_2, codes_1, codes_2);
};

template <MetricType metric>
float
TransformQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    const auto* meta_offset = base_meta_offsets_.data();

    auto quantize_dist = quantizer_->Compute(codes1, codes2);
    auto dist =
        ExecuteChainDistanceRecovery(quantize_dist, meta_offset, meta_offset, codes1, codes2);

    return dist;
}

template <MetricType metric>
void
TransformQuantizer<metric>::ScanBatchDistImpl(Computer<TransformQuantizer<metric>>& computer,
                                              uint64_t count,
                                              const uint8_t* codes,
                                              float* dists) const {
    for (uint64_t i = 0; i < count; ++i) {
        this->ComputeDistImpl(computer, codes + i * this->code_size_, dists + i);
    }
}

template <MetricType metric>
void
TransformQuantizer<metric>::ReleaseComputerImpl(
    Computer<TransformQuantizer<metric>>& computer) const {
    this->allocator_->Deallocate(computer.buf_);
}

template <MetricType metric>
bool
TransformQuantizer<metric>::EncodeBatchImpl(const DataType* data,
                                            uint8_t* codes,
                                            uint64_t count) const {
    for (uint64_t i = 0; i < count; ++i) {
        EncodeOneImpl(data + i * this->dim_, codes + i * this->code_size_);
    }
    return true;
}

TEMPLATE_QUANTIZER(TransformQuantizer)

}  // namespace vsag
