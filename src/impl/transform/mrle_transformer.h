
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

#include "metric_type.h"
#include "simd/normalize.h"
#include "vector_transformer.h"
#include "vsag_exception.h"

namespace vsag {
struct MRLETMeta : public TransformerMeta {};

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class MRLETransformer : public VectorTransformer {
public:
    explicit MRLETransformer(Allocator* allocator, int64_t input_dim, int64_t output_dim)
        : VectorTransformer(allocator, input_dim, output_dim) {
        this->type_ = VectorTransformerType::MRLE;
    }

    virtual ~MRLETransformer() override = default;

    TransformerMetaPtr
    Transform(const float* original_vec, float* transformed_vec) const override {
        auto meta = std::make_shared<MRLETMeta>();
        memcpy(transformed_vec, original_vec, this->output_dim_ * sizeof(float));
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            Normalize(transformed_vec, transformed_vec, this->output_dim_);
        }
        return meta;
    }

    void
    InverseTransform(const float* transformed_vec, float* original_vec) const override {
        throw VsagException(ErrorType::INTERNAL_ERROR, "InverseTransform not implement");
    }

    void
    Serialize(StreamWriter& writer) const override{};

    void
    Deserialize(StreamReader& reader) override{};

    void
    Train(const float* data, uint64_t count) override{};
};

}  // namespace vsag
