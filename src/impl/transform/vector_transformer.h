

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

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag/allocator.h"

namespace vsag {

class VectorTransformer;
using VectorTransformerPtr = std::shared_ptr<VectorTransformer>;

enum class VectorTransformerType { NONE, PCA, RANDOM_ORTHOGONAL, FHT, RESIDUAL, NORMALIZE };

struct TransformerMeta {
    virtual void
    EncodeMeta(uint8_t* code) {
        return;
    };

    virtual void
    DecodeMeta(uint8_t* code, uint32_t align_size) {
        return;
    };
};

using TransformerMetaPtr = std::shared_ptr<TransformerMeta>;

class VectorTransformer {
public:
    explicit VectorTransformer(Allocator* allocator, int64_t input_dim, int64_t output_dim);

    explicit VectorTransformer(Allocator* allocator, int64_t input_dim)
        : VectorTransformer(allocator, input_dim, input_dim) {
    }

    virtual ~VectorTransformer() = default;

    virtual TransformerMetaPtr
    Transform(const float* input_vec, float* output_vec) const {
        return nullptr;
    };

    virtual void
    Serialize(StreamWriter& writer) const = 0;

    virtual void
    Deserialize(StreamReader& reader) = 0;

    virtual float
    RecoveryDistance(float dist, const uint8_t* meta1, const uint8_t* meta2) const {
        return dist;
    };

    virtual uint32_t
    GetMetaSize() const {
        return 0;
    };  // return original meta size

    virtual uint32_t
    GetMetaSize(uint32_t align_size) const {
        return 0;
    };  // return aligned meta size

    virtual uint32_t
    GetAlignSize() const {
        return 0;
    }

public:
    virtual void
    Train(const float* data, uint64_t count){};

    virtual void
    InverseTransform(const float* input_vec, float* output_vec) const;

    [[nodiscard]] int64_t
    GetInputDim() const {
        return this->input_dim_;
    }

    [[nodiscard]] int64_t
    GetOutputDim() const {
        return this->output_dim_;
    }

    [[nodiscard]] VectorTransformerType
    GetType() const {
        return this->type_;
    }

    [[nodiscard]] uint32_t
    GetExtraCodeSize() const {
        return this->extra_code_size_;
    }

protected:
    uint32_t extra_code_size_{0};  // e.g., sizeof(float)
    int64_t input_dim_{0};
    int64_t output_dim_{0};
    Allocator* const allocator_{nullptr};

    VectorTransformerType type_{VectorTransformerType::NONE};
};

}  // namespace vsag
