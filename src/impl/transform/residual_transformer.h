
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

#include "algorithm/ivf_partition/ivf_nearest_partition.h"
#include "common.h"
#include "vector_transformer.h"

namespace vsag {

struct ResidualMeta : public TransformerMeta {
    float residual_norm;  // (x - c)^2
    float cross_norm;     // (x - c) * c

    void
    EncodeMeta(uint8_t* code) override {
        *((float*)code) = residual_norm;
        *((float*)(code + sizeof(float))) = cross_norm;
    }

    void
    DecodeMeta(const uint8_t* code, uint32_t align_size) override {
        residual_norm = *((float*)code);
        cross_norm = *((float*)(code + align_size));
    }

    static uint32_t
    GetMetaSize() {
        return sizeof(float);
    }

    static uint32_t
    GetMetaSize(uint32_t align_size) {
        return (static_cast<uint32_t>(sizeof(float)) * 2 + align_size - 1) / align_size *
               align_size;
    }

    static uint32_t
    GetAlignSize() {
        return sizeof(float);
    }
};

class ResidualTransformer : public VectorTransformer {
public:
    /*
     * USAGE:
     * 1. when centroids_count == 0, centroids_storage != nullptr: use outside centroids (e.g., QG)
     * 2. when centroids_count != 0, centroids_storage == nullptr: use ivf-partition centroids (e.g., kmeans + Graph)
     * 3. when both is ok, use runtime InnerTransformParamPtr& param in Transform to control
     * 4. when both is default value, use 2. with centroids_count = DEFAULT_CENTROIDS_COUNT
     *
     */
    explicit ResidualTransformer(const IndexCommonParam& common_param,
                                 int64_t input_dim,
                                 int64_t output_dim,
                                 uint32_t centroids_count = 0,
                                 FlattenInterfacePtr outside_centroids_storage = nullptr);

    void
    Train(const float* data, uint64_t count) override;

    TransformerMetaPtr
    Transform(const float* input_vec,
              float* output_vec,
              const InnerTransformParamPtr param = nullptr) const override;

    void
    InverseTransform(const float* input_vec,
                     float* output_vec,
                     const InnerTransformParamPtr param = nullptr) const override;

    virtual void
    Serialize(StreamWriter& writer) const override {
        return;
    };

    virtual void
    Deserialize(StreamReader& reader) override {
        return;
    };

    float
    RecoveryDistance(float dist, const uint8_t* meta1, const uint8_t* meta2) const override;

    uint32_t
    GetMetaSize() const override {
        return ResidualMeta::GetMetaSize();
    }

    uint32_t
    GetMetaSize(uint32_t align_size) const override {
        return ResidualMeta::GetMetaSize(align_size);
    }

    uint32_t
    GetAlignSize() const override {
        return ResidualMeta::GetAlignSize();
    }

public:
    // make public for test

private:
    uint32_t centroids_count_{0};
    FlattenInterfacePtr outside_centroids_storage_{nullptr};  // no need for serialize
    IVFPartitionStrategyPtr partition_strategy_{nullptr};
};

}  // namespace vsag
