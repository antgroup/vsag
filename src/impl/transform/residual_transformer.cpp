
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

#include "residual_transformer.h"

#include "algorithm/ivf_partition/ivf_partition_strategy.h"
#include "simd/fp32_simd.h"

namespace vsag {

ResidualTransformer::ResidualTransformer(const IndexCommonParam& common_param,
                                         int64_t input_dim,
                                         int64_t output_dim,
                                         uint32_t centroids_count,
                                         FlattenInterfacePtr outside_centroids_storage)
    : VectorTransformer(common_param.allocator_.get(), input_dim, output_dim),
      centroids_count_(centroids_count),
      outside_centroids_storage_(outside_centroids_storage) {
    if (centroids_count_ != 0) {
        // partition related param
        JsonType json;
        json[IVF_PARTITION_STRATEGY_TYPE_KEY] == IVF_PARTITION_STRATEGY_TYPE_NEAREST;
        json[IVF_TRAIN_TYPE_KEY] == IVF_TRAIN_TYPE_KMEANS;
        auto ivf_partition_strategy_parameter = std::make_shared<IVFPartitionStrategyParameters>();
        ivf_partition_strategy_parameter->FromJson(json);

        // partition related common param
        IndexCommonParam inner_common_param = common_param;
        inner_common_param.dim_ = input_dim;

        this->partition_strategy_ = std::make_shared<IVFNearestPartition>(
            centroids_count, inner_common_param, ivf_partition_strategy_parameter);
    }

    this->type_ = VectorTransformerType::RESIDUAL;
}

void
ResidualTransformer::Train(const float* data, uint64_t count) {
    auto dataset = Dataset::Make();
    dataset->Float32Vectors(data)->NumElements(count)->Owner(false);
    this->partition_strategy_->Train(dataset);
}

TransformerMetaPtr
ResidualTransformer::Transform(const float* input_vec,
                               float* output_vec,
                               const InnerTransformParamPtr param) const {
    auto meta = std::make_shared<ResidualMeta>();

    // 1. get centroid
    Vector<float> centroid_vec(this->input_dim_, 0, allocator_);
    uint32_t centroid_id = 0;
    if (param and param->target_centroid != -1) {
        centroid_id = param->target_centroid;
        outside_centroids_storage_->GetCodesById(centroid_id, (uint8_t*)centroid_vec.data());
    } else {
        centroid_id = this->partition_strategy_->ClassifyDatas(input_vec, 1, 1)[0];
        this->partition_strategy_->GetCentroid(centroid_id, centroid_vec);
    }

    // 2. compute residual part and norm
    for (int i = 0; i < this->input_dim_; i++) {
        output_vec[i] = input_vec[i] - centroid_vec[i];
    }
    meta->residual_norm = FP32ComputeIP(output_vec, output_vec, input_dim_);
    meta->cross_norm = FP32ComputeIP(output_vec, (const float*)(centroid_vec.data()), input_dim_);

    if (not param->is_base) {
        // recovery transform since q is not relevant to c
        for (int i = 0; i < this->input_dim_; i++) {
            output_vec[i] = input_vec[i];
        }
    }

    return meta;
}

void
ResidualTransformer::InverseTransform(const float* input_vec,
                                      float* output_vec,
                                      const InnerTransformParamPtr param) const {
    if (not param->is_base) {
        for (int i = 0; i < this->input_dim_; i++) {
            output_vec[i] = input_vec[i];
        }
        return;
    }

    // 1. get centroid
    Vector<float> centroid_vec(this->input_dim_, 0, allocator_);
    uint32_t centroid_id = 0;
    if (param != nullptr) {
        centroid_id = param->target_centroid;
    } else {
        centroid_id = this->partition_strategy_->ClassifyDatas(input_vec, 1, 1)[0];
    }
    this->partition_strategy_->GetCentroid(centroid_id, centroid_vec);

    // 2. compute residual part and norm
    for (int i = 0; i < this->input_dim_; i++) {
        output_vec[i] = input_vec[i] + centroid_vec[i];
    }
}

float
ResidualTransformer::RecoveryDistance(float dist,
                                      const uint8_t* meta1,
                                      const uint8_t* meta2) const {
    // meta1: base; meta2: query
    float n1 = *((float*)meta2);  // input meta2 as (q - c)^2

    auto meta = std::make_shared<ResidualMeta>();
    meta->DecodeMeta(meta1, 0);
    float n2 = meta->residual_norm;
    float n3 = meta->cross_norm;
    return n1 + n2 + n3 - 2 * dist;
}

}  // namespace vsag