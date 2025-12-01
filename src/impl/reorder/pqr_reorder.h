
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

#include <memory>

#include "datacell/flatten_interface.h"
#include "impl/heap/distance_heap.h"
#include "impl/reorder/reorder.h"
#include "pqr_reorder_parameter.h"
#include "quantization/product_quantization/product_quantizer_parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
class PqrReorder : public ReorderInterface {
public:
    PqrReorder(const FlattenInterfacePtr& flatten,
               const IndexCommonParam& common_param,
               const ReorderParameterPtr& reorder_param)
        : flatten_(flatten),
          allocator_(common_param.allocator_.get()),
          bias_(common_param.allocator_.get()) {
        auto inner_common = common_param;
        metric_ = common_param.metric_;
        dim_ = inner_common.dim_;
        inner_common.metric_ = MetricType::METRIC_TYPE_IP;
        auto pqr_reorder_param = std::dynamic_pointer_cast<PqrReorderParameter>(reorder_param);
        if (pqr_reorder_param == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "Invalid reorder parameter type for PqrReorder");
        }
        reorder_code_ =
            FlattenInterface::MakeInstance(pqr_reorder_param->residual_param_, inner_common);
    }

    DistHeapPtr
    Reorder(const DistHeapPtr& input,
            const float* query,
            int64_t topk,
            Allocator* allocator = nullptr) const override;

    void
    InsertVector(const void* vector, vsag::InnerIdType id) override;

    void
    Train(const void* vector, uint64_t count) override;

    void
    Resize(uint64_t new_size) override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Deserialize(StreamReader& reader) override;

private:
    const FlattenInterfacePtr flatten_;
    FlattenInterfacePtr reorder_code_;

    Allocator* allocator_{nullptr};
    Vector<float> bias_;
    int64_t dim_{0};
    MetricType metric_{MetricType::METRIC_TYPE_IP};
};
}  // namespace vsag
