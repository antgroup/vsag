
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

#include "impl/label_table/label_table.h"
#include "vsag/filter.h"

namespace vsag {

class InnerIdWrapperFilter : public Filter {
public:
    InnerIdWrapperFilter(const FilterPtr filter_impl, const LabelTable& label_table)
        : filter_impl_(filter_impl), label_table_(label_table){};

    [[nodiscard]] bool
    CheckValid(int64_t inner_id) const override {
        return filter_impl_->CheckValid(label_table_.GetLabelById(inner_id));
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return filter_impl_->ValidRatio();
    }

    [[nodiscard]] Distribution
    FilterDistribution() const override {
        return filter_impl_->FilterDistribution();
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        filter_impl_->GetValidIds(valid_ids, count);
    }

    [[nodiscard]] const uint8_t*
    GetValidBitmap(uint64_t* size) const override {
        // The wrapped filter observes labels, so the bitmap it can offer is indexed by label. It is
        // usable as an inner-id bitmap only when labels and inner ids are the same ids.
        if (not label_table_.IsIdentityMapping()) {
            if (size != nullptr) {
                *size = 0;
            }
            return nullptr;
        }
        return filter_impl_->GetValidBitmap(size);
    }

private:
    const FilterPtr filter_impl_;
    const LabelTable& label_table_;
};
}  // namespace vsag
