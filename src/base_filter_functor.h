
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

#include <functional>

#include "bitset_impl.h"
#include "common.h"
#include "data_cell/extra_info_interface.h"
#include "label_table.h"
#include "typing.h"
#include "vsag/filter.h"

namespace vsag {

class UniqueFilter : public Filter {
public:
    UniqueFilter(const std::function<bool(int64_t)>& fallback_func)
        : fallback_func_(fallback_func), is_bitset_filter_(false){};

    UniqueFilter(const BitsetPtr& bitset) : bitset_(bitset), is_bitset_filter_(true){};

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        if (is_bitset_filter_) {
            int64_t bit_index = id & ROW_ID_MASK;
            return not bitset_->Test(bit_index);
        } else {
            return not fallback_func_(id);
        }
    }

private:
    std::function<bool(int64_t)> fallback_func_{nullptr};
    const BitsetPtr bitset_{nullptr};
    const bool is_bitset_filter_{false};
};

class CommonInnerIdFilter : public Filter {
public:
    CommonInnerIdFilter(const FilterPtr filter_impl, const LabelTable& label_table)
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

private:
    const FilterPtr filter_impl_;
    const LabelTable& label_table_;
};

class CommonExtraInfoFilter : public Filter {
public:
    CommonExtraInfoFilter(const FilterPtr filter_impl, const ExtraInfoInterfacePtr& extra_infos)
        : filter_impl_(filter_impl), extra_infos_(extra_infos){};

    [[nodiscard]] bool
    CheckValid(int64_t inner_id) const override {
        bool need_release = false;
        const char* extra_info = extra_infos_->GetExtraInfoById(inner_id, need_release);
        bool valid = filter_impl_->CheckValid(extra_info);
        if (need_release) {
            extra_infos_->Release(extra_info);
        }
        return valid;
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return filter_impl_->ValidRatio();
    }

    [[nodiscard]] Distribution
    FilterDistribution() const override {
        return filter_impl_->FilterDistribution();
    }

private:
    const FilterPtr filter_impl_;
    const ExtraInfoInterfacePtr& extra_infos_;
};

}  // namespace vsag
