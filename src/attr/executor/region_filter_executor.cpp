
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

#include "region_filter_executor.h"

#include "impl/bitset/fast_bitset.h"
#include "vsag_exception.h"

namespace vsag {

const std::string RegionFilterExecutor::region_type_field_name{"region_type"};
const std::string RegionFilterExecutor::region_list_field_name{"region_list"};
const std::string RegionFilterExecutor::residence_list_field_name{"residence_tag_list"};

const AttributeValue<int16_t>
RegionFilterExecutor::init_type_query() {
    AttributeValue<int16_t> attr;
    attr.name_ = region_type_field_name;
    for (int16_t i = 0; i < 6; i++) {
        attr.GetValue().emplace_back(i);
    }
    attr.GetValue().emplace_back(-1);
    return attr;
}

const AttributeValue<int16_t> RegionFilterExecutor::type_query_attr =
    RegionFilterExecutor::init_type_query();

RegionFilterExecutor::RegionFilterExecutor(Allocator* allocator,
                                           const ExprPtr& expr,
                                           const AttrInvertedInterfacePtr& attr_index)
    : Executor(allocator, expr, attr_index) {
    auto region_expr = std::dynamic_pointer_cast<const RegionFilterExpression>(expr);
    if (region_expr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match(region filter)");
    }
    auto region_list_expr =
        std::dynamic_pointer_cast<const IntListConstant<int64_t>>(region_expr->arg0);
    if (region_list_expr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
    }

    auto residence_list_expr =
        std::dynamic_pointer_cast<const IntListConstant<int64_t>>(region_expr->arg1);
    if (residence_list_expr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
    }

    auto trigger_list_expr =
        std::dynamic_pointer_cast<const IntListConstant<int64_t>>(region_expr->arg2);
    if (trigger_list_expr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
    }

    this->region_type_managers_ = attr_index->GetBitsetsByAttr(type_query_attr);

    AttributeValue<int64_t> list_value;
    list_value.name_ = region_list_field_name;
    list_value.GetValue() = region_list_expr->values;
    this->region_list_managers_ = attr_index->GetBitsetsByAttr(list_value);

    list_value.name_ = residence_list_field_name;
    list_value.GetValue() = residence_list_expr->values;
    this->residence_list_managers_ = attr_index->GetBitsetsByAttr(list_value);

    list_value.GetValue() = trigger_list_expr->values;
    this->trigger_list_managers_ = attr_index->GetBitsetsByAttr(list_value);
}

void
RegionFilterExecutor::Clear() {
    Executor::Clear();
    if (this->region_bitset_ != nullptr) {
        this->region_bitset_->Clear();
    }
    if (this->residence_bitset_ != nullptr) {
        this->residence_bitset_->Clear();
    }
    if (this->trigger_bitset_ != nullptr) {
        this->trigger_bitset_->Clear();
    }

    for (auto* bitset : type_bitsets_) {
        if (bitset != nullptr) {
            bitset->Clear();
        }
    }
}

Filter*
RegionFilterExecutor::Run(BucketIdType bucket_id) {
    for (const auto* manager : region_list_managers_) {
        if (manager == nullptr) {
            continue;
        }
        region_bitset_->Or(manager->GetOneBitset(bucket_id));
    }
    for (const auto* manager : residence_list_managers_) {
        if (manager == nullptr) {
            continue;
        }
        residence_bitset_->Or(manager->GetOneBitset(bucket_id));
    }
    for (const auto* manager : trigger_list_managers_) {
        if (manager == nullptr) {
            continue;
        }
        trigger_bitset_->Or(manager->GetOneBitset(bucket_id));
    }

    // type = -1;
    ComputableBitset* bitset = nullptr;
    if (region_type_managers_[6] != nullptr) {
        bitset = region_type_managers_[6]->GetOneBitset(bucket_id);
    }
    this->bitset_->Or(bitset);

    // type = 1;
    if (region_type_managers_[1] != nullptr) {
        const auto* ptr = region_type_managers_[1]->GetOneBitset(bucket_id);
        if (ptr != nullptr) {
            type_bitsets_[1]->Or(residence_bitset_);
            type_bitsets_[1]->And(ptr);
        }
    }

    // type = 2;
    if (region_type_managers_[2] != nullptr) {
        const auto* ptr = region_type_managers_[2]->GetOneBitset(bucket_id);
        if (ptr != nullptr) {
            type_bitsets_[2]->Or(region_bitset_);
            type_bitsets_[2]->And(ptr);
        }
    }
    // type = 3;
    if (region_type_managers_[3] != nullptr) {
        const auto* ptr = region_type_managers_[3]->GetOneBitset(bucket_id);
        if (ptr != nullptr) {
            type_bitsets_[3]->Or(region_bitset_);
            type_bitsets_[3]->Or(residence_bitset_);
            type_bitsets_[3]->And(ptr);
        }
    }

    // type = 4;
    if (region_type_managers_[4] != nullptr) {
        const auto* ptr = region_type_managers_[4]->GetOneBitset(bucket_id);
        if (ptr != nullptr) {
            type_bitsets_[4]->Or(region_bitset_);
            type_bitsets_[4]->And(residence_bitset_);
            type_bitsets_[4]->And(ptr);
        }
    }

    // type = 5;
    if (region_type_managers_[5] != nullptr) {
        const auto* ptr = region_type_managers_[5]->GetOneBitset(bucket_id);
        if (ptr != nullptr) {
            type_bitsets_[5]->Or(residence_bitset_);
            type_bitsets_[5]->Not();
            this->only_bitset_ = false;
            type_bitsets_[5]->And(region_bitset_);
            type_bitsets_[5]->And(ptr);
        }
    }

    // type = 0;
    if (region_type_managers_[0] != nullptr) {
        const auto* ptr = region_type_managers_[0]->GetOneBitset(bucket_id);
        if (ptr != nullptr) {
            type_bitsets_[0]->Or(trigger_bitset_);
            region_bitset_->And(residence_bitset_);
            type_bitsets_[0]->Or(region_bitset_);
            type_bitsets_[0]->And(ptr);
        }
    }

    for (int i = 0; i < 6; ++i) {
        bitset_->Or(type_bitsets_[i]);
    }

    if (bitset_type_ == ComputableBitsetType::FastBitset) {
        this->only_bitset_ = true;
        WhiteListFilter::TryToUpdate(this->filter_, this->bitset_);
    } else {
        // todo bugfix
    }

    return this->filter_;
}

void
RegionFilterExecutor::Init() {
    if (this->bitset_ == nullptr) {
        this->bitset_ = ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_);
        this->own_bitset_ = true;
    }
    if (this->region_bitset_ == nullptr) {
        this->region_bitset_ =
            ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_);
    }
    if (this->residence_bitset_ == nullptr) {
        this->residence_bitset_ =
            ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_);
    }
    if (this->trigger_bitset_ == nullptr) {
        this->trigger_bitset_ =
            ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_);
    }

    for (auto& bitset : type_bitsets_) {
        if (bitset == nullptr) {
            bitset = ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_);
        }
    }
}

}  // namespace vsag
