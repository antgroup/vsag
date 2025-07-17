
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

#include "string_list_executor.h"

#include "impl/bitset/fast_bitset.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

StringListExecutor::StringListExecutor(Allocator* allocator,
                                       const ExprPtr& expr,
                                       const AttrInvertedInterfacePtr& attr_index)
    : Executor(allocator, expr, attr_index) {
    auto list_expr = std::dynamic_pointer_cast<const StrListExpression>(expr);
    if (list_expr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
    }
    this->is_not_in_ = list_expr->is_not_in;
    auto field_expr = std::dynamic_pointer_cast<const FieldExpression>(list_expr->field);
    if (field_expr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
    }
    auto list_constant = std::dynamic_pointer_cast<const StrListConstant>(list_expr->values);
    if (list_constant == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
    }
    this->field_name_ = field_expr->fieldName;
    auto attr_value = std::make_shared<AttributeValue<std::string>>();
    copy_vector(list_constant->values, attr_value->GetValue());
    this->filter_attribute_ = attr_value;
    this->filter_attribute_->name_ = this->field_name_;
}

void
StringListExecutor::Clear() {
    Executor::Clear();
}

Filter*
StringListExecutor::Run(BucketIdType bucket_id) {
    for (const auto* manager : managers_) {
        if (manager == nullptr) {
            continue;
        }
        auto* bitset = manager->GetOneBitset(bucket_id);
        this->bitset_->Or(bitset);
    }

    if (not this->is_not_in_) {
        this->only_bitset_ = true;
        WhiteListFilter::TryToUpdate(this->filter_, this->bitset_);
    } else {
        if (bitset_type_ == ComputableBitsetType::FastBitset) {
            this->bitset_->Not();
            this->only_bitset_ = true;
            WhiteListFilter::TryToUpdate(this->filter_, this->bitset_);
        } else {
            this->only_bitset_ = false;
            this->filter_ = new BlackListFilter(this->bitset_);
        }
    }
    return this->filter_;
}

void
StringListExecutor::Init() {
    if (this->bitset_ == nullptr) {
        this->bitset_ = ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_);
        this->own_bitset_ = true;
    }
    this->managers_ = this->attr_index_->GetBitsetsByAttr(*this->filter_attribute_);
}

}  // namespace vsag
