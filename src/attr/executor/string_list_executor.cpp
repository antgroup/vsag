
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

FilterPtr
StringListExecutor::Run() {
    if (this->bitset_ == nullptr) {
        this->bitset_ =
            ComputableBitset::MakeInstance(ComputableBitsetType::SparseBitset, this->allocator_);
    }

    auto bitset_lists = this->attr_index_->GetBitsetsByAttr(*this->filter_attribute_);
    for (auto& bitset : bitset_lists) {
        this->bitset_->Or(bitset);
    }
    if (this->is_not_in_) {
        this->only_bitset_ = false;
        this->filter_ = std::make_shared<BlackListFilter>(this->bitset_);
    } else {
        this->only_bitset_ = true;
        this->filter_ = std::make_shared<WhiteListFilter>(this->bitset_);
    }
    return this->filter_;
}

FilterPtr
StringListExecutor::RunWithBucket(BucketIdType bucket_id) {
    if (this->bitset_ == nullptr) {
        this->bitset_ =
            ComputableBitset::MakeInstance(ComputableBitsetType::FastBitset, this->allocator_);
    }

    auto bitset_lists =
        this->attr_index_->GetBitsetsByAttrAndBucketId(*this->filter_attribute_, bucket_id);
    for (auto& bitset : bitset_lists) {
        this->bitset_->Or(bitset);
    }
    this->only_bitset_ = true;
    if (this->is_not_in_) {
        this->bitset_->Not();
    }
    this->filter_ = std::make_shared<WhiteListFilter>(this->bitset_);
    return this->filter_;
}

}  // namespace vsag
