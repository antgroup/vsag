
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

#include "comparison_executor.h"

#include "impl/bitset/fast_bitset.h"
#include "vsag_exception.h"

namespace vsag {

ComparisonExecutor::ComparisonExecutor(Allocator* allocator,
                                       const ExprPtr& expr,
                                       const AttrInvertedInterfacePtr& attr_index)
    : Executor(allocator, expr, attr_index) {
    auto comp_expr = std::dynamic_pointer_cast<const ComparisonExpression>(expr);
    if (comp_expr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
    }
    this->op_ = comp_expr->op;
    auto field_expr = std::dynamic_pointer_cast<const FieldExpression>(comp_expr->left);
    if (field_expr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
    }

    this->field_name_ = field_expr->fieldName;
    auto value_type = this->attr_index_->GetTypeOfField(this->field_name_);
    if (value_type == AttrValueType::STRING) {
        auto constant = std::dynamic_pointer_cast<const StringConstant>(comp_expr->right);
        if (constant == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
        }
        auto attr_value = std::make_shared<AttributeValue<std::string>>();
        attr_value->GetValue().emplace_back(constant->value);
        this->filter_attribute_ = attr_value;
    } else {
        auto constant = std::dynamic_pointer_cast<const NumericConstant>(comp_expr->right);
        if (constant == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
        }
        if (value_type == AttrValueType::INT8) {
            auto attr_value = std::make_shared<AttributeValue<int8_t>>();
            attr_value->GetValue().emplace_back(GetNumericValue<int8_t>(constant->value));
            this->filter_attribute_ = attr_value;
        } else if (value_type == AttrValueType::INT16) {
            auto attr_value = std::make_shared<AttributeValue<int16_t>>();
            attr_value->GetValue().emplace_back(GetNumericValue<int16_t>(constant->value));
            this->filter_attribute_ = attr_value;
        } else if (value_type == AttrValueType::INT32) {
            auto attr_value = std::make_shared<AttributeValue<int32_t>>();
            attr_value->GetValue().emplace_back(GetNumericValue<int32_t>(constant->value));
            this->filter_attribute_ = attr_value;
        } else if (value_type == AttrValueType::INT64) {
            auto attr_value = std::make_shared<AttributeValue<int64_t>>();
            attr_value->GetValue().emplace_back(GetNumericValue<int64_t>(constant->value));
            this->filter_attribute_ = attr_value;
        } else if (value_type == AttrValueType::UINT8) {
            auto attr_value = std::make_shared<AttributeValue<uint8_t>>();
            attr_value->GetValue().emplace_back(GetNumericValue<uint8_t>(constant->value));
            this->filter_attribute_ = attr_value;
        } else if (value_type == AttrValueType::UINT16) {
            auto attr_value = std::make_shared<AttributeValue<uint16_t>>();
            attr_value->GetValue().emplace_back(GetNumericValue<uint16_t>(constant->value));
            this->filter_attribute_ = attr_value;
        } else if (value_type == AttrValueType::UINT32) {
            auto attr_value = std::make_shared<AttributeValue<uint32_t>>();
            attr_value->GetValue().emplace_back(GetNumericValue<uint32_t>(constant->value));
            this->filter_attribute_ = attr_value;
        } else if (value_type == AttrValueType::UINT64) {
            auto attr_value = std::make_shared<AttributeValue<uint64_t>>();
            attr_value->GetValue().emplace_back(GetNumericValue<uint64_t>(constant->value));
            this->filter_attribute_ = attr_value;
        } else {
            throw VsagException(ErrorType::INTERNAL_ERROR, "unsupported attribute type");
        }
    }
    this->filter_attribute_->name_ = this->field_name_;
}

void
ComparisonExecutor::Clear() {
    Executor::Clear();
}

Filter*
ComparisonExecutor::Run(BucketIdType bucket_id) {
    if (this->op_ != ComparisonOperator::EQ and this->op_ != ComparisonOperator::NE) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "unsupported comparison operator");
    }

    for (const auto* manager : managers_) {
        if (manager == nullptr) {
            continue;
        }
        auto* bitset = manager->GetOneBitset(bucket_id);
        this->bitset_->Or(bitset);
    }
    if (this->op_ == ComparisonOperator::EQ) {
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
ComparisonExecutor::Init() {
    if (this->bitset_ == nullptr) {
        this->bitset_ = ComputableBitset::MakeRawInstance(this->bitset_type_, this->allocator_);
        this->own_bitset_ = true;
    }
    this->managers_ = this->attr_index_->GetBitsetsByAttr(*this->filter_attribute_);
}

}  // namespace vsag
