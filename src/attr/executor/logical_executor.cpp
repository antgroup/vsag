
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

#include "logical_executor.h"

#include "vsag_exception.h"
namespace vsag {

LogicalExecutor::LogicalExecutor(Allocator* allocator,
                                 const ExprPtr& expr,
                                 const AttrInvertedInterfacePtr& attr_index)
    : Executor(allocator, expr, attr_index) {
    auto logic_expr = std::dynamic_pointer_cast<const LogicalExpression>(expr);
    if (logic_expr == nullptr) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "expression type not match");
    }
    this->left_ = Executor::MakeInstance(this->allocator_, logic_expr->left, attr_index);
    this->right_ = Executor::MakeInstance(this->allocator_, logic_expr->right, attr_index);
    this->op_ = logic_expr->op;
}

void
LogicalExecutor::Clear() {
    this->left_->Clear();
    this->right_->Clear();
    Executor::Clear();
}

Filter*
LogicalExecutor::Run(BucketIdType bucket_id) {
    this->left_->Run(bucket_id);
    this->right_->Run(bucket_id);
    return this->logical_run();
}

Filter*
LogicalExecutor::logical_run() {
    if (this->op_ == LogicalOperator::AND) {
        if (this->left_->only_bitset_ and this->right_->only_bitset_) {
            this->only_bitset_ = true;
            this->bitset_ = this->left_->bitset_;
            this->bitset_->And(this->right_->bitset_);
            WhiteListFilter::TryToUpdate(this->filter_, this->bitset_);
        } else {
            this->only_bitset_ = false;
            auto filter_func = [this](int64_t id) -> bool {
                return this->left_->filter_->CheckValid(id) and
                       this->right_->filter_->CheckValid(id);
            };
            WhiteListFilter::TryToUpdate(this->filter_, filter_func);
        }
    } else if (this->op_ == LogicalOperator::OR) {
        if (this->left_->only_bitset_ and this->right_->only_bitset_) {
            this->only_bitset_ = true;
            this->bitset_ = this->left_->bitset_;
            this->bitset_->Or(this->right_->bitset_);
            WhiteListFilter::TryToUpdate(this->filter_, this->bitset_);
        } else {
            this->only_bitset_ = false;
            auto filter_func = [this](int64_t id) -> bool {
                return this->left_->filter_->CheckValid(id) or
                       this->right_->filter_->CheckValid(id);
            };
            WhiteListFilter::TryToUpdate(this->filter_, filter_func);
        }
    } else {
        // TODO(LHT129): NOT operator implementation
        throw VsagException(ErrorType::INTERNAL_ERROR, "logical operator not supported");
    }
    return this->filter_;
}
void
LogicalExecutor::Init() {
    this->left_->Init();
    this->right_->Init();
}

}  // namespace vsag
