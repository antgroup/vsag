
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
#include "attr/expression.h"
#include "data_cell/attribute_inverted_interface.h"
#include "impl/bitset/computable_bitset.h"
#include "impl/filter/filter_headers.h"

namespace vsag {
class Executor;
using ExecutorPtr = std::shared_ptr<Executor>;

class Executor {
public:
    static ExecutorPtr
    MakeInstance(Allocator* allocator,
                 const ExprPtr& expression,
                 const AttrInvertedInterfacePtr& attr_index);

    Executor(Allocator* allocator,
             const ExprPtr& expression,
             const AttrInvertedInterfacePtr& attr_index)
        : expr_(expression),
          attr_index_(attr_index),
          allocator_(allocator),
          bitset_type_(attr_index->GetBitsetType()){};

    virtual ~Executor() {
        if (this->own_bitset_) {
            delete bitset_;
            bitset_ = nullptr;
        }
        delete filter_;
    }

    virtual void
    Clear() {
        if (this->bitset_ != nullptr) {
            this->bitset_->Clear();
        }
    };

    virtual void
    Init(){};

    virtual Filter*
    Run(BucketIdType bucket_id) = 0;

    Filter*
    Run() {
        return this->Run(0);
    }

public:
    bool only_bitset_{true};

    Filter* filter_{nullptr};

    ComputableBitset* bitset_{nullptr};

    ExprPtr expr_{nullptr};

    AttrInvertedInterfacePtr attr_index_{nullptr};

    Allocator* const allocator_{nullptr};

    bool own_bitset_{false};

    ComputableBitsetType bitset_type_{ComputableBitsetType::FastBitset};
};
}  // namespace vsag
