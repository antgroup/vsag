
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

#include "label_table.h"

namespace vsag {

void
LabelTable::MergeOther(const LabelTablePtr& other, InnerIdType bias) {
    this->label_table_.reserve(this->label_table_.size() + other->label_table_.size());
    std::copy(other->label_table_.begin(),
              other->label_table_.end(),
              std::back_inserter(this->label_table_));
    for (auto& [label, id] : other->label_remap_) {
        this->label_remap_[label] = id + bias;
    }
}
}  // namespace vsag
