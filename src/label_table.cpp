
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
LabelTable::MergeOther(const LabelTablePtr& other, const IdMapFunction& id_map) {
    auto cur_size = this->label_table_.size();
    auto other_size = other->label_table_.size();
    this->label_table_.resize(this->label_table_.size() + other->label_table_.size());
    for (int64_t i = 0; i < other_size; ++i) {
        auto new_label = std::get<1>(id_map(other->label_table_[i]));
        this->label_table_[i + cur_size] = new_label;
        this->label_remap_[new_label] = i + cur_size;
    }
}
}  // namespace vsag
