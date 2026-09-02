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

#include <mutex>
#include <vector>

#include "ivf.h"  // IWYU pragma: keep

namespace vsag {

uint32_t
IVF::Remove(const std::vector<int64_t>& ids, RemoveMode mode) {
    uint32_t delete_count = 0;
    if (mode == RemoveMode::MARK_REMOVE) {
        std::scoped_lock label_lock(this->label_lookup_mutex_);
        delete_count = this->label_table_->MarkRemove(ids);
        delete_count_ += delete_count;
    }
    return delete_count;
}

void
IVF::UpdateAttribute(int64_t id, const AttributeSet& new_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    auto [bucket_id, offset_id] = this->get_location(inner_id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, offset_id, bucket_id);
}

void
IVF::UpdateAttribute(int64_t id, const AttributeSet& new_attrs, const AttributeSet& origin_attrs) {
    auto inner_id = this->label_table_->GetIdByLabel(id);
    auto [bucket_id, offset_id] = this->get_location(inner_id);
    this->attr_filter_index_->UpdateBitsetsByAttr(new_attrs, offset_id, bucket_id, origin_attrs);
}

}  // namespace vsag
