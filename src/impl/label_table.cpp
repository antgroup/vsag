
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

class RemoveListFilter : public Filter {
public:
    explicit RemoveListFilter(const UnorderedSet<InnerIdType>& remove_ids,
                              std::shared_mutex& delete_ids_mutex)
        : Filter(), remove_ids_(remove_ids), delete_ids_mutex_(delete_ids_mutex){};

    [[nodiscard]] bool
    CheckValid(int64_t inner_id) const override {
        std::shared_lock rlock(delete_ids_mutex_);
        return remove_ids_.count(inner_id) == 0;
    }

private:
    const UnorderedSet<InnerIdType>& remove_ids_;
    std::shared_mutex& delete_ids_mutex_;
};

LabelTable::LabelTable(Allocator* allocator, bool use_reverse_map, bool compress_redundant_data)
    : allocator_(allocator),
      label_table_(0, allocator),
      label_remap_(0, allocator),
      use_reverse_map_(use_reverse_map),
      deleted_ids_(allocator),
      compress_duplicate_data_(compress_redundant_data),
      duplicate_ids_(0, allocator) {
    deleted_ids_filter_ = std::make_shared<RemoveListFilter>(deleted_ids_, delete_ids_mutex_);
}

bool
LabelTable::CheckLabel(LabelType label) const {
    bool is_exist = false;
    InnerIdType inner_id;
    if (use_reverse_map_) {
        auto iter = label_remap_.find(label);
        is_exist = iter != label_remap_.end();
        if (not is_exist) {
            return false;
        }
        inner_id = iter->second;
    } else {
        auto result = std::find(label_table_.begin(), label_table_.end(), label);
        is_exist = (result != label_table_.end());
        if (not is_exist) {
            return false;
        }
        inner_id = result - label_table_.begin();
    }
    {
        std::shared_lock rlock(this->delete_ids_mutex_);
        if (this->deleted_ids_.count(inner_id) > 0) {
            return false;
        }
    }
    return true;
}

InnerIdType
LabelTable::get_id_by_label_with_reverse_map(LabelType label) const noexcept {
    const auto iter = this->label_remap_.find(label);
    if (iter == this->label_remap_.end()) {
        return INVALID_ID;
    }
    return iter->second;
}

InnerIdType
LabelTable::get_id_by_label_with_label_table(LabelType label) const noexcept {
    auto result = std::find(label_table_.begin(), label_table_.end(), label);
    if (result == label_table_.end()) {
        return INVALID_ID;
    }
    return result - label_table_.begin();
}

InnerIdType
LabelTable::GetIdByLabel(LabelType label, bool return_even_removed) const {
    InnerIdType id;
    if (use_reverse_map_) {
        id = this->get_id_by_label_with_reverse_map(label);
    } else {
        id = this->get_id_by_label_with_label_table(label);
    }
    if (id == INVALID_ID) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("label {} does not exist", label));
    }
    if (not return_even_removed) {
        std::shared_lock rlock(delete_ids_mutex_);
        if (this->deleted_ids_.count(id) > 0) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("label {} is removed", label));
        }
    }
    return id;
}

uint32_t
LabelTable::MarkRemove(const std::vector<LabelType>& labels) {
    uint32_t init_delete_size;
    {
        std::shared_lock rlock(delete_ids_mutex_);
        init_delete_size = this->deleted_ids_.size();
    }
    for (const auto& label : labels) {
        InnerIdType id;
        if (this->use_reverse_map_) {
            id = this->get_id_by_label_with_reverse_map(label);
        } else {
            id = this->get_id_by_label_with_label_table(label);
        }
        if (id == INVALID_ID) {
            continue;
        }
        std::scoped_lock wlock(delete_ids_mutex_);
        this->deleted_ids_.insert(id);
    }
    std::shared_lock rlock(delete_ids_mutex_);
    return this->deleted_ids_.size() - init_delete_size;
}

void
LabelTable::MergeOther(const LabelTablePtr& other, const IdMapFunction& id_map) {
    auto other_size = other->GetTotalCount();
    this->label_table_.resize(total_count_ + other_size);
    for (int64_t i = 0; i < other_size; ++i) {
        auto new_label = std::get<1>(id_map(other->label_table_[i]));
        this->label_table_[i + total_count_] = new_label;
        this->label_remap_[new_label] = i + total_count_;
    }
    total_count_ += other_size;
}
}  // namespace vsag
