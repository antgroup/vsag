
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
        : Filter(), remove_ids_(remove_ids), delete_ids_mutex_(delete_ids_mutex) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t inner_id) const override {
        std::shared_lock rlock(delete_ids_mutex_);
        return remove_ids_.count(inner_id) == 0;
    }

private:
    const UnorderedSet<InnerIdType>& remove_ids_;
    std::shared_mutex& delete_ids_mutex_;
};

LabelTable::LabelRemap::LabelRemap(Allocator* allocator, LabelRemapType remap_type)
    : allocator_(allocator), remap_type_(remap_type) {
    if (remap_type_ == LabelRemapType::ROBIN) {
        robin_map_ = std::make_unique<UnorderedMap<LabelType, InnerIdType>>(0, allocator);
        robin_map_->max_load_factor(0.75F);
    } else {
        pg_map_ = std::make_unique<PGUnorderedMap<LabelType, InnerIdType>>(0, allocator);
        pg_map_->max_load_factor(0.75F);
    }
}

void
LabelTable::LabelRemap::Reset() {
    if (remap_type_ == LabelRemapType::ROBIN) {
        robin_map_ = std::make_unique<UnorderedMap<LabelType, InnerIdType>>(0, allocator_);
        robin_map_->max_load_factor(0.75F);
        return;
    }

    pg_map_ = std::make_unique<PGUnorderedMap<LabelType, InnerIdType>>(0, allocator_);
    pg_map_->max_load_factor(0.75F);
}

void
LabelTable::LabelRemap::Clear() {
    if (pg_map_ != nullptr) {
        pg_map_->clear();
        return;
    }
    robin_map_->clear();
}

void
LabelTable::LabelRemap::Reserve(uint64_t size) {
    if (pg_map_ != nullptr) {
        pg_map_->reserve(size);
        return;
    }
    robin_map_->reserve(size);
}

uint64_t
LabelTable::LabelRemap::Size() const {
    if (pg_map_ != nullptr) {
        return pg_map_->size();
    }
    return robin_map_->size();
}

void
LabelTable::LabelRemap::InsertOrAssign(LabelType label, InnerIdType inner_id) {
    if (pg_map_ != nullptr) {
        (*pg_map_)[label] = inner_id;
        return;
    }
    (*robin_map_)[label] = inner_id;
}

void
LabelTable::LabelRemap::Emplace(LabelType label, InnerIdType inner_id) {
    if (pg_map_ != nullptr) {
        pg_map_->emplace(label, inner_id);
        return;
    }
    robin_map_->emplace(label, inner_id);
}

bool
LabelTable::LabelRemap::Erase(LabelType label) {
    if (pg_map_ != nullptr) {
        return pg_map_->erase(label) > 0;
    }
    return robin_map_->erase(label) > 0;
}

bool
LabelTable::LabelRemap::Find(LabelType label, InnerIdType& inner_id) const {
    if (pg_map_ != nullptr) {
        const auto iter = pg_map_->find(label);
        if (iter == pg_map_->end()) {
            return false;
        }
        inner_id = iter->second;
        return true;
    }

    const auto iter = robin_map_->find(label);
    if (iter == robin_map_->end()) {
        return false;
    }
    inner_id = iter->second;
    return true;
}

LabelTable::LabelTable(Allocator* allocator,
                       bool use_reverse_map,
                       bool compress_redundant_data,
                       LabelRemapType label_remap_type)
    : label_table_(0, allocator),
      use_reverse_map_(use_reverse_map),
      label_remap_(allocator, label_remap_type),
      allocator_(allocator),
      deleted_ids_(allocator),
      source_id_table_(0, allocator),
      hole_list_(0, allocator) {
    (void)compress_redundant_data;
    deleted_ids_filter_ = std::make_shared<RemoveListFilter>(deleted_ids_, delete_ids_mutex_);
}

bool
LabelTable::CheckLabel(LabelType label) const {
    bool is_exist = false;
    InnerIdType inner_id = INVALID_ID;
    if (use_reverse_map_) {
        is_exist = label_remap_.Find(label, inner_id);
        if (not is_exist) {
            return false;
        }
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
    InnerIdType inner_id = INVALID_ID;
    if (not this->label_remap_.Find(label, inner_id)) {
        return INVALID_ID;
    }
    return inner_id;
}

InnerIdType
LabelTable::get_id_by_label_with_label_table(LabelType label) const noexcept {
    auto result = std::find(label_table_.begin(), label_table_.end(), label);
    if (result == label_table_.end()) {
        return INVALID_ID;
    }
    return result - label_table_.begin();
}

std::pair<bool, InnerIdType>
LabelTable::TryGetIdByLabel(LabelType label, bool return_even_removed) const noexcept {
    InnerIdType id;
    if (use_reverse_map_) {
        id = this->get_id_by_label_with_reverse_map(label);
    } else {
        id = this->get_id_by_label_with_label_table(label);
    }
    if (id == INVALID_ID) {
        return {false, 0};
    }
    if (not return_even_removed) {
        std::shared_lock rlock(delete_ids_mutex_);
        if (this->deleted_ids_.count(id) > 0) {
            return {false, 0};
        }
    }
    return {true, id};
}

InnerIdType
LabelTable::GetIdByLabel(LabelType label, bool return_even_removed) const {
    auto [success, inner_id] = TryGetIdByLabel(label, return_even_removed);
    if (not success) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("label {} does not exist or is removed", label));
    }
    return inner_id;
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
LabelTable::Deserialize(StreamReader& reader) {
    StreamReader::ReadVector(reader, label_table_);
    if (use_reverse_map_) {
        this->label_remap_.Clear();
        this->label_remap_.Reserve(label_table_.size());
        for (InnerIdType id = 0; id < label_table_.size(); ++id) {
            this->label_remap_.InsertOrAssign(label_table_[id], id);
        }
    }

    if (is_legacy_duplicate_format_ && duplicate_tracker_ != nullptr) {
        duplicate_tracker_->DeserializeFromLegacyFormat(reader, label_table_.size());
    }
    is_legacy_duplicate_format_ = false;

    if (support_tombstone_) {
        StreamReader::ReadObj(reader, deleted_ids_);
    }

    this->total_count_.store(static_cast<int64_t>(label_table_.size()));
}

void
LabelTable::MergeOther(const LabelTablePtr& other, const IdMapFunction& id_map) {
    auto other_size = other->GetTotalCount();
    auto current_total_count = total_count_.load();
    auto other_size_u = static_cast<uint64_t>(other_size);
    auto current_total_count_u = static_cast<uint64_t>(current_total_count);
    this->label_table_.resize(current_total_count_u + other_size_u);
    if (use_reverse_map_) {
        this->label_remap_.Reserve(this->label_remap_.Size() + other_size_u);
        for (uint64_t i = 0; i < other_size_u; ++i) {
            auto new_label = std::get<1>(id_map(other->label_table_[i]));
            auto new_inner_id = static_cast<InnerIdType>(i + current_total_count_u);
            this->label_table_[i + current_total_count_u] = new_label;
            this->label_remap_.InsertOrAssign(new_label, new_inner_id);
        }
    } else {
        for (uint64_t i = 0; i < other_size_u; ++i) {
            auto new_label = std::get<1>(id_map(other->label_table_[i]));
            this->label_table_[i + current_total_count_u] = new_label;
        }
    }
    total_count_ += static_cast<int64_t>(other_size_u);
}
}  // namespace vsag
