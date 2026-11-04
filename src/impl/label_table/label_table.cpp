
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

#include "utils/search_lock.h"

namespace vsag {

class RemoveListFilter : public Filter {
public:
    explicit RemoveListFilter(const UnorderedSet<InnerIdType>& remove_ids,
                              std::shared_mutex& delete_ids_mutex)
        : Filter(), remove_ids_(remove_ids), delete_ids_mutex_(delete_ids_mutex) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t inner_id) const override {
        auto rlock = make_search_read_lock(delete_ids_mutex_);
        return remove_ids_.count(inner_id) == 0;
    }

private:
    const UnorderedSet<InnerIdType>& remove_ids_;
    std::shared_mutex& delete_ids_mutex_;
};

class RemoveListReadView : public Filter {
public:
    // The guard pins ids against mutation, but does not own its LabelTable or mutex.
    // The caller must keep the table alive until every copy of the view is destroyed.
    RemoveListReadView(const UnorderedSet<InnerIdType>& ids,
                       std::shared_lock<std::shared_mutex>&& guard)
        : guard_(std::move(guard)), ids_(ids) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        return ids_.count(id) == 0;
    }

private:
    std::shared_lock<std::shared_mutex> guard_;
    const UnorderedSet<InnerIdType>& ids_;
};

FilterPtr
LabelTable::GetDeletedIdsReadView() const {
    auto guard = make_search_read_lock(delete_ids_mutex_);
    if (deleted_ids_.empty()) {
        return nullptr;
    }
    return std::make_shared<RemoveListReadView>(deleted_ids_, std::move(guard));
}

void
LabelTable::RestoreDeletedIds(const Vector<InnerIdType>& ids, uint64_t total) {
    // Reads label_table_ and writes label_remap_ while updating the delete set, so it needs
    // both locks (delete first, per the class-wide order declared on storage_mutex_).
    std::scoped_lock lock(delete_ids_mutex_);
    std::scoped_lock slock(storage_mutex_);
    CHECK_ARGUMENT(total <= label_table_.size(), "logical count exceeds label storage");
    UnorderedSet<InnerIdType> restored(allocator_);
    for (auto id : ids) {
        CHECK_ARGUMENT(id < total, "deleted inner ID exceeds logical count");
        restored.insert(id);
    }
    deleted_ids_.swap(restored);
    // Deserialization may have recorded -1 slots before the persisted MCI mask was known.
    // Only occupied live rows count; unused label capacity may also contain -1.
    active_padding_label_ids_.clear();
    // Prefer the live incarnation when tombstone moves changed physical row order.
    // This O(total) pass is required for shadowed labels; restoring deleted IDs alone
    // cannot identify all live labels overwritten by tombstones during deserialization.
    for (uint64_t id = 0; id < total; ++id) {
        if (deleted_ids_.count(static_cast<InnerIdType>(id)) == 0) {
            if (label_table_[id] == -1) {
                active_padding_label_ids_.insert(static_cast<InnerIdType>(id));
            }
            if (use_reverse_map_) {
                label_remap_.InsertOrAssign(label_table_[id], static_cast<InnerIdType>(id));
            }
        }
    }
    total_count_.store(static_cast<int64_t>(total));
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
      active_padding_label_ids_(allocator),
      source_id_table_(0, allocator) {
    (void)compress_redundant_data;
    deleted_ids_filter_ = std::make_shared<RemoveListFilter>(deleted_ids_, delete_ids_mutex_);
}

bool
LabelTable::CheckLabel(LabelType label) const {
    // All deletion-state writers also hold storage_mutex_ exclusively. A search callback
    // may already pin delete_ids_mutex_ through GetDeletedIdsReadView(), so avoid taking
    // that non-recursive mutex again, even when the label exists.
    auto slock = make_search_read_lock(storage_mutex_);
    return CheckLabelLocked(label);
}

bool
LabelTable::CheckLabelLocked(LabelType label) const {
    // Caller holds storage_mutex_; every deletion-state writer holds it exclusively.
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
    if (this->deleted_ids_.count(inner_id) > 0) {
        return false;
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
    // The storage lock protects the label scan/remap against reallocation and deletion
    // state against mutation. Acquiring the deletion mutex here would recursively lock a
    // pinned search view on the same thread.
    auto slock = make_search_read_lock(storage_mutex_);
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
    // Resolving labels reads label_table_/label_remap_ while the delete set is written, so hold
    // both locks across the whole operation (delete first, per the class-wide order).
    std::scoped_lock wlock(delete_ids_mutex_);
    std::scoped_lock slock(storage_mutex_);
    std::vector<InnerIdType> ids;
    ids.reserve(labels.size());
    for (const auto& label : labels) {
        InnerIdType id;
        if (this->use_reverse_map_) {
            id = this->get_id_by_label_with_reverse_map(label);
        } else {
            id = this->get_id_by_label_with_label_table(label);
        }
        if (id != INVALID_ID) {
            ids.push_back(id);
        }
    }
    uint32_t removed_count = 0;
    for (const auto& id : ids) {
        if (this->deleted_ids_.insert(id).second) {
            active_padding_label_ids_.erase(id);
            ++removed_count;
        }
    }
    return removed_count;
}

void
LabelTable::Deserialize(StreamReader& reader) {
    std::scoped_lock wlock(delete_ids_mutex_);
    std::scoped_lock slock(storage_mutex_);
    StreamReader::ReadVector(reader, label_table_);
    MarkLabelsMutated();
    // Locked helper: both locks are already held.
    RebuildActivePaddingLabelIdsLocked();
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

    this->total_count_.store(static_cast<int64_t>(label_table_.size()));
}

void
LabelTable::TrimUnusedSlots(uint64_t valid_count) {
    std::scoped_lock wlock(delete_ids_mutex_);
    std::scoped_lock slock(storage_mutex_);
    if (valid_count > label_table_.size()) {
        throw VsagException(ErrorType::READ_ERROR,
                            "label table is smaller than the stored vector count");
    }
    label_table_.resize(valid_count);
    total_count_.store(static_cast<int64_t>(valid_count));
    // Locked helper: both locks are already held.
    RebuildActivePaddingLabelIdsLocked();
    if (use_reverse_map_) {
        label_remap_.Clear();
        label_remap_.Reserve(valid_count);
        for (InnerIdType id = 0; id < valid_count; ++id) {
            label_remap_.InsertOrAssign(label_table_[id], id);
        }
    }
}

void
LabelTable::MergeOther(const LabelTablePtr& other, const IdMapFunction& id_map) {
    //            -- Lock order --
    //  1) Snapshot `other` under its lock (delete-then-storage).
    //  2) Reserve space once under our lock, then for each source row:
    //     call id_map outside any lock, briefly re-acquire our locks to
    //     publish ONE label.  This preserves incremental-visibility:
    //     a callback that calls destination->CheckIdExist(candidate) sees
    //     every label already mapped from earlier source rows, while
    //     callbacks still run outside the non-recursive mutexes so
    //     re-entrant reads are safe. Merge must be exclusive of other destination
    //     mutations: id_map may check destination labels before publishing, and
    //     per-row locks cannot make that check-and-publish atomic against Add.

    // -- 1. Snapshot `other` under its lock --
    // A concurrent shrink can leave the logical count larger than physical storage.
    // Reject that inconsistent state rather than indexing past the end or silently
    // dropping source rows (which would shift the merged inner-id layout).
    std::vector<LabelType> other_labels;
    std::vector<bool> other_removed;
    {
        std::shared_lock other_delete_lock(other->delete_ids_mutex_);
        std::shared_lock other_storage_lock(other->storage_mutex_);
        const auto other_count = other->total_count_.load();
        CHECK_ARGUMENT(other_count >= 0, "merge source logical count is negative");
        CHECK_ARGUMENT(static_cast<uint64_t>(other_count) <= other->label_table_.size(),
                       "merge source logical count exceeds label storage");
        const auto other_size = static_cast<uint64_t>(other_count);
        other_labels.reserve(other_size);
        other_removed.reserve(other_size);
        for (uint64_t i = 0; i < other_size; ++i) {
            other_labels.push_back(other->label_table_[i]);
            other_removed.push_back(other->deleted_ids_.count(static_cast<InnerIdType>(i)) > 0);
        }
    }
    const auto other_size_u = other_labels.size();

    // -- 2. Reserve / resize once --
    const uint64_t base = [&]() -> uint64_t {
        std::unique_lock<std::shared_mutex> delete_lock(delete_ids_mutex_);
        std::unique_lock<std::shared_mutex> write_lock(storage_mutex_);
        auto b = static_cast<uint64_t>(total_count_.load());
        label_table_.resize(b + other_size_u);
        if (use_reverse_map_) {
            label_remap_.Reserve(label_remap_.Size() + other_size_u);
        }
        return b;
    }();

    // -- 3. Per-element: callback outside lock, write inside lock --
    for (uint64_t i = 0; i < other_size_u; ++i) {
        // Callback runs outside any lock; re-entrant reads are safe.
        LabelType new_label = std::get<1>(id_map(other_labels[i]));

        // Publish ONE label so the next callback sees it.
        std::unique_lock<std::shared_mutex> delete_lock(delete_ids_mutex_);
        std::unique_lock<std::shared_mutex> write_lock(storage_mutex_);
        auto new_inner_id = static_cast<InnerIdType>(i + base);
        CHECK_ARGUMENT(i + base < label_table_.size(),
                       "merge destination label storage shrank during callbacks");
        label_table_[i + base] = new_label;
        if (new_label == -1 && not other_removed[i]) {
            active_padding_label_ids_.insert(new_inner_id);
        }
        if (use_reverse_map_) {
            label_remap_.InsertOrAssign(new_label, new_inner_id);
        }
    }
    MarkLabelsMutated();
    total_count_ += static_cast<int64_t>(other_size_u);
}
}  // namespace vsag
