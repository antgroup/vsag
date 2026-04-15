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

/// @file label_table.h
/// @brief Label table for managing mappings between internal IDs and external labels.

#pragma once

#include <fmt/format.h>
#include <vsag/filter.h>

#include <atomic>
#include <shared_mutex>

#include "datacell/duplicate_interface.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag_exception.h"

namespace vsag {

DEFINE_POINTER(LabelTable);

/// @brief Function type for mapping IDs between tables.
using IdMapFunction = std::function<std::tuple<bool, int64_t>(int64_t)>;

/// @brief Label table for managing bidirectional mappings between internal IDs and external labels.
///
/// This class provides efficient label-to-ID and ID-to-label lookups, supports tombstone
/// deletion, and handles serialization/deserialization for persistence.
class LabelTable {
public:
    /// @brief Constructs a label table with the specified options.
    /// @param allocator Allocator for memory management.
    /// @param use_reverse_map Whether to maintain reverse map for fast label-to-ID lookup.
    /// @param compress_redundant_data Whether to compress redundant data.
    explicit LabelTable(Allocator* allocator,
                        bool use_reverse_map = true,
                        bool compress_redundant_data = false);

    /// Invalid ID constant, used to mark deleted or non-existent entries.
    static constexpr InnerIdType INVALID_ID = std::numeric_limits<InnerIdType>::max();

    /// @brief Inserts a new label-ID mapping.
    /// @param id Internal ID.
    /// @param label External label.
    void
    Insert(InnerIdType id, LabelType label) {
        if (use_reverse_map_) {
            label_remap_[label] = id;
        }
        if (id + 1 > label_table_.size()) {
            label_table_.resize(id + 1);
        }
        label_table_[id] = label;
        total_count_++;
    }

    /// @brief Marks the table as immutable and releases the reverse map.
    void
    SetImmutable() {
        this->use_reverse_map_ = false;
        PGUnorderedMap<LabelType, InnerIdType> empty_remap(allocator_);
        this->label_remap_.swap(empty_remap);
    }

    /// @brief Marks labels as removed.
    /// @param labels The labels to mark as removed.
    /// @return The number of labels marked as removed.
    uint32_t
    MarkRemove(const std::vector<LabelType>& labels);

    /// @brief Marks a label as removed.
    /// @param label The label to mark as removed.
    /// @return The number of labels marked as removed.
    uint32_t
    MarkRemove(const LabelType& label) {
        return MarkRemove(std::vector<LabelType>({label}));
    }

    /// @brief Recovers a previously removed label.
    /// @param label The label to recover.
    /// @return True if recovery was successful.
    inline bool
    RecoverRemove(LabelType label) {
        // 1. check is removed
        if (not use_reverse_map_) {
            return false;
        }
        auto iter = label_remap_.find(label);
        if (iter == label_remap_.end() or iter->second != std::numeric_limits<InnerIdType>::max()) {
            return false;
        }

        // 2. find inner_id
        auto inner_id = GetIdByLabel(label, true);

        // 3. recover
        deleted_ids_.erase(inner_id);
        label_remap_[label] = inner_id;
        return true;
    }

    /// @brief Checks if a label is a tombstone (marked for deletion).
    /// @param label The label to check.
    /// @return True if the label is a tombstone.
    inline bool
    IsTombstoneLabel(LabelType label) {
        if (not use_reverse_map_) {
            return false;
        }
        auto iter = label_remap_.find(label);
        return (iter != label_remap_.end() and
                iter->second == std::numeric_limits<InnerIdType>::max());
    }

    /// @brief Checks whether an id is removed.
    /// @param id The id to check.
    /// @return True if the id is removed, false otherwise.
    bool
    IsRemoved(InnerIdType id) {
        std::shared_lock rlock(delete_ids_mutex_);
        return deleted_ids_.count(id) != 0;
    }

    /// @brief Erases an ID from the deleted set.
    /// @param id The ID to erase.
    void
    EraseFromDeletedIds(InnerIdType id) {
        std::scoped_lock wlock(delete_ids_mutex_);
        deleted_ids_.erase(id);
    }

    /// @brief Gets ID by label.
    /// @param label The label to query.
    /// @param return_even_removed Whether to return even if the id is removed.
    /// @return The ID corresponding to the label.
    /// @throws VsagException if the label does not exist or is removed.
    InnerIdType
    GetIdByLabel(LabelType label, bool return_even_removed = false) const;

    /// @brief Tries to get ID by label without throwing exception.
    /// @param label The label to query.
    /// @param return_even_removed Whether to return even if the id is removed.
    /// @return A pair where first indicates success and second is the inner_id.
    std::pair<bool, InnerIdType>
    TryGetIdByLabel(LabelType label, bool return_even_removed = false) const noexcept;

    /// @brief Checks whether a label exists and not been removed.
    /// @param label The label to check.
    /// @return True if the label exists and not been removed, false otherwise.
    bool
    CheckLabel(LabelType label) const;

    /// @brief Updates a label to a new value.
    /// @param old_label Old label to replace.
    /// @param new_label New label to use.
    /// @throws VsagException if new_label is already occupied or old_label does not exist.
    void
    UpdateLabel(LabelType old_label, LabelType new_label) {
        // 1. check whether new_label is occupied
        if (CheckLabel(new_label)) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("new label {} has been in Index", new_label));
        }

        // 2. update label_table_
        // Important: there may be multiple occurrences of old_label, so we need to update every one
        bool found = false;
        for (size_t i = 0; i < label_table_.size(); ++i) {
            if (label_table_[i] == old_label) {
                label_table_[i] = new_label;
                found = true;
            }
        }
        if (not found) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                fmt::format("old label {} does not exist", old_label));
        }

        // 3. update label_remap_
        if (use_reverse_map_) {
            // note that currently, old_label must exist
            auto iter_old = label_remap_.find(old_label);
            auto internal_id = iter_old->second;
            label_remap_.erase(iter_old);
            label_remap_[new_label] = internal_id;
        }
    }

    /// @brief Gets label by internal ID.
    /// @param inner_id Internal ID to query.
    /// @return The label corresponding to the internal ID.
    /// @throws VsagException if the ID is out of range.
    LabelType
    GetLabelById(InnerIdType inner_id) const {
        if (inner_id >= label_table_.size()) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("id is too large {} >= {}", inner_id, label_table_.size()));
        }
        return this->label_table_[inner_id];
    }

    /// @brief Gets pointer to all labels array.
    /// @return Pointer to the label array.
    inline const LabelType*
    GetAllLabels() const {
        return label_table_.data();
    }

    /// @brief Serializes the label table to a stream writer.
    /// @param writer Stream writer for output.
    void
    Serialize(StreamWriter& writer) const {
        StreamWriter::WriteVector(writer, label_table_);
        if (support_tombstone_) {
            StreamWriter::WriteObj(writer, deleted_ids_);
        }
    }

    /// @brief Deserializes the label table from a stream reader.
    /// @param reader Stream reader for input.
    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        StreamReader::ReadVector(reader, label_table_);
        if (use_reverse_map_) {
            for (InnerIdType id = 0; id < label_table_.size(); ++id) {
                this->label_remap_[label_table_[id]] = id;
            }
        }
        if (support_tombstone_) {
            StreamReader::ReadObj(reader, deleted_ids_);
        }

        this->total_count_.store(label_table_.size());
    }

    /// @brief Deserializes the label table from a stream reader.
    /// @param reader Stream reader for input.
    void
    Deserialize(StreamReader& reader);

    /// @brief Resizes the label table.
    /// @param new_size New size for the table.
    void
    Resize(uint64_t new_size) {
        if (new_size < total_count_) {
            return;
        }
        label_table_.resize(new_size);
    }

    /// @brief Gets the total count of labels.
    /// @return Total count of labels.
    int64_t
    GetTotalCount() {
        return total_count_;
    }

    /// @brief Sets the duplicate tracker.
    /// @param tracker Duplicate tracker to use.
    void
    SetDuplicateTracker(DuplicateTrackerPtr tracker) {
        duplicate_tracker_ = std::move(tracker);
    }

    /// @brief Merges another label table into this one.
    /// @param other Other label table to merge.
    /// @param id_map Optional function to map IDs during merge.
    void
    MergeOther(const LabelTablePtr& other, const IdMapFunction& id_map = nullptr);

    /// @brief Gets memory usage of the label table.
    /// @return Memory usage in bytes.
    int64_t
    GetMemoryUsage() {
        return sizeof(LabelTable) + label_table_.size() * sizeof(LabelType) +
               label_remap_.size() * (sizeof(LabelType) + sizeof(InnerIdType)) +
               deleted_ids_.size() * sizeof(InnerIdType) + hole_list_.size() * sizeof(InnerIdType);
    }

    /// @brief Gets filter to filter out deleted IDs.
    /// @return Filter for deleted IDs, or nullptr if no deleted IDs.
    FilterPtr
    GetDeletedIdsFilter() {
        std::shared_lock rlock(delete_ids_mutex_);
        if (deleted_ids_.empty()) {
            return nullptr;
        }
        return deleted_ids_filter_;
    }

    /// @brief Gets a limited number of deleted IDs.
    /// @param max_count Maximum number of IDs to return.
    /// @return Vector of deleted IDs.
    std::vector<InnerIdType>
    GetDeletedIds(InnerIdType max_count) {
        std::shared_lock rlock(delete_ids_mutex_);
        if (deleted_ids_.empty()) {
            return {};
        }
        auto size = std::min(static_cast<uint64_t>(max_count), deleted_ids_.size());
        return std::vector<InnerIdType>(deleted_ids_.begin(),
                                        std::next(deleted_ids_.begin(), size));
    }

    /// @brief Gets all deleted IDs.
    /// @return Vector of all deleted IDs.
    std::vector<InnerIdType>
    GetAllDeletedIds() {
        std::shared_lock rlock(delete_ids_mutex_);
        return std::vector<InnerIdType>(deleted_ids_.begin(), deleted_ids_.end());
    }

private:
    /// @brief Gets ID by label using reverse map.
    /// @param label Label to query.
    /// @return Internal ID, or INVALID_ID if not found.
    InnerIdType
    get_id_by_label_with_reverse_map(LabelType label) const noexcept;

    /// @brief Gets ID by label using linear scan of label table.
    /// @param label Label to query.
    /// @return Internal ID, or INVALID_ID if not found.
    InnerIdType
    get_id_by_label_with_label_table(LabelType label) const noexcept;

public:
    /// Label table, map from ID to label.
    Vector<LabelType> label_table_;

    /// Temporary compatibility switch for legacy duplicate payload layout.
    bool is_legacy_duplicate_format_{false};

    /// Whether to use reverse map to speed up GetIdByLabel.
    bool use_reverse_map_{true};
    /// Reverse map from label to ID.
    PGUnorderedMap<LabelType, InnerIdType> label_remap_;

    /// Whether tombstone deletion is supported.
    bool support_tombstone_{false};
    /// Duplicate tracker for handling duplicate labels.
    DuplicateTrackerPtr duplicate_tracker_{nullptr};

    /// Allocator for memory management.
    Allocator* allocator_{nullptr};
    /// Total count of labels in the table.
    std::atomic<int64_t> total_count_{0L};

    /// @brief Pushes a hole (reusable ID) to the hole list.
    /// @param id The ID to push.
    void
    PushHole(InnerIdType id) {
        std::scoped_lock wlock(hole_mutex_);
        hole_list_.push_back(id);
    }

    /// @brief Pops a hole (reusable ID) from the hole list.
    /// @return Pair of (success, id) where success indicates if a hole was available.
    std::pair<bool, InnerIdType>
    PopHole() {
        std::scoped_lock wlock(hole_mutex_);
        if (hole_list_.empty()) {
            return {false, 0};
        }
        InnerIdType id = hole_list_.back();
        hole_list_.pop_back();
        return {true, id};
    }

    /// @brief Checks if there are any holes available.
    /// @return True if holes exist.
    bool
    HasHole() const {
        std::shared_lock rlock(hole_mutex_);
        return not hole_list_.empty();
    }

    /// @brief Gets the count of holes.
    /// @return Number of holes.
    size_t
    GetHoleCount() const {
        std::shared_lock rlock(hole_mutex_);
        return hole_list_.size();
    }

    /// @brief Removes a specific ID from the hole list.
    /// @param id The ID to remove.
    /// @return True if the ID was found and removed.
    bool
    RemoveHole(InnerIdType id) {
        std::scoped_lock wlock(hole_mutex_);
        auto it = std::find(hole_list_.begin(), hole_list_.end(), id);
        if (it != hole_list_.end()) {
            hole_list_.erase(it);
            return true;
        }
        return false;
    }

private:
    /// Record of deleted IDs.
    UnorderedSet<InnerIdType> deleted_ids_;
    /// Filter to filter out deleted IDs.
    FilterPtr deleted_ids_filter_{nullptr};
    /// Mutex to protect deleted_ids_.
    mutable std::shared_mutex delete_ids_mutex_;

    /// Reusable ID list (stack structure).
    Vector<InnerIdType> hole_list_;
    /// Mutex to protect hole_list_.
    mutable std::shared_mutex hole_mutex_;
};

}  // namespace vsag