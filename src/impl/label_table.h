
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

#include <fmt/format.h>
#include <vsag/filter.h>

#include <atomic>
#include <shared_mutex>

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag_exception.h"

namespace vsag {

DEFINE_POINTER(LabelTable);

using IdMapFunction = std::function<std::tuple<bool, int64_t>(int64_t)>;

struct DuplicateRecord {
    std::mutex duplicate_mutex;
    UnorderedSet<InnerIdType> duplicate_ids;
    DuplicateRecord(Allocator* allocator) : duplicate_ids(allocator) {
    }
};

class LabelTable {
public:
    explicit LabelTable(Allocator* allocator,
                        bool use_reverse_map = true,
                        bool compress_redundant_data = false);

    ~LabelTable() {
        for (auto& record : duplicate_records_) {
            allocator_->Delete(record);
        }
    }

    static constexpr InnerIdType INVALID_ID = std::numeric_limits<InnerIdType>::max();

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

    void
    SetImmutable() {
        this->use_reverse_map_ = false;
        STLUnorderedMap<LabelType, InnerIdType> empty_remap(allocator_);
        this->label_remap_.swap(empty_remap);
    }

    /**
     * Mark labels as removed.
     * @param labels The labels to mark as removed.
     * @return The number of labels marked as removed.
     */
    uint32_t
    MarkRemove(const std::vector<LabelType>& labels);

    /**
     * Mark a label as removed.
     * @param label The label to mark as removed.
     * @return The number of labels marked as removed.
     */
    uint32_t
    MarkRemove(const LabelType& label) {
        return MarkRemove(std::vector<LabelType>({label}));
    }

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

    inline bool
    IsTombstoneLabel(LabelType label) {
        if (not use_reverse_map_) {
            return false;
        }
        auto iter = label_remap_.find(label);
        return (iter != label_remap_.end() and
                iter->second == std::numeric_limits<InnerIdType>::max());
    }

    /**
     * Check whether an id is removed.
     * @param id The id to check.
     * @return True if the id is removed, false otherwise.
     */
    bool
    IsRemoved(InnerIdType id) {
        std::shared_lock rlock(delete_ids_mutex_);
        return deleted_ids_.count(id) != 0;
    }

    /**
     * Get id by label.
     * @param label The label to query.
     * @param return_even_removed Whether to return even if the id is removed.
     * @return The id corresponding to the label.
     * @throws VsagException if the label does not exist or is removed.
     */
    InnerIdType
    GetIdByLabel(LabelType label, bool return_even_removed = false) const;

    /**
     * Check whether a label exists and not been removed.
     * @param label The label to check.
     * @return True if the label exists and not been removed, false otherwise.
     */
    bool
    CheckLabel(LabelType label) const;

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

    LabelType
    GetLabelById(InnerIdType inner_id) const {
        if (inner_id >= label_table_.size()) {
            throw VsagException(
                ErrorType::INTERNAL_ERROR,
                fmt::format("id is too large {} >= {}", inner_id, label_table_.size()));
        }
        return this->label_table_[inner_id];
    }

    inline const LabelType*
    GetAllLabels() const {
        return label_table_.data();
    }

    void
    Serialize(StreamWriter& writer) const {
        StreamWriter::WriteVector(writer, label_table_);
        if (compress_duplicate_data_) {
            StreamWriter::WriteObj(writer, duplicate_count_);
            for (InnerIdType i = 0; i < label_table_.size(); ++i) {
                if (duplicate_records_[i] != nullptr) {
                    StreamWriter::WriteObj(writer, i);
                    Vector<InnerIdType> id_list(allocator_);
                    for (const auto& duplicate_id : duplicate_records_[i]->duplicate_ids) {
                        id_list.emplace_back(duplicate_id);
                    }
                    StreamWriter::WriteVector(writer, id_list);
                }
            }
        }
        if (support_tombstone_) {
            StreamWriter::WriteObj(writer, deleted_ids_);
        }
    }

    void
    Deserialize(lvalue_or_rvalue<StreamReader> reader) {
        StreamReader::ReadVector(reader, label_table_);
        if (use_reverse_map_) {
            for (InnerIdType id = 0; id < label_table_.size(); ++id) {
                this->label_remap_[label_table_[id]] = id;
            }
        }
        if (compress_duplicate_data_) {
            StreamReader::ReadObj(reader, duplicate_count_);
            duplicate_records_.resize(label_table_.size(), nullptr);
            for (InnerIdType i = 0; i < duplicate_count_; ++i) {
                InnerIdType id;
                StreamReader::ReadObj<InnerIdType>(reader, id);
                duplicate_records_[id] = allocator_->New<DuplicateRecord>(allocator_);
                Vector<InnerIdType> id_list(allocator_);
                StreamReader::ReadVector(reader, id_list);
                for (const auto& duplicate_id : id_list) {
                    duplicate_records_[id]->duplicate_ids.insert(duplicate_id);
                }
            }
        }
        if (support_tombstone_) {
            StreamReader::ReadObj(reader, deleted_ids_);
        }
        this->total_count_.store(label_table_.size());
    }

    void
    Resize(uint64_t new_size) {
        if (new_size < total_count_) {
            return;
        }
        label_table_.resize(new_size);
        if (compress_duplicate_data_) {
            duplicate_records_.resize(new_size, nullptr);
        }
    }

    int64_t
    GetTotalCount() {
        return total_count_;
    }

    inline bool
    CompressDuplicateData() const {
        return compress_duplicate_data_;
    }

    inline void
    SetDuplicateId(InnerIdType previous_id, InnerIdType current_id) {
        std::lock_guard duplicate_lock(duplicate_mutex_);
        if (duplicate_records_[previous_id] == nullptr) {
            duplicate_records_[previous_id] = allocator_->New<DuplicateRecord>(allocator_);
            duplicate_count_++;
        }
        std::lock_guard lock(duplicate_records_[previous_id]->duplicate_mutex);
        duplicate_records_[previous_id]->duplicate_ids.insert(current_id);
    }

    const UnorderedSet<InnerIdType>
    GetDuplicateId(InnerIdType id) const {
        if (duplicate_records_[id] == nullptr) {
            return UnorderedSet<InnerIdType>(allocator_);
        }
        std::lock_guard lock(duplicate_records_[id]->duplicate_mutex);
        return duplicate_records_[id]->duplicate_ids;
    }

    void
    MergeOther(const LabelTablePtr& other, const IdMapFunction& id_map = nullptr);

    /**
     * Get memory usage of the label table.
     * @return The memory usage in bytes.
     */
    int64_t
    GetMemoryUsage() {
        return sizeof(LabelTable) + label_table_.size() * sizeof(LabelType) +
               label_remap_.size() * (sizeof(LabelType) + sizeof(InnerIdType)) +
               deleted_ids_.size() * sizeof(InnerIdType) +
               duplicate_records_.size() * (sizeof(DuplicateRecord*) + sizeof(DuplicateRecord));
    }

    /**
     * Get filter to filter out deleted ids.
     * @return The filter.
     */
    FilterPtr
    GetDeletedIdsFilter() {
        std::shared_lock rlock(delete_ids_mutex_);
        if (deleted_ids_.empty()) {
            return nullptr;
        }
        return deleted_ids_filter_;
    }

private:
    InnerIdType
    get_id_by_label_with_reverse_map(LabelType label) const noexcept;

    InnerIdType
    get_id_by_label_with_label_table(LabelType label) const noexcept;

public:
    // Label table, map from id to label.
    Vector<LabelType> label_table_;

    // Whether to use reverse map to speed up GetIdByLabel.
    bool use_reverse_map_{true};
    // Reverse map from label to id.
    STLUnorderedMap<LabelType, InnerIdType> label_remap_;

    bool compress_duplicate_data_{true};
    bool support_tombstone_{false};

    std::mutex duplicate_mutex_;
    uint64_t duplicate_count_{0L};
    Vector<DuplicateRecord*> duplicate_records_;

    Allocator* allocator_{nullptr};
    std::atomic<int64_t> total_count_{0L};

private:
    UnorderedSet<InnerIdType> deleted_ids_;       // Record deleted ids.
    FilterPtr deleted_ids_filter_{nullptr};       // Filter to filter out deleted ids.
    mutable std::shared_mutex delete_ids_mutex_;  // Mutex to protect deleted_ids_.
};

}  // namespace vsag
