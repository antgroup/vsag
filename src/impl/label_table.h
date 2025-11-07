
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

#include <atomic>

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {

DEFINE_POINTER(LabelTable);

using IdMapFunction = std::function<std::tuple<bool, int64_t>(int64_t)>;

struct DuplicateRecord {
    std::mutex duplicate_mutex;
    Vector<InnerIdType> duplicate_ids;
    DuplicateRecord(Allocator* allocator) : duplicate_ids(allocator) {
    }
};

class LabelTable {
public:
    explicit LabelTable(Allocator* allocator,
                        bool use_reverse_map = true,
                        bool compress_redundant_data = false)
        : allocator_(allocator),
          label_table_(0, allocator),
          label_remap_(0, allocator),
          use_reverse_map_(use_reverse_map),
          deleted_ids_(allocator),
          compress_duplicate_data_(compress_redundant_data),
          duplicate_records_(0, allocator){};

    ~LabelTable() {
        for (int i = 0; i < duplicate_records_.size(); ++i) {
            allocator_->Delete(duplicate_records_[i]);
        }
    }

    inline void
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

    inline bool
    Remove(LabelType label) {
        if (not use_reverse_map_) {
            return true;
        }
        auto iter = label_remap_.find(label);
        if (iter == label_remap_.end() or iter->second == std::numeric_limits<InnerIdType>::max()) {
            return false;
        }
        deleted_ids_.insert(iter->second);

        label_remap_[label] = std::numeric_limits<InnerIdType>::max();
        return true;
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

    inline bool
    IsRemoved(InnerIdType id) {
        return not deleted_ids_.empty() && deleted_ids_.count(id) != 0;
    }

    inline InnerIdType
    GetIdByLabel(LabelType label, bool return_even_removed = false) const {
        if (use_reverse_map_ and not return_even_removed) {
            if (this->label_remap_.count(label) == 0) {
                throw std::runtime_error(fmt::format("label {} is not exists", label));
            }
            auto id = this->label_remap_.at(label);
            if (id != std::numeric_limits<InnerIdType>::max()) {
                return id;
            } else {
                throw std::runtime_error(fmt::format("label {} is removed", label));
            }
        }
        auto result = std::find(label_table_.begin(), label_table_.end(), label);
        if (result == label_table_.end()) {
            throw std::runtime_error(fmt::format("label {} is not exists", label));
        }
        return result - label_table_.begin();
    }

    inline bool
    CheckLabel(LabelType label) const {
        // return true when label exists and not been deleted
        if (use_reverse_map_) {
            auto iter = label_remap_.find(label);
            return iter != label_remap_.end() and
                   iter->second != std::numeric_limits<InnerIdType>::max();
        }
        auto result = std::find(label_table_.begin(), label_table_.end(), label);
        return result != label_table_.end();
    }

    inline void
    UpdateLabel(LabelType old_label, LabelType new_label) {
        // 1. check whether new_label is occupied
        if (CheckLabel(new_label)) {
            throw std::runtime_error(fmt::format("new label {} has been in HGraph", new_label));
        }

        // 2. update label_table_
        auto result = std::find(label_table_.begin(), label_table_.end(), old_label);
        if (result == label_table_.end()) {
            throw std::runtime_error(fmt::format("old label {} is not exists", old_label));
        }
        InnerIdType internal_id = result - label_table_.begin();
        label_table_[internal_id] = new_label;

        // 3. update label_remap_
        if (use_reverse_map_) {
            // note that currently, old_label must exist
            auto iter_old = label_remap_.find(old_label);
            internal_id = iter_old->second;
            label_remap_.erase(iter_old);
            label_remap_[new_label] = internal_id;
        }
    }

    inline LabelType
    GetLabelById(InnerIdType inner_id) const {
        if (inner_id >= label_table_.size()) {
            throw std::runtime_error(
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
                    StreamWriter::WriteVector(writer, duplicate_records_[i]->duplicate_ids);
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
                StreamReader::ReadVector(reader, duplicate_records_[id]->duplicate_ids);
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
        duplicate_records_[previous_id]->duplicate_ids.push_back(current_id);
    }

    const Vector<InnerIdType>
    GetDuplicateId(InnerIdType id) const {
        if (duplicate_records_[id] == nullptr) {
            return Vector<InnerIdType>(allocator_);
        }
        std::lock_guard lock(duplicate_records_[id]->duplicate_mutex);
        return duplicate_records_[id]->duplicate_ids;
    }

    void
    MergeOther(const LabelTablePtr& other, const IdMapFunction& id_map = nullptr);

public:
    Vector<LabelType> label_table_;
    STLUnorderedMap<LabelType, InnerIdType> label_remap_;
    UnorderedSet<InnerIdType> deleted_ids_;

    bool compress_duplicate_data_{true};
    bool support_tombstone_{false};

    std::mutex duplicate_mutex_;
    uint64_t duplicate_count_{0L};
    Vector<DuplicateRecord*> duplicate_records_;

    Allocator* allocator_{nullptr};
    std::atomic<int64_t> total_count_{0L};
    bool use_reverse_map_{true};
};

}  // namespace vsag
