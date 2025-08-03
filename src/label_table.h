
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

namespace vsag {

class LabelTable;
using LabelTablePtr = std::shared_ptr<LabelTable>;
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
        if (iter == label_remap_.end()) {
            return false;
        }
        label_remap_.erase(iter);
        return true;
    }

    inline InnerIdType
    GetIdByLabel(LabelType label) const {
        if (use_reverse_map_) {
            if (this->label_remap_.count(label) == 0) {
                throw std::runtime_error(fmt::format("label {} is not exists", label));
            }
            return this->label_remap_.at(label);
        }
        auto result = std::find(label_table_.begin(), label_table_.end(), label);
        if (result == label_table_.end()) {
            throw std::runtime_error(fmt::format("label {} is not exists", label));
        }
        return result - label_table_.begin();
    }

    inline bool
    CheckLabel(LabelType label) const {
        if (use_reverse_map_) {
            return label_remap_.find(label) != label_remap_.end();
        }
        auto result = std::find(label_table_.begin(), label_table_.end(), label);
        return result != label_table_.end();
    }

    inline LabelType
    GetLabelById(InnerIdType inner_id) const {
        if (inner_id >= label_table_.size()) {
            throw std::runtime_error(
                fmt::format("id is too large {} >= {}", inner_id, label_table_.size()));
        }
        return this->label_table_[inner_id];
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
    UnorderedMap<LabelType, InnerIdType> label_remap_;

    bool compress_duplicate_data_{true};

    std::mutex duplicate_mutex_;
    uint64_t duplicate_count_{0L};
    Vector<DuplicateRecord*> duplicate_records_;

    Allocator* allocator_{nullptr};
    std::atomic<int64_t> total_count_{0L};
    bool use_reverse_map_{true};
};

}  // namespace vsag
