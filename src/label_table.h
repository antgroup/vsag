
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

#include <fmt/format-inl.h>

#include "stream_reader.h"
#include "stream_writer.h"
#include "typing.h"

namespace vsag {
class LabelTable {
public:
    explicit LabelTable(Allocator* allocator)
        : allocator_(allocator), label_table_(0, allocator), label_remap_(0, allocator){};

    inline void
    Insert(InnerIdType id, LabelType label) {
        label_remap_[label] = id;
        if (id + 1 > label_table_.size()) {
            label_table_.resize(id + 1);
        }
        label_table_[id] = label;
    }

    inline InnerIdType
    GetIdByLabel(LabelType label) const {
        if (this->label_remap_.count(label) == 0) {
            throw std::runtime_error(fmt::format("label {} is not exists", label));
        }
        return this->label_remap_.at(label);
    }

    inline bool
    CheckLabel(LabelType label) const {
        return label_remap_.find(label) != label_remap_.end();
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
    }

    void
    Deserialize(StreamReader& reader) {
        StreamReader::ReadVector(reader, label_table_);
        for (InnerIdType id = 0; id < label_table_.size(); ++id) {
            this->label_remap_[label_table_[id]] = id;
        }
    }

public:
    Vector<LabelType> label_table_;
    UnorderedMap<LabelType, InnerIdType> label_remap_;

    Allocator* allocator_{nullptr};
};

using LabelTablePtr = std::shared_ptr<LabelTable>;

}  // namespace vsag
