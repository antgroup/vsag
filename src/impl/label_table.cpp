
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

LabelTable::LabelRemap::LabelRemap(Allocator* allocator, LabelRemapType remap_type)
    : allocator_(allocator), remap_type_(remap_type) {
    Reset();
}

void
LabelTable::LabelRemap::Reset() {
    if (remap_type_ == LabelRemapType::ROBIN) {
        robin_map_ = std::make_unique<UnorderedMap<LabelType, InnerIdType>>(0, allocator_);
        robin_map_->max_load_factor(0.75F);
        pg_map_.reset();
        return;
    }
    pg_map_ = std::make_unique<PGUnorderedMap<LabelType, InnerIdType>>(0, allocator_);
    pg_map_->max_load_factor(0.75F);
    robin_map_.reset();
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
        auto iter = pg_map_->find(label);
        if (iter == pg_map_->end()) {
            return false;
        }
        inner_id = iter->second;
        return true;
    }
    auto iter = robin_map_->find(label);
    if (iter == robin_map_->end()) {
        return false;
    }
    inner_id = iter->second;
    return true;
}

void
LabelTable::MergeOther(const LabelTablePtr& other, const IdMapFunction& id_map) {
    auto other_size = other->GetTotalCount();
    this->label_table_.resize(total_count_ + other_size);
    for (int64_t i = 0; i < other_size; ++i) {
        auto new_label = std::get<1>(id_map(other->label_table_[i]));
        this->label_table_[i + total_count_] = new_label;
        this->label_remap_.InsertOrAssign(new_label, i + total_count_);
    }
    total_count_ += other_size;
}
}  // namespace vsag
