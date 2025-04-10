
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

#include "inner_index_interface.h"

#include "base_filter_functor.h"
#include "empty_index_binary_set.h"
#include "utils/slow_task_timer.h"

namespace vsag {

InnerIndexInterface::InnerIndexInterface(const ParamPtr& index_param,
                                         const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()) {
    this->label_table_ = std::make_shared<LabelTable>(allocator_);
    this->index_feature_list_ = std::make_shared<IndexFeatureList>();
}

std::vector<int64_t>
InnerIndexInterface::Build(const DatasetPtr& base) {
    return this->Add(base);
}

DatasetPtr
InnerIndexInterface::KnnSearch(const DatasetPtr& query,
                               int64_t k,
                               const std::string& parameters,
                               const std::function<bool(int64_t)>& filter) const {
    FilterPtr filter_ptr = nullptr;
    if (filter != nullptr) {
        filter_ptr = std::make_shared<UniqueFilter>(filter);
    }
    return this->KnnSearch(query, k, parameters, filter_ptr);
}

DatasetPtr
InnerIndexInterface::KnnSearch(const DatasetPtr& query,
                               int64_t k,
                               const std::string& parameters,
                               const BitsetPtr& invalid) const {
    FilterPtr filter_ptr = nullptr;
    if (invalid != nullptr) {
        filter_ptr = std::make_shared<UniqueFilter>(invalid);
    }
    return this->KnnSearch(query, k, parameters, filter_ptr);
}

DatasetPtr
InnerIndexInterface::RangeSearch(const DatasetPtr& query,
                                 float radius,
                                 const std::string& parameters,
                                 const BitsetPtr& invalid,
                                 int64_t limited_size) const {
    FilterPtr filter_ptr = nullptr;
    if (invalid != nullptr) {
        filter_ptr = std::make_shared<UniqueFilter>(invalid);
    }
    return this->RangeSearch(query, radius, parameters, filter_ptr, limited_size);
}

DatasetPtr
InnerIndexInterface::RangeSearch(const DatasetPtr& query,
                                 float radius,
                                 const std::string& parameters,
                                 const std::function<bool(int64_t)>& filter,
                                 int64_t limited_size) const {
    FilterPtr filter_ptr = nullptr;
    if (filter != nullptr) {
        filter_ptr = std::make_shared<UniqueFilter>(filter);
    }
    return this->RangeSearch(query, radius, parameters, filter_ptr, limited_size);
}

BinarySet
InnerIndexInterface::Serialize() const {
    if (GetNumElements() == 0) {
        return EmptyIndexBinarySet::Make(this->GetName());
    }
    std::string time_record_name = this->GetName() + " Serialize";
    SlowTaskTimer t(time_record_name);
    uint64_t num_bytes = this->CalSerializeSize();
    // TODO(LHT): use try catch

    std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
    auto* buffer = reinterpret_cast<char*>(const_cast<int8_t*>(bin.get()));
    BufferStreamWriter writer(buffer);
    this->Serialize(writer);
    Binary b{
        .data = bin,
        .size = num_bytes,
    };
    BinarySet bs;
    bs.Set(this->GetName(), b);

    return bs;
}

void
InnerIndexInterface::Serialize(std::ostream& out_stream) const {
    IOStreamWriter writer(out_stream);
    this->Serialize(writer);
}

void
InnerIndexInterface::Deserialize(const BinarySet& binary_set) {
    std::string time_record_name = this->GetName() + " Deserialize";
    SlowTaskTimer t(time_record_name);
    if (this->GetNumElements() > 0) {
        throw VsagException(ErrorType::INDEX_NOT_EMPTY,
                            "failed to Deserialize: index is not empty");
    }

    // check if binary set is an empty index
    if (binary_set.Contains(BLANK_INDEX)) {
        return;
    }

    Binary b = binary_set.Get(this->GetName());
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        std::memcpy(dest, b.data.get() + offset, len);
    };

    try {
        uint64_t cursor = 0;
        auto reader = ReadFuncStreamReader(func, cursor);
        this->Deserialize(reader);
    } catch (const std::runtime_error& e) {
        throw VsagException(ErrorType::READ_ERROR, "failed to Deserialize: ", e.what());
    }
}

void
InnerIndexInterface::Deserialize(const ReaderSet& reader_set) {
    std::string time_record_name = this->GetName() + " Deserialize";
    SlowTaskTimer t(time_record_name);
    if (this->GetNumElements() > 0) {
        throw VsagException(ErrorType::INDEX_NOT_EMPTY,
                            "failed to Deserialize: index is not empty");
    }
    try {
        auto index_reader = reader_set.Get(this->GetName());
        auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
            index_reader->Read(offset, len, dest);
        };
        uint64_t cursor = 0;
        auto reader = ReadFuncStreamReader(func, cursor);
        this->Deserialize(reader);
        return;
    } catch (const std::bad_alloc& e) {
        throw VsagException(ErrorType::READ_ERROR, "failed to Deserialize: ", e.what());
    }
}

void
InnerIndexInterface::Deserialize(std::istream& in_stream) {
    std::string time_record_name = this->GetName() + " Deserialize";
    SlowTaskTimer t(time_record_name);
    if (this->GetNumElements() > 0) {
        throw VsagException(ErrorType::INDEX_NOT_EMPTY,
                            "failed to Deserialize: index is not empty");
    }
    try {
        IOStreamReader reader(in_stream);
        this->Deserialize(reader);
        return;
    } catch (const std::bad_alloc& e) {
        throw VsagException(ErrorType::READ_ERROR, "failed to Deserialize: ", e.what());
    }
}
uint64_t
InnerIndexInterface::CalSerializeSize() const {
    auto cal_size_func = [](uint64_t cursor, uint64_t size, void* buf) { return; };
    WriteFuncStreamWriter writer(cal_size_func, 0);
    this->Serialize(writer);
    return writer.cursor_;
}

DatasetPtr
InnerIndexInterface::CalDistanceById(const float* query, const int64_t* ids, int64_t count) const {
    auto result = Dataset::Make();
    result->Owner(true, allocator_);
    auto* distances = (float*)allocator_->Allocate(sizeof(float) * count);
    result->Distances(distances);
    for (int64_t i = 0; i < count; ++i) {
        distances[i] = this->CalcDistanceById(query, ids[i]);
    }
    return result;
}

}  // namespace vsag
