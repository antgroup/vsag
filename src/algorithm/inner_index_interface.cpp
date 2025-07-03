
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

#include <fmt/format.h>

#include "brute_force.h"
#include "empty_index_binary_set.h"
#include "hgraph.h"
#include "impl/filter/filter_headers.h"
#include "storage/serialization.h"
#include "utils/slow_task_timer.h"
#include "utils/util_functions.h"

namespace vsag {

InnerIndexInterface::InnerIndexInterface(ParamPtr index_param, const IndexCommonParam& common_param)
    : allocator_(common_param.allocator_.get()),
      create_param_ptr_(std::move(index_param)),
      dim_(common_param.dim_),
      metric_(common_param.metric_),
      data_type_(common_param.data_type_) {
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
        filter_ptr = std::make_shared<BlackListFilter>(filter);
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
        filter_ptr = std::make_shared<BlackListFilter>(invalid);
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
        filter_ptr = std::make_shared<BlackListFilter>(invalid);
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
        filter_ptr = std::make_shared<BlackListFilter>(filter);
    }
    return this->RangeSearch(query, radius, parameters, filter_ptr, limited_size);
}

BinarySet
InnerIndexInterface::Serialize() const {
    std::string time_record_name = this->GetName() + " Serialize";
    SlowTaskTimer t(time_record_name);

    if (GetNumElements() == 0) {
        // TODO(wxyu): remove this if condition
        // if (not Options::Instance().new_version()) {
        //     return EmptyIndexBinarySet::Make(this->GetName());
        // }

        auto metadata = std::make_shared<Metadata>();
        metadata->SetEmptyIndex(true);

        BinarySet bs;
        bs.Set(SERIAL_META_KEY, metadata->ToBinary());
        return bs;
    }

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

#define CHECK_SELF_EMPTY                                                  \
    if (this->GetNumElements() > 0) {                                     \
        throw VsagException(ErrorType::INDEX_NOT_EMPTY,                   \
                            "failed to Deserialize: index is not empty"); \
    }

void
InnerIndexInterface::Deserialize(const BinarySet& binary_set) {
    CHECK_SELF_EMPTY;

    std::string time_record_name = this->GetName() + " Deserialize";
    SlowTaskTimer t(time_record_name);

    // new version serialization will contains the META_KEY
    if (binary_set.Contains(SERIAL_META_KEY)) {
        logger::debug("parse with new version format");
        auto metadata = std::make_shared<Metadata>(binary_set.Get(SERIAL_META_KEY));

        if (metadata->EmptyIndex()) {
            return;
        }
    } else {
        logger::debug("parse with v0.11 version format");

        // check if binary set is an empty index
        if (binary_set.Contains(BLANK_INDEX)) {
            return;
        }
    }

    Binary b = binary_set.Get(this->GetName());
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        // logger::debug("read offset {} len {}", offset, len);
        std::memcpy(dest, b.data.get() + offset, len);
    };

    try {
        uint64_t cursor = 0;
        auto reader = ReadFuncStreamReader(func, cursor, b.size);
        this->Deserialize(reader);
    } catch (const std::runtime_error& e) {
        throw VsagException(ErrorType::READ_ERROR, "failed to Deserialize: ", e.what());
    }
}

void
InnerIndexInterface::Deserialize(const ReaderSet& reader_set) {
    CHECK_SELF_EMPTY;

    std::string time_record_name = this->GetName() + " Deserialize";
    SlowTaskTimer t(time_record_name);

    try {
        auto index_reader = reader_set.Get(this->GetName());
        auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
            index_reader->Read(offset, len, dest);
        };
        uint64_t cursor = 0;
        auto reader = ReadFuncStreamReader(func, cursor, index_reader->Size());
        this->Deserialize(reader);
        return;
    } catch (const std::bad_alloc& e) {
        throw VsagException(ErrorType::READ_ERROR, "failed to Deserialize: ", e.what());
    }
}

void
InnerIndexInterface::Serialize(std::ostream& out_stream) const {
    std::string time_record_name = this->GetName() + " Deserialize";
    SlowTaskTimer t(time_record_name);

    if (GetNumElements() == 0) {
        // TODO(wxyu): remove this if condition
        // if (not Options::Instance().new_version()) {
        //     return;
        // }

        auto metadata = std::make_shared<Metadata>();
        metadata->SetEmptyIndex(true);
        auto footer = std::make_shared<Footer>(metadata);
        IOStreamWriter writer(out_stream);
        footer->Write(writer);
        return;
    }

    IOStreamWriter writer(out_stream);
    this->Serialize(writer);
}

void
InnerIndexInterface::Deserialize(std::istream& in_stream) {
    CHECK_SELF_EMPTY;

    std::string time_record_name = this->GetName() + " Deserialize";
    SlowTaskTimer t(time_record_name);
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

InnerIndexPtr
InnerIndexInterface::Clone(const IndexCommonParam& param) {
    std::stringstream ss;
    IOStreamWriter writer(ss);
    this->Serialize(writer);
    ss.seekg(0, std::ios::beg);
    IOStreamReader reader(ss);
    auto max_size = this->CalSerializeSize();
    BufferStreamReader buffer_reader(&reader, max_size, this->allocator_);
    auto index = this->Fork(param);
    index->Deserialize(buffer_reader);
    return index;
}

InnerIndexPtr
InnerIndexInterface::FastCreateIndex(const std::string& index_fast_str,
                                     const IndexCommonParam& common_param) {
    auto strs = split_string(index_fast_str, fast_string_delimiter);
    if (strs.size() < 2) {
        throw VsagException(ErrorType::INVALID_ARGUMENT, "fast str is too short");
    }
    if (strs[0] == INDEX_TYPE_HGRAPH) {
        if (strs.size() < 3) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "fast str(hgraph) is too short");
        }
        constexpr const char* build_string_temp = R"(
        {{
            "max_degree": {},
            "base_quantization_type": "{}",
            "use_reorder": {},
            "precise_quantization_type": "{}"
        }}
        )";
        auto max_degree = std::stoi(strs[1]);
        auto base_quantization_type = strs[2];
        bool use_reorder = false;
        std::string precise_quantization_type = "fp32";
        if (strs.size() == 4) {
            use_reorder = true;
            precise_quantization_type = strs[3];
        }
        JsonType json = JsonType::parse(fmt::format(build_string_temp,
                                                    max_degree,
                                                    base_quantization_type,
                                                    use_reorder,
                                                    precise_quantization_type));
        auto param_ptr = HGraph::CheckAndMappingExternalParam(json, common_param);
        return std::make_shared<HGraph>(param_ptr, common_param);
    }
    if (strs[0] == INDEX_BRUTE_FORCE) {
        constexpr const char* build_string_temp = R"(
        {{
            "quantization_type": "{}"
        }}
        )";
        JsonType json = JsonType::parse(fmt::format(build_string_temp, strs[1]));
        auto param_ptr = BruteForce::CheckAndMappingExternalParam(json, common_param);
        return std::make_shared<BruteForce>(param_ptr, common_param);
    }
    throw VsagException(ErrorType::INVALID_ARGUMENT,
                        fmt::format("not support fast string create type: {},"
                                    " only support bruteforce and hgraph",
                                    strs[0]));
}

}  // namespace vsag
