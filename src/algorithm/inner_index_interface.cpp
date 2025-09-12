
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
#include "index/index_common_param.h"
#include "index_feature_list.h"
#include "label_table.h"
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
    this->tomb_label_table_ = std::make_shared<LabelTable>(allocator_);
    this->index_feature_list_ = std::make_shared<IndexFeatureList>();
    this->index_feature_list_->SetFeature(SUPPORT_EXPORT_IDS);
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
    if (reader_set.Contains(SERIAL_META_KEY)) {
        logger::debug("parse with new version format");
        const auto& meta_reader = reader_set.Get(SERIAL_META_KEY);
        uint64_t size = meta_reader->Size();
        Binary binary{.data = std::shared_ptr<int8_t[]>(new int8_t[size]), .size = size};
        meta_reader->Read(0, size, binary.data.get());
        auto metadata = std::make_shared<Metadata>(binary);
        if (metadata->EmptyIndex()) {
            return;
        }
    } else {
        logger::debug("parse with v0.14 version format");
        // check if binary set is an empty index
        if (reader_set.Contains(BLANK_INDEX)) {
            return;
        }
    }

    try {
        auto index_reader = reader_set.Get(this->GetName());
        auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
            index_reader->Read(offset, len, dest);
        };
        uint64_t cursor = 0;
        auto reader = ReadFuncStreamReader(func, cursor, index_reader->Size());
        this->Deserialize(reader);
        this->SetIO(index_reader);
        return;
    } catch (const std::bad_alloc& e) {
        throw VsagException(ErrorType::READ_ERROR, "failed to Deserialize: ", e.what());
    }
}

bool
InnerIndexInterface::CheckFeature(IndexFeature feature) const {
    return this->index_feature_list_->CheckFeature(feature);
}

bool
InnerIndexInterface::CheckIdExist(int64_t id) const {
    return this->label_table_->CheckLabel(id);
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

        auto footer = Footer::Parse(reader);
        if (footer != nullptr) {
            auto metadata = footer->GetMetadata();
            if (metadata->EmptyIndex()) {
                return;
            }
        }
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

DatasetPtr
InnerIndexInterface::CalDistanceById(const DatasetPtr& query,
                                     const int64_t* ids,
                                     int64_t count) const {
    auto result = Dataset::Make();
    result->Owner(true, allocator_);
    auto* distances = (float*)allocator_->Allocate(sizeof(float) * count);
    result->Distances(distances);
    for (int64_t i = 0; i < count; ++i) {
        try {
            distances[i] = this->CalcDistanceById(query, ids[i]);
        } catch (std::runtime_error& e) {
            logger::debug(fmt::format("failed to find id: {}", ids[i]));
            distances[i] = -1;
        }
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

DatasetPtr
InnerIndexInterface::GetVectorByIds(const int64_t* ids, int64_t count) const {
    DatasetPtr vectors = Dataset::Make();
    auto* float_vectors = (float*)allocator_->Allocate(sizeof(float) * count * dim_);
    if (float_vectors == nullptr) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "failed to allocate memory for vectors");
    }
    vectors->NumElements(count)->Dim(dim_)->Float32Vectors(float_vectors)->Owner(true, allocator_);
    for (int i = 0; i < count; ++i) {
        InnerIdType inner_id = this->label_table_->GetIdByLabel(ids[i]);
        this->GetVectorByInnerId(inner_id, float_vectors + i * dim_);
    }
    return vectors;
}

DatasetPtr
InnerIndexInterface::ExportIDs() const {
    std::shared_lock lock(this->label_lookup_mutex_);
    DatasetPtr result = Dataset::Make();
    auto num_element = this->label_table_->GetTotalCount();
    auto* labels = (LabelType*)allocator_->Allocate(sizeof(LabelType) * num_element);
    const auto* origin_label = this->label_table_->GetAllLabels();
    memcpy(labels, origin_label, sizeof(LabelType) * num_element);
    result->NumElements(num_element)->Ids(labels)->Dim(1)->Owner(true, allocator_);
    return result;
}

DatasetPtr
InnerIndexInterface::GetDataByIds(const int64_t* ids, int64_t count) const {
    uint64_t selected_flag = DATA_FLAG_ID;
    if (this->has_raw_vector_) {
        selected_flag |= DATA_FLAG_FLOAT32_VECTOR;
    }
    if (this->has_attribute_) {
        selected_flag |= DATA_FLAG_ATTRIBUTE;
    }
    if (this->extra_info_size_ > 0) {
        selected_flag |= DATA_FLAG_EXTRA_INFO;
    }
    return this->GetDataByIdsWithFlag(ids, count, selected_flag);
}

DatasetPtr
InnerIndexInterface::GetDataByIdsWithFlag(const int64_t* ids,
                                          int64_t count,
                                          uint64_t selected_data_flag) const {
    auto* inner_ids =
        reinterpret_cast<InnerIdType*>(this->allocator_->Allocate(count * sizeof(InnerIdType)));
    {
        std::shared_lock lock(this->label_lookup_mutex_);
        for (int64_t i = 0; i < count; ++i) {
            inner_ids[i] = this->label_table_->GetIdByLabel(ids[i]);
        }
    }
    auto dataset = Dataset::Make();
    dataset->NumElements(count)->Dim(dim_)->Owner(true, allocator_);
    if ((selected_data_flag & DATA_FLAG_FLOAT32_VECTOR) != 0U) {
        if (not this->has_raw_vector_) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "has_raw_vector_ is false");
        }
        auto* fp32_data = reinterpret_cast<float*>(
            this->allocator_->Allocate(count * this->dim_ * sizeof(float)));
        dataset->Float32Vectors(fp32_data);
        for (int64_t i = 0; i < count; ++i) {
            auto inner_id = this->label_table_->GetIdByLabel(ids[i]);
            this->GetVectorByInnerId(inner_id, fp32_data + i * this->dim_);
        }
    }

    if ((selected_data_flag & DATA_FLAG_ATTRIBUTE) != 0U) {
        if (not this->has_attribute_) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "has_attribute_ is false");
        }
        auto* attribute_data = new AttributeSet[count];
        dataset->AttributeSets(attribute_data);
        for (int64_t i = 0; i < count; ++i) {
            auto inner_id = this->label_table_->GetIdByLabel(ids[i]);
            this->GetAttributeSetByInnerId(inner_id, attribute_data + i);
        }
    }

    if ((selected_data_flag & DATA_FLAG_EXTRA_INFO) != 0U) {
        if (extra_info_size_ == 0) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "extra_info_size_ is 0");
        }
        auto* extra_info =
            reinterpret_cast<char*>(this->allocator_->Allocate(count * extra_info_size_));
        dataset->ExtraInfos(extra_info);
        for (int64_t i = 0; i < count; ++i) {
            auto inner_id = this->label_table_->GetIdByLabel(ids[i]);
            this->extra_infos_->GetExtraInfoById(inner_id, extra_info + i * extra_info_size_);
        }
    }
    if ((selected_data_flag & DATA_FLAG_ID) != 0U) {
        auto* new_ids =
            reinterpret_cast<int64_t*>(this->allocator_->Allocate(count * sizeof(int64_t)));
        memcpy(new_ids, ids, count * sizeof(int64_t));
        dataset->Ids(new_ids);
    }
    this->allocator_->Deallocate(inner_ids);
    return dataset;
}

void
InnerIndexInterface::analyze_quantizer(JsonType& stats,
                                       const float* data,
                                       uint64_t sample_data_size,
                                       int64_t topk,
                                       const std::string& search_param) const {
    // record quantized information
    if (this->use_reorder_) {
        logger::info("analyze_quantizer: sample_data_size = {}, topk = {}", sample_data_size, topk);
        float bias_ratio = 0.0F;
        float inversion_count_rate = 0.0F;
        for (uint64_t i = 0; i < sample_data_size; ++i) {
            float tmp_bias_ratio = 0.0F;
            float tmp_inversion_count_rate = 0.0F;
            this->use_reorder_ = false;
            const auto* query_data = data + i * dim_;
            auto query = Dataset::Make();
            FilterPtr filter = nullptr;
            query->Owner(false)->NumElements(1)->Float32Vectors(query_data)->Dim(dim_);
            auto search_result = this->KnnSearch(query, topk, search_param, filter);
            this->use_reorder_ = true;
            auto distance_result =
                this->CalDistanceById(query_data, search_result->GetIds(), search_result->GetDim());
            const auto* ground_distances = distance_result->GetDistances();
            const auto* approximate_distances = search_result->GetDistances();
            for (int64_t j = 0; j < topk; ++j) {
                if (ground_distances[j] > 0) {
                    tmp_bias_ratio += std::abs(approximate_distances[j] - ground_distances[j]) /
                                      ground_distances[j];
                }
            }
            tmp_bias_ratio /= static_cast<float>(topk);
            bias_ratio += tmp_bias_ratio;
            // calculate inversion count rate
            int64_t inversion_count = 0;
            for (int64_t j = 0; j < search_result->GetDim() - 1; ++j) {
                for (int64_t k = j + 1; k < search_result->GetDim(); ++k) {
                    if (ground_distances[j] > ground_distances[k]) {
                        inversion_count++;
                    }
                }
            }
            int64_t search_count = search_result->GetDim();
            tmp_inversion_count_rate =
                static_cast<float>(inversion_count) /
                (static_cast<float>(search_count * (search_count - 1)) / 2.0F);
            inversion_count_rate += tmp_inversion_count_rate;
        }
        stats["quantization_bias_ratio"] = bias_ratio / static_cast<float>(sample_data_size);
        stats["quantization_inversion_count_rate"] =
            inversion_count_rate / static_cast<float>(sample_data_size);
    }
}

}  // namespace vsag
