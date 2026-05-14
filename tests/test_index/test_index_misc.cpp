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

#include "test_index_common.h"

namespace fixtures {

void
TestIndex::TestEstimateMemory(const std::string& index_name,
                              const std::string& build_param,
                              const TestDatasetPtr& dataset) {
    auto allocator = std::make_shared<fixtures::MemoryRecordAllocator>();
    {
        vsag::Resource resource(allocator.get(), nullptr);
        vsag::Engine engine(&resource);
        auto index1 = engine.CreateIndex(index_name, build_param).value();
        REQUIRE(index1->GetNumElements() == 0);
        if (index1->CheckFeature(vsag::SUPPORT_ESTIMATE_MEMORY)) {
            auto data_size = dataset->base_->GetNumElements();
            auto estimate_memory = index1->EstimateMemory(data_size);
            auto build_index = index1->Build(dataset->base_);
            auto real_memory = allocator->GetCurrentMemory();
            if (estimate_memory <= static_cast<uint64_t>(real_memory * 0.8) or
                estimate_memory >= static_cast<uint64_t>(real_memory * 1.2)) {
                WARN(fmt::format("estimate_memory({}) is not in range [{}, {}]",
                                 estimate_memory,
                                 static_cast<uint64_t>(real_memory * 0.8),
                                 static_cast<uint64_t>(real_memory * 1.2)));
            }

            REQUIRE(estimate_memory >= static_cast<uint64_t>(real_memory * 0.1));
            REQUIRE(estimate_memory <= static_cast<uint64_t>(real_memory * 5.0));
        }
    }
}

void
TestIndex::TestGetMemoryUsage(const std::string& index_name,
                              const std::string& build_param,
                              const TestDatasetPtr& dataset) {
    auto allocator = std::make_shared<fixtures::MemoryRecordAllocator>();
    {
        vsag::Resource resource(allocator.get(), nullptr);
        vsag::Engine engine(&resource);
        auto index1 = engine.CreateIndex(index_name, build_param).value();
        REQUIRE(index1->GetNumElements() == 0);
        auto index2 = vsag::Factory::CreateIndex(index_name, build_param).value();
        REQUIRE(index2->GetNumElements() == 0);
        fixtures::TempDir dir("index");
        auto path = dir.GenerateRandomFile();
        if (index1->CheckFeature(vsag::SUPPORT_GET_MEMORY_USAGE)) {
            auto data_size = dataset->base_->GetNumElements();
            auto build_index = index2->Build(dataset->base_);
            REQUIRE(build_index.has_value());
            std::ofstream outf(path, std::ios::binary);
            index2->Serialize(outf);
            outf.close();
            std::ifstream inf(path, std::ios::binary);
            index1->Deserialize(inf);
            auto real_memory = allocator->GetCurrentMemory();
            auto get_memory = index1->GetMemoryUsage();

            if (get_memory <= static_cast<uint64_t>(real_memory * 0.8) or
                get_memory >= static_cast<uint64_t>(real_memory * 1.2)) {
                WARN(fmt::format("get_memory({}) is not in range [{}, {}]",
                                 get_memory,
                                 static_cast<uint64_t>(real_memory * 0.8),
                                 static_cast<uint64_t>(real_memory * 1.2)));
            }

            REQUIRE(get_memory >= static_cast<uint64_t>(real_memory * 0.2));
            REQUIRE(get_memory <= static_cast<uint64_t>(real_memory * 3.2));
            inf.close();
        }
    }
}

void
TestIndex::TestCheckIdExist(const TestIndex::IndexPtr& index,
                            const TestDatasetPtr& dataset,
                            bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_CHECK_ID_EXIST)) {
        return;
    }
    auto data_count = dataset->base_->GetNumElements();
    auto* ids = dataset->base_->GetIds();
    int N = 10;
    for (int i = 0; i < N; ++i) {
        auto good_id = ids[random() % data_count];
        REQUIRE(index->CheckIdExist(good_id) == expected_success);
    }
    std::unordered_set<int64_t> exist_ids(ids, ids + data_count);
    int64_t bad_id = 97;
    while (N > 0) {
        for (; bad_id < data_count * N; ++bad_id) {
            if (exist_ids.count(bad_id) == 0) {
                break;
            }
        }
        REQUIRE(index->CheckIdExist(bad_id) == false);
        --N;
    }
}

template <class T>
std::string
create_attr_string(const std::string& name, const std::vector<T>& values) {
    if (values.size() == 1) {
        std::stringstream ss;
        if constexpr (std::is_same_v<T, std::string>) {
            ss << name << " = \"" << values[0] << "\"";
        } else {
            ss << name << " = " << std::to_string(values[0]);
        }
        return ss.str();
    }
    std::ostringstream oss;
    for (uint64_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            oss << "|";
        }
        if constexpr (std::is_same_v<T, std::string>) {
            oss << values[i];
        } else {
            oss << std::to_string(values[i]);
        }
    }
    return "multi_in(" + name + ", \"" + oss.str() + "\", \"|\")";
}

std::string
trans_attr_to_string(const vsag::Attribute& attr) {
    using namespace vsag;
    auto name = attr.name_;
    auto type = attr.GetValueType();
    if (type == AttrValueType::STRING) {
        const auto temp = dynamic_cast<const AttributeValue<std::string>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::UINT8) {
        const auto temp = dynamic_cast<const AttributeValue<uint8_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::UINT16) {
        const auto temp = dynamic_cast<const AttributeValue<uint16_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::UINT32) {
        const auto temp = dynamic_cast<const AttributeValue<uint32_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::UINT64) {
        const auto temp = dynamic_cast<const AttributeValue<uint64_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::INT8) {
        const auto temp = dynamic_cast<const AttributeValue<int8_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::INT16) {
        const auto temp = dynamic_cast<const AttributeValue<int16_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::INT32) {
        const auto temp = dynamic_cast<const AttributeValue<int32_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    } else if (type == AttrValueType::INT64) {
        const auto temp = dynamic_cast<const AttributeValue<int64_t>*>(&attr);
        auto values = temp->GetValue();
        return create_attr_string(name, values);
    }
    return "";
}

template <typename T>
static vsag::Attribute*
mock_value(const vsag::AttributeValue<T>* attr) {
    auto result = new vsag::AttributeValue<T>();
    result->name_ = attr->name_;
    auto old_values = std::unordered_set<T>(attr->GetValue().begin(), attr->GetValue().end());
    T random_new_value;
    if constexpr (std::is_same_v<T, std::string>) {
        random_new_value = "random_string";
    } else {
        random_new_value = static_cast<T>(rand());
        while (old_values.count(random_new_value)) {
            random_new_value = static_cast<T>(rand());
        }
    }
    result->GetValue().emplace_back(random_new_value);
    return result;
}

static void
mock_attrset(vsag::Attribute& attr, vsag::AttributeSet& old_attrs, vsag::AttributeSet& new_attrs) {
    using namespace vsag;
    old_attrs.attrs_.emplace_back(&attr);
    auto name = attr.name_;
    auto type = attr.GetValueType();
    if (type == AttrValueType::STRING) {
        const auto temp = dynamic_cast<const AttributeValue<std::string>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<std::string>(temp));
    } else if (type == AttrValueType::UINT8) {
        const auto temp = dynamic_cast<const AttributeValue<uint8_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<uint8_t>(temp));
    } else if (type == AttrValueType::UINT16) {
        const auto temp = dynamic_cast<const AttributeValue<uint16_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<uint16_t>(temp));
    } else if (type == AttrValueType::UINT32) {
        const auto temp = dynamic_cast<const AttributeValue<uint32_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<uint32_t>(temp));
    } else if (type == AttrValueType::UINT64) {
        const auto temp = dynamic_cast<const AttributeValue<uint64_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<uint64_t>(temp));
    } else if (type == AttrValueType::INT8) {
        const auto temp = dynamic_cast<const AttributeValue<int8_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<int8_t>(temp));
    } else if (type == AttrValueType::INT16) {
        const auto temp = dynamic_cast<const AttributeValue<int16_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<int16_t>(temp));
    } else if (type == AttrValueType::INT32) {
        const auto temp = dynamic_cast<const AttributeValue<int32_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<int32_t>(temp));
    } else if (type == AttrValueType::INT64) {
        const auto temp = dynamic_cast<const AttributeValue<int64_t>*>(&attr);
        new_attrs.attrs_.emplace_back(mock_value<int64_t>(temp));
    }
}

static void
release_attrset(vsag::AttributeSet& attrset) {
    for (auto* attr : attrset.attrs_) {
        delete attr;
    }
}

void
TestIndex::TestWithAttr(const IndexPtr& index,
                        const TestDatasetPtr& dataset,
                        const std::string& search_param,
                        bool with_update) {
    using namespace vsag;
    auto attrsets = dataset->base_->GetAttributeSets();
    auto* query_vec = dataset->base_->GetFloat32Vectors();
    auto count = std::min<int64_t>(dataset->base_->GetNumElements(), 200);
    auto dim = dataset->base_->GetDim();
    const auto* labels = dataset->base_->GetIds();

    for (int i = 0; i < count; ++i) {
        SearchRequest req;
        auto query = vsag::Dataset::Make();
        query->Float32Vectors(query_vec + i * dim)->Dim(dim)->Owner(false)->NumElements(1);
        auto attrset = attrsets[i].attrs_;
        int j1 = random() % attrset.size();
        int j2 = random() % attrset.size();
        req.topk_ = 10;
        req.filter_ = nullptr;
        req.params_str_ = search_param;
        req.enable_attribute_filter_ = true;
        req.query_ = query;
        req.attribute_filter_str_ = "(" + trans_attr_to_string(*attrset[j2]) + ") AND (" +
                                    trans_attr_to_string(*attrset[j1]) + ")";
        auto the_id = dataset->base_->GetIds()[i];
        auto result = index->SearchWithRequest(req);
        REQUIRE(result.has_value());
        auto ids = result.value()->GetIds();
        auto result_count = result.value()->GetDim();
        std::unordered_set<int64_t> sets(ids, ids + result_count);
        REQUIRE(sets.find(the_id) != sets.end());
        if (not with_update) {
            continue;
        }
        AttributeSet new_attrs;
        AttributeSet old_attrs;
        mock_attrset(*attrset[j1], old_attrs, new_attrs);

        auto test_func = [&]() -> void {
            auto result1 = index->SearchWithRequest(req);
            REQUIRE(result1.has_value());
            auto* ids = result1.value()->GetIds();
            auto result_count = result1.value()->GetDim();
            if (result_count != 0) {
                std::unordered_set<int64_t> sets1(ids, ids + result_count);
                REQUIRE(sets1.find(the_id) == sets1.end());
            }

            req.attribute_filter_str_ = "(" + trans_attr_to_string(*new_attrs.attrs_[0]) + ")";

            auto result2 = index->SearchWithRequest(req);
            REQUIRE(result2.has_value());
            ids = result2.value()->GetIds();
            result_count = result2.value()->GetDim();
            std::unordered_set<int64_t> sets1(ids, ids + result_count);
            REQUIRE(sets1.find(the_id) != sets1.end());
        };

        if (i % 2 == 0) {
            index->UpdateAttribute(labels[i], new_attrs, old_attrs);
            test_func();
        } else {
            index->UpdateAttribute(labels[i], new_attrs);
            test_func();
        }
        release_attrset(new_attrs);
    }
}

void
TestIndex::TestGetRawVectorByIds(const IndexPtr& index,
                                 const TestDatasetPtr& dataset,
                                 bool expected_success) {
    if (not index->CheckFeature(vsag::SUPPORT_GET_RAW_VECTOR_BY_IDS)) {
        return;
    }

    // get with not existed id
    {
        int64_t non_exist_id = -9999999;
        auto failed_res = index->GetRawVectorByIds(&non_exist_id, 1);
        REQUIRE(not failed_res.has_value());
    }

    auto count = static_cast<int64_t>(dataset->count_);
    vsag::IndexDetailInfo info;
    auto data_type = index->GetDetailDataByName("data_type", info).value()->GetDataScalarString();

    for (bool use_specific_allocator : {true, false}) {
        // specific_allocator
        vsag::DefaultAllocator allocator;
        void* mem = nullptr;

        // common case
        auto vectors = index->GetRawVectorByIds(dataset->base_->GetIds(), count);
        REQUIRE(vectors.has_value());

        if (use_specific_allocator) {
            vectors = index->GetRawVectorByIds(dataset->base_->GetIds(), count, &allocator);
            // create a delegate task (via Dataset Owner mechanism) to check and release the memory which allocated from the external allocator.
            vectors.value()->Owner(true, &allocator);
        }

        if (data_type == vsag::DATATYPE_SPARSE) {
            mem = (void*)vectors.value()->GetSparseVectors();
            for (int i = 0; i < count; i++) {
                // get single data
                auto single_dataset = vsag::Dataset::Make();
                auto sparse_vectors = vectors.value()->GetSparseVectors() + i;
                single_dataset->SparseVectors(sparse_vectors)->NumElements(1)->Owner(false);
                if (not expected_success) {
                    return;
                }

                // self distance
                auto dists_res =
                    index->CalDistanceById(single_dataset, dataset->base_->GetIds() + i, 1);
                REQUIRE(dists_res.has_value());
                auto dist = dists_res.value()->GetDistances()[0];

                // ground truth distance
                float gt_dist = 0;
                for (int j = 0; j < sparse_vectors->len_; j++) {
                    gt_dist += sparse_vectors->vals_[j] * sparse_vectors->vals_[j];
                }
                gt_dist = 1 - gt_dist;
                REQUIRE(std::abs(gt_dist - dist) < 1e-3);
            }
        } else if (data_type == vsag::DATATYPE_FLOAT32) {
            auto float_vectors = vectors.value()->GetFloat32Vectors();
            mem = (void*)float_vectors;
            auto dim = dataset->base_->GetDim();
            if (not expected_success) {
                return;
            }
            for (int i = 0; i < count; ++i) {
                REQUIRE(std::memcmp(float_vectors + i * dim,
                                    dataset->base_->GetFloat32Vectors() + i * dim,
                                    dim * sizeof(float)) == 0);
            }
        } else if (data_type == vsag::DATATYPE_INT8) {
            auto int8_vectors = vectors.value()->GetInt8Vectors();
            mem = (void*)int8_vectors;
            auto dim = dataset->base_->GetDim();
            if (not expected_success) {
                return;
            }
            for (int i = 0; i < count; ++i) {
                REQUIRE(std::memcmp(int8_vectors + i * dim,
                                    dataset->base_->GetInt8Vectors() + i * dim,
                                    dim) == 0);
            }
        } else {
            throw std::invalid_argument("Invalid data type: " + data_type);
        }
    }
}

void
TestIndex::TestExportIDs(const IndexPtr& index, const TestDatasetPtr& dataset) {
    if (not index->CheckFeature(vsag::SUPPORT_EXPORT_IDS)) {
        return;
    }
    auto result = index->ExportIDs();
    REQUIRE(result.has_value());
    const auto* ids = result.value()->GetIds();
    auto num_element = result.value()->GetNumElements();
    REQUIRE(num_element == dataset->base_->GetNumElements());
    auto* origin_ids = dataset->base_->GetIds();
    // check ids, no order
    std::unordered_set<int64_t> id_set(origin_ids, origin_ids + num_element);
    for (int64_t i = 0; i < num_element; ++i) {
        REQUIRE(id_set.find(ids[i]) != id_set.end());
    }
    std::unordered_set<int64_t> id_set2(ids, ids + num_element);
    REQUIRE(id_set2.size() == num_element);
}

template <typename T>
static void
compare_attr_value(const vsag::Attribute* attr1, const vsag::Attribute* attr2) {
    auto count = attr1->GetValueCount();
    auto* ptr1 = dynamic_cast<const vsag::AttributeValue<T>*>(attr1);
    auto* ptr2 = dynamic_cast<const vsag::AttributeValue<T>*>(attr2);
    const auto& temp_vec1 = ptr1->GetValue();
    const auto& temp_vec2 = ptr2->GetValue();
    std::unordered_set<T> temp_set1(temp_vec1.begin(), temp_vec1.end());
    std::unordered_set<T> temp_set2(temp_vec2.begin(), temp_vec2.end());
    REQUIRE(temp_set1 == temp_set2);
}

static void
compare_attr_set(const vsag::AttributeSet& attr1, const vsag::AttributeSet& attr2) {
    REQUIRE(attr1.attrs_.size() == attr2.attrs_.size());
    auto size = attr1.attrs_.size();
    auto temp_vec1 = attr1.attrs_;
    auto temp_vec2 = attr2.attrs_;
    std::sort(temp_vec1.begin(), temp_vec1.end(), [](const auto& a, const auto& b) {
        return a->name_ < b->name_;
    });
    std::sort(temp_vec2.begin(), temp_vec2.end(), [](const auto& a, const auto& b) {
        return a->name_ < b->name_;
    });
    for (int i = 0; i < size; ++i) {
        auto& attr = temp_vec1[i];
        auto& gt_attr = temp_vec2[i];
        REQUIRE(attr->name_ == gt_attr->name_);
        REQUIRE(attr->GetValueType() == gt_attr->GetValueType());
        if (attr->GetValueType() == vsag::AttrValueType::UINT64) {
            compare_attr_value<uint64_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::INT64) {
            compare_attr_value<int64_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::UINT32) {
            compare_attr_value<uint32_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::INT32) {
            compare_attr_value<int32_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::UINT16) {
            compare_attr_value<uint16_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::INT16) {
            compare_attr_value<int16_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::UINT8) {
            compare_attr_value<uint8_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::INT8) {
            compare_attr_value<int8_t>(attr, gt_attr);
        } else if (attr->GetValueType() == vsag::AttrValueType::STRING) {
            compare_attr_value<std::string>(attr, gt_attr);
        }
    }
}

void
TestIndex::TestGetDataById(const IndexPtr& index, const TestDatasetPtr& dataset) {
    if (not index->CheckFeature(vsag::SUPPORT_GET_DATA_BY_IDS)) {
        return;
    }
    auto result = index->GetDataByIds(dataset->base_->GetIds(), dataset->base_->GetNumElements());
    REQUIRE(result.has_value());
    auto data = result.value();
    REQUIRE(data->GetNumElements() == dataset->base_->GetNumElements());
    REQUIRE(data->GetDim() == dataset->base_->GetDim());
    // vectors
    auto float_vectors = data->GetFloat32Vectors();
    for (int i = 0; i < data->GetNumElements(); ++i) {
        REQUIRE(memcmp(float_vectors + i * data->GetDim(),
                       dataset->base_->GetFloat32Vectors() + i * data->GetDim(),
                       data->GetDim() * sizeof(float)) == 0);
    }
    // attributes
    auto attrs = data->GetAttributeSets();
    auto gt_attrs = dataset->base_->GetAttributeSets();
    for (int i = 0; i < data->GetNumElements(); ++i) {
        auto& attr = attrs[i];
        auto& gt_attr = gt_attrs[i];
        compare_attr_set(attr, gt_attr);
    }
}

void
TestIndex::TestIndexStatus(const IndexPtr& index) {
    auto set_result = index->SetImmutable();
    if (not set_result.has_value()) {
        return;
    }
    REQUIRE_FALSE(index->Train(nullptr));
    REQUIRE_FALSE(index->Build(nullptr));
    REQUIRE_FALSE(index->Add(nullptr));
    std::ifstream inf;
    REQUIRE_FALSE(index->Deserialize(inf));
    REQUIRE_FALSE(index->Remove(0));
    std::vector<vsag::MergeUnit> merge_units;
    REQUIRE_FALSE(index->Merge(merge_units));
    vsag::AttributeSet new_attrs;
    REQUIRE_FALSE(index->UpdateAttribute(0, new_attrs));
    REQUIRE_FALSE(index->UpdateAttribute(0, new_attrs, new_attrs));
    REQUIRE_FALSE(index->UpdateId(0, 0));
    REQUIRE_FALSE(index->UpdateVector(0, nullptr, false));
}

void
TestIndex::TestGetDataByIdWithFlag(const IndexPtr& index, const TestDatasetPtr& dataset) {
}

void
TestIndex::TestIndexDetailData(const IndexPtr& index) {
    auto infos = index->GetIndexDetailInfos();
    REQUIRE(infos.has_value());
    const auto& detail_infos = infos.value();
    REQUIRE(detail_infos.size() > 0);
    for (const auto& info : detail_infos) {
        vsag::IndexDetailInfo r_info;
        auto detail_data_value = index->GetDetailDataByName(info.name, r_info);
        REQUIRE(info.name == r_info.name);
        REQUIRE(info.type == r_info.type);
        REQUIRE(info.description == r_info.description);

        REQUIRE(detail_data_value.has_value());
        auto detail_data = detail_data_value.value();
        if (info.type == vsag::IndexDetailDataType::TYPE_SCALAR_INT64) {
            REQUIRE_NOTHROW(detail_data->GetDataScalarInt64());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_SCALAR_DOUBLE) {
            REQUIRE_NOTHROW(detail_data->GetDataScalarDouble());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_SCALAR_STRING) {
            REQUIRE_NOTHROW(detail_data->GetDataScalarString());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_SCALAR_BOOL) {
            REQUIRE_NOTHROW(detail_data->GetDataScalarBool());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_1DArray_INT64) {
            REQUIRE_NOTHROW(detail_data->GetData1DArrayInt64());
        } else if (info.type == vsag::IndexDetailDataType::TYPE_2DArray_INT64) {
            REQUIRE_NOTHROW(detail_data->GetData2DArrayInt64());
        } else {
            REQUIRE(false);
        }
    }
}

}  // namespace fixtures
