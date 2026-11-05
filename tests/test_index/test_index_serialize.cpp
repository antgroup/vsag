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
TestIndex::TestSerializeFile(const IndexPtr& index_from,
                             const IndexPtr& index_to,
                             const TestDatasetPtr& dataset,
                             const std::string& search_param,
                             bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_FILE) or
        not index_to->CheckFeature(vsag::SUPPORT_DESERIALIZE_FILE)) {
        return;
    }
    auto dir = fixtures::TempDir("serialize");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path, std::ios::out | std::ios::binary);
    auto serialize_index = index_from->Serialize(outfile);
    REQUIRE(serialize_index.has_value() == expected_success);
    outfile.close();

    std::ifstream infile(path, std::ios::in | std::ios::binary);
    auto deserialize_index = index_to->Deserialize(infile);
    REQUIRE(deserialize_index.has_value() == expected_success);
    infile.close();
    if (index_to->GetNumElements() == 0) {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(std::abs(res_from.value()->GetDistances()[j] -
                             res_to.value()->GetDistances()[j]) <= 2e-6);
        }
    }
}

void
TestIndex::TestSerializeBinarySet(const IndexPtr& index_from,
                                  const IndexPtr& index_to,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_BINARY_SET) or
        not index_to->CheckFeature(vsag::SUPPORT_DESERIALIZE_BINARY_SET)) {
        return;
    }
    auto serialize_binary = index_from->Serialize();
    REQUIRE(serialize_binary.has_value() == expected_success);

    auto deserialize_index = index_to->Deserialize(serialize_binary.value());
    REQUIRE(deserialize_index.has_value() == expected_success);
    if (index_to->GetNumElements() == 0) {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestSerializeReaderSet(const IndexPtr& index_from,
                                  const IndexPtr& index_to,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  const std::string& index_name,
                                  bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_BINARY_SET) or
        not index_to->CheckFeature(vsag::SUPPORT_DESERIALIZE_READER_SET)) {
        return;
    }
    vsag::ReaderSet rs;
    auto serialize_binary = index_from->Serialize();
    REQUIRE(serialize_binary.has_value() == expected_success);
    auto binary_set = serialize_binary.value();
    for (const auto& key : binary_set.GetKeys()) {
        rs.Set(key, std::make_shared<TestReader>(binary_set.Get(key)));
    }
    REQUIRE(rs.Get("this_is_a_wrong_key") == nullptr);
    auto deserialize_index = index_to->Deserialize(rs);
    REQUIRE(deserialize_index.has_value() == expected_success);
    if (index_to->GetNumElements() == 0) {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestSerializeWriteFunc(const IndexPtr& index_from,
                                  const IndexPtr& index_to,
                                  const TestDatasetPtr& dataset,
                                  const std::string& search_param,
                                  bool expected_success) {
    if (not index_from->CheckFeature(vsag::SUPPORT_SERIALIZE_WRITE_FUNC)) {
        return;
    }
    auto dir = fixtures::TempDir("serialize");
    auto path = dir.GenerateRandomFile();
    std::ofstream outfile(path, std::ios::out | std::ios::binary);
    vsag::WriteFuncType write_func =
        [&outfile](vsag::OffsetType offset, vsag::SizeType size, const void* data) -> void {
        outfile.seekp(offset);
        outfile.write(reinterpret_cast<const char*>(data), size);
    };
    auto serialize_index = index_from->Serialize(write_func);
    REQUIRE(serialize_index.has_value() == expected_success);
    outfile.close();

    std::ifstream infile(path, std::ios::in | std::ios::binary);
    auto deserialize_index = index_to->Deserialize(infile);
    REQUIRE(deserialize_index.has_value() == expected_success);
    infile.close();
    if (index_to->GetNumElements() == 0) {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index_from->KnnSearch(query, topk, search_param);
        auto res_to = index_to->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestClone(const TestIndex::IndexPtr& index,
                     const TestDatasetPtr& dataset,
                     const std::string& search_param) {
    if (not index->CheckFeature(vsag::SUPPORT_CLONE)) {
        return;
    }
    auto index_clone_result = index->Clone();
    REQUIRE(index_clone_result.has_value() == true);
    auto& index_clone = index_clone_result.value();

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto topk = 10;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res_from = index->KnnSearch(query, topk, search_param);
        auto res_to = index_clone->KnnSearch(query, topk, search_param);
        REQUIRE(res_from.has_value());
        REQUIRE(res_to.has_value());
        REQUIRE(res_from.value()->GetDim() == res_to.value()->GetDim());
        int64_t result_count = res_from.value()->GetDim();
        for (int64_t j = 0; j < result_count; ++j) {
            REQUIRE(res_to.value()->GetIds()[j] == res_from.value()->GetIds()[j]);
        }
    }
}

void
TestIndex::TestExportModel(const TestIndex::IndexPtr& index,
                           const TestIndex::IndexPtr& index2,
                           const TestDatasetPtr& dataset,
                           const std::string& search_param) {
    if (not index->CheckFeature(vsag::SUPPORT_EXPORT_MODEL)) {
        return;
    }
    auto index_model_result = index->ExportModel();
    REQUIRE(index_model_result.has_value() == true);
    auto index_model = index_model_result.value();
    fixtures::test_serializion_file(*index_model, *index2, "export_model_test");
    index_model = index2;
    tl::expected<std::vector<int64_t>, vsag::Error> add_index;
    if (index->CheckFeature(vsag::SUPPORT_ADD_AFTER_BUILD)) {
        add_index = index_model->Add(dataset->base_);
        REQUIRE(add_index.has_value());
    } else if (index->CheckFeature(vsag::SUPPORT_BUILD)) {
        add_index = index_model->Build(dataset->base_);
        REQUIRE(add_index.has_value());
    } else {
        return;
    }

    const auto& queries = dataset->query_;
    auto query_count = queries->GetNumElements();
    auto dim = queries->GetDim();
    auto gts = dataset->ground_truth_;
    auto gt_topK = dataset->top_k;
    float recall1 = 0.0F;
    float recall2 = 0.0F;
    auto topk = gt_topK;
    for (auto i = 0; i < query_count; ++i) {
        auto query = get_one_query(queries, i);
        auto res1 = index->KnnSearch(query, topk, search_param);
        REQUIRE(res1.has_value());
        auto result1 = res1.value()->GetIds();
        auto gt = gts->GetIds() + gt_topK * i;
        auto val = Intersection(gt, gt_topK, result1, topk);
        recall1 += static_cast<float>(val) / static_cast<float>(gt_topK);

        auto res2 = index_model->KnnSearch(query, topk, search_param);
        REQUIRE(res2.has_value());
        auto result2 = res2.value()->GetIds();
        val = Intersection(gt, gt_topK, result2, topk);
        recall2 += static_cast<float>(val) / static_cast<float>(gt_topK);
    }

    REQUIRE(std::abs(recall1 - recall2) < 0.01F * query_count);
}

}  // namespace fixtures
