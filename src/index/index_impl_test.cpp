
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

#include "index_impl.h"

#include <catch2/catch_test_macros.hpp>
#include <sstream>

#include "algorithm/hgraph.h"
#include "vsag/engine.h"

TEST_CASE("immutable index test", "[ut][index_impl]") {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = 128;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.allocator_ = vsag::Engine::CreateDefaultAllocator();
    auto build_parameter_json = R"(
        {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 100
        }
    )";

    vsag::JsonType hgraph_json;
    hgraph_json = vsag::JsonType::parse(build_parameter_json);
    auto index = std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);

    vsag::DatasetPtr dataset = vsag::Dataset::Make();
    vsag::BinarySet binary_set;
    std::vector<int64_t> base_tag_ids;
    std::vector<vsag::MergeUnit> merge_units;
    std::stringstream ss;

    auto result_immutable = index->SetImmutable();
    REQUIRE(result_immutable.has_value());
    // test SetImmutable Again
    auto result_immutable_again = index->SetImmutable();
    REQUIRE(result_immutable_again.has_value());

    auto result_build = index->Build(dataset);
    REQUIRE_FALSE(result_build.has_value());
    REQUIRE(result_build.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto result_train = index->Train(dataset);
    REQUIRE_FALSE(result_train.has_value());
    REQUIRE(result_train.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto result_continue_build = index->ContinueBuild(dataset, binary_set);
    REQUIRE_FALSE(result_continue_build.has_value());
    REQUIRE(result_continue_build.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto result_add = index->Add(dataset);
    REQUIRE_FALSE(result_add.has_value());
    REQUIRE(result_add.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto result_remove = index->Remove(0);
    REQUIRE_FALSE(result_remove.has_value());
    REQUIRE(result_remove.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto result_update_id = index->UpdateId(0, 0);
    REQUIRE_FALSE(result_update_id.has_value());
    REQUIRE(result_update_id.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto result_update_vector = index->UpdateVector(0, dataset);
    REQUIRE_FALSE(result_update_vector.has_value());
    REQUIRE(result_update_vector.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto result_pretrain = index->Pretrain(base_tag_ids, 0, "");
    REQUIRE_FALSE(result_pretrain.has_value());
    REQUIRE(result_pretrain.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto result_feedback = index->Feedback(dataset, 0, "");
    REQUIRE_FALSE(result_feedback.has_value());
    REQUIRE(result_feedback.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    auto result_merge = index->Merge(merge_units);
    REQUIRE_FALSE(result_merge.has_value());
    REQUIRE(result_merge.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
}
