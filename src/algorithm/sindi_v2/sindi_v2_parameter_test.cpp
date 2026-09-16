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

#include "sindi_v2_parameter.h"

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "inner_string_params.h"
#include "sindi_v2.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("SINDIV2 rejects unknown external parameters", "[ut][SINDIV2Parameter]") {
    IndexCommonParam common_param;
    REQUIRE_NOTHROW(SINDIV2::CheckAndMappingExternalParam(
        JsonType::Parse(R"({"term_id_limit": 50000000, "window_size": 50000})"), common_param));
    REQUIRE_THROWS_WITH(SINDIV2::CheckAndMappingExternalParam(
                            JsonType::Parse(R"({"unknown_param": true})"), common_param),
                        Catch::Matchers::ContainsSubstring("invalid config param: unknown_param"));
}

TEST_CASE("SINDIV2 term prune parameter validation", "[ut][SINDIV2Parameter]") {
    auto parse = [](const std::string& parameters) {
        SINDIV2SearchParameter search_parameter;
        search_parameter.FromJson(JsonType::Parse(parameters));
        return search_parameter;
    };

    const auto configured = parse(R"({
        "sindi_v2": {
            "query_prune_ratio": 0.99,
            "term_prune_ratio": 0.99,
            "term_retain_threshold": 4096
        }
    })");
    REQUIRE(configured.query_prune_ratio == 0.99F);
    REQUIRE(configured.term_prune_ratio == 0.99F);
    REQUIRE(configured.term_retain_threshold == 4096);
    const auto roundtrip = configured.ToJson();
    REQUIRE(roundtrip[INDEX_SINDI_V2][SPARSE_TERM_RETAIN_THRESHOLD].GetUint64() == 4096);
    REQUIRE_FALSE(roundtrip[INDEX_SINDI_V2].Contains("use_term_lists_heap_insert"));

    const auto legacy = parse(R"({
        "sindi_v2": {
            "query_prune_ratio": 0.2,
            "use_term_lists_heap_insert": false
        }
    })");
    REQUIRE(legacy.query_prune_ratio == 0.2F);
    REQUIRE_FALSE(legacy.ToJson()[INDEX_SINDI_V2].Contains("use_term_lists_heap_insert"));

    const auto explicit_zero =
        parse(R"({"sindi_v2": {"term_prune_ratio": 0.0, "term_retain_threshold": 0}})");
    REQUIRE(explicit_zero.term_prune_ratio == 0.0F);
    REQUIRE(explicit_zero.term_retain_threshold == 0);

    const auto old_threshold = parse(R"({"sindi_v2": {"term_prune_threshold": 10}})");
    REQUIRE(old_threshold.term_retain_threshold == DEFAULT_TERM_RETAIN_THRESHOLD);
    REQUIRE_FALSE(old_threshold.ToJson()[INDEX_SINDI_V2].Contains("term_prune_threshold"));

    REQUIRE_THROWS(parse(R"({"sindi_v2": {"query_prune_ratio": -0.1}})"));
    REQUIRE_THROWS(parse(R"({"sindi_v2": {"query_prune_ratio": 1.0}})"));
    REQUIRE_THROWS(parse(R"({"sindi_v2": {"term_prune_ratio": -0.1}})"));
    REQUIRE_THROWS(parse(R"({"sindi_v2": {"term_prune_ratio": 1.0}})"));
    REQUIRE_THROWS(parse(R"({"sindi_v2": {"term_retain_threshold": -1}})"));
    REQUIRE_THROWS(parse(R"({"sindi_v2": {"term_retain_threshold": 2.5}})"));
    REQUIRE_THROWS(parse(R"({"sindi_v2": {"term_retain_threshold": 18446744073709551616}})"));
}

TEST_CASE("SINDIV2 doc prune ratio boundaries", "[ut][SINDIV2Parameter]") {
    SINDIV2Parameter param;
    REQUIRE_NOTHROW(param.FromJson(JsonType::Parse(R"({"doc_prune_ratio": 0.99})")));
    REQUIRE(param.doc_prune_ratio == 0.99F);

    for (const auto& invalid_param :
         {R"({"doc_prune_ratio": -0.1})", R"({"doc_prune_ratio": 1.0})"}) {
        REQUIRE_THROWS(param.FromJson(JsonType::Parse(invalid_param)));
    }
}

TEST_CASE("SINDIV2 term_id_limit upper bound", "[ut][SINDIV2Parameter]") {
    auto valid_param = std::make_shared<SINDIV2Parameter>();
    REQUIRE_NOTHROW(valid_param->FromJson(JsonType::Parse(R"({
        "term_id_limit": 50000000,
        "window_size": 50000
    })")));
    REQUIRE(valid_param->term_id_limit == 50'000'000);

    auto invalid_param = std::make_shared<SINDIV2Parameter>();
    REQUIRE_THROWS(invalid_param->FromJson(JsonType::Parse(R"({
        "term_id_limit": 50000001,
        "window_size": 50000
    })")));
}

TEST_CASE("SINDIV2 unsigned parameters reject negative values", "[ut][SINDIV2Parameter]") {
    SINDIV2Parameter index_parameter;
    REQUIRE_THROWS(index_parameter.FromJson(JsonType::Parse(R"({"term_id_limit": -1})")));
    REQUIRE_THROWS(index_parameter.FromJson(JsonType::Parse(R"({"avg_doc_term_length": -1})")));

    SINDIV2SearchParameter search_parameter;
    REQUIRE_THROWS(
        search_parameter.FromJson(JsonType::Parse(R"({"sindi_v2": {"n_candidate": -1}})")));
}

TEST_CASE("SINDIV2 rejects unsupported or conflicting io", "[ut][SINDIV2Parameter]") {
    SINDIV2Parameter parameter;
    REQUIRE_THROWS_WITH(parameter.FromJson(JsonType::Parse(R"({
            "term_io": {"type": "block_memory_io"}
        })")),
                        Catch::Matchers::ContainsSubstring("unsupported SINDIV2 term_io type"));
    REQUIRE_NOTHROW(parameter.FromJson(JsonType::Parse(R"({
            "use_reorder": true,
            "term_io": {"type": "memory_io"},
            "rerank_io": {"type": "reader_io"}
        })")));
    REQUIRE(parameter.rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_READER_IO);
    REQUIRE_THROWS_WITH(parameter.FromJson(JsonType::Parse(R"({
            "use_reorder": true,
            "term_io": {"type": "buffer_io", "file_path": "/tmp/sindi-v2"},
            "rerank_io": {"type": "mmap_io", "file_path": "/tmp/sindi-v2"}
        })")),
                        Catch::Matchers::ContainsSubstring("must use different file_path values"));
}

TEST_CASE("SINDIV2 default rerank io uses block memory io", "[ut][SINDIV2Parameter]") {
    auto param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "term_io": {
            "type": "reader_io"
        }
    })";

    auto param = std::make_shared<vsag::SINDIV2Parameter>();
    param->FromJson(vsag::JsonType::Parse(param_str));

    REQUIRE(param->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_BLOCK_MEMORY_IO);
}

TEST_CASE("SINDIV2 rerank memory io uses block memory io", "[ut][SINDIV2Parameter]") {
    auto param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "term_io": {
            "type": "reader_io"
        },
        "rerank_io": {
            "type": "memory_io"
        }
    })";

    auto param = std::make_shared<vsag::SINDIV2Parameter>();
    param->FromJson(vsag::JsonType::Parse(param_str));

    REQUIRE(param->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_BLOCK_MEMORY_IO);
}

TEST_CASE("SINDIV2 rerank io derives file path", "[ut][SINDIV2Parameter]") {
    auto param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "term_io": {
            "type": "mmap_io",
            "file_path": "/tmp/sindi_v2.index"
        },
        "rerank_io": {
            "type": "mmap_io"
        }
    })";

    auto param = std::make_shared<vsag::SINDIV2Parameter>();
    param->FromJson(vsag::JsonType::Parse(param_str));

    REQUIRE(param->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_MMAP_IO);
    REQUIRE(param->rerank_io_parameter->ToJson()[IO_FILE_PATH_KEY].GetString() ==
            "/tmp/sindi_v2.index.rerank");
}

TEST_CASE("SINDIV2 parameter compatibility ignores io type", "[ut][SINDIV2Parameter]") {
    auto mmap_param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "term_io": {
            "type": "mmap_io",
            "file_path": "/tmp/sindi_v2.term.index"
        },
        "rerank_io": {
            "type": "mmap_io",
            "file_path": "/tmp/sindi_v2.rerank.index"
        }
    })";
    auto async_param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "term_io": {
            "type": "async_io",
            "file_path": "/tmp/sindi_v2.term.index"
        },
        "rerank_io": {
            "type": "async_io",
            "file_path": "/tmp/sindi_v2.rerank.index"
        }
    })";

    auto mmap_param = std::make_shared<vsag::SINDIV2Parameter>();
    mmap_param->FromJson(vsag::JsonType::Parse(mmap_param_str));
    auto async_param = std::make_shared<vsag::SINDIV2Parameter>();
    async_param->FromJson(vsag::JsonType::Parse(async_param_str));

    REQUIRE(async_param->CheckCompatibility(mmap_param));
    REQUIRE(mmap_param->CheckCompatibility(async_param));
}

TEST_CASE("SINDIV2 term io accepts memory io", "[ut][SINDIV2Parameter]") {
    auto param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "term_io": {
            "type": "memory_io"
        },
        "rerank_io": {
            "type": "mmap_io",
            "file_path": "/tmp/sindi_v2.rerank.index"
        }
    })";

    auto param = std::make_shared<vsag::SINDIV2Parameter>();
    REQUIRE_NOTHROW(param->FromJson(vsag::JsonType::Parse(param_str)));
    REQUIRE(param->term_io_parameter->GetTypeName() == IO_TYPE_VALUE_MEMORY_IO);
}

TEST_CASE("SINDIV2 immutable parameter participates in format compatibility",
          "[ut][SINDIV2Parameter]") {
    auto mutable_param = std::make_shared<vsag::SINDIV2Parameter>();
    mutable_param->FromJson(vsag::JsonType::Parse(R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "immutable": false
    })"));
    auto immutable_param = std::make_shared<vsag::SINDIV2Parameter>();
    immutable_param->FromJson(vsag::JsonType::Parse(R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "immutable": true
    })"));

    REQUIRE(immutable_param->immutable);
    REQUIRE(immutable_param->ToJson()[SPARSE_IMMUTABLE].GetBool());
    REQUIRE_FALSE(immutable_param->CheckCompatibility(mutable_param));
    REQUIRE_FALSE(mutable_param->CheckCompatibility(immutable_param));
}

TEST_CASE("SINDIV2 FP16 parameter roundtrip", "[ut][SINDIV2Parameter]") {
    auto parameter = std::make_shared<vsag::SINDIV2Parameter>();
    parameter->FromJson(vsag::JsonType::Parse(R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "use_quantization": "fp16"
    })"));

    REQUIRE(parameter->use_quantization);
    REQUIRE(parameter->sparse_value_quant_type == SparseValueQuantizationType::FP16);
    REQUIRE(parameter->ToJson()[USE_QUANTIZATION].GetString() == QUANTIZATION_TYPE_VALUE_FP16);

    auto restored = std::make_shared<vsag::SINDIV2Parameter>();
    restored->FromJson(parameter->ToJson());
    REQUIRE(restored->sparse_value_quant_type == SparseValueQuantizationType::FP16);
    REQUIRE(parameter->CheckCompatibility(restored));
}

TEST_CASE("SINDIV2 rerank layout uses the top terms count", "[ut][SINDIV2Parameter]") {
    auto param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "rerank_layout": 8,
        "term_io": {
            "type": "reader_io"
        }
    })";

    auto param = std::make_shared<vsag::SINDIV2Parameter>();
    param->FromJson(vsag::JsonType::Parse(param_str));

    REQUIRE(param->rerank_layout == 8);
    REQUIRE(param->ToJson()["rerank_layout"].GetInt() == 8);

    auto no_reorder = vsag::JsonType::Parse(param_str);
    no_reorder[USE_REORDER_KEY].SetBool(false);
    REQUIRE_THROWS_WITH(
        param->FromJson(no_reorder),
        Catch::Matchers::ContainsSubstring("SINDIV2 rerank_layout requires use_reorder=true"));

    auto non_integer = vsag::JsonType::Parse(param_str);
    non_integer["rerank_layout"].SetString("top_terms_signature");
    REQUIRE_THROWS_WITH(
        param->FromJson(non_integer),
        Catch::Matchers::ContainsSubstring("rerank_layout must be a non-negative integer"));

    auto negative = vsag::JsonType::Parse(param_str);
    negative["rerank_layout"].SetInt(-1);
    REQUIRE_THROWS_WITH(
        param->FromJson(negative),
        Catch::Matchers::ContainsSubstring("SINDIV2 rerank_layout must be in uint32 range"));

    auto disabled = vsag::JsonType::Parse(param_str);
    disabled["rerank_layout"].SetInt(0);
    REQUIRE_NOTHROW(param->FromJson(disabled));
    REQUIRE(param->rerank_layout == 0);
}

TEST_CASE("SINDIV2 DMQ parameter validation and compatibility", "[ut][SINDIV2Parameter]") {
    const auto dmq_json = JsonType::Parse(R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "use_reorder": true,
        "rerank_type": "dmq8",
        "dmq_shared_codebook_threshold": 2048,
        "term_io": {"type": "memory_io"},
        "rerank_io": {"type": "block_memory_io"}
    })");

    auto parameter = std::make_shared<SINDIV2Parameter>();
    REQUIRE_NOTHROW(parameter->FromJson(dmq_json));
    REQUIRE(parameter->rerank_type == SPARSE_RERANK_TYPE_DMQ8);
    REQUIRE(parameter->dmq_shared_codebook_threshold == 2048);
    REQUIRE(parameter->ToJson()[SPARSE_RERANK_TYPE].GetString() == SPARSE_RERANK_TYPE_DMQ8);
    REQUIRE(parameter->ToJson()[SPARSE_DMQ_SHARED_CODEBOOK_THRESHOLD].GetInt() == 2048);

    auto restored = std::make_shared<SINDIV2Parameter>();
    restored->FromJson(parameter->ToJson());
    REQUIRE(parameter->CheckCompatibility(restored));

    auto different_threshold = std::make_shared<SINDIV2Parameter>();
    auto different_threshold_json = dmq_json;
    different_threshold_json[SPARSE_DMQ_SHARED_CODEBOOK_THRESHOLD].SetInt(2049);
    different_threshold->FromJson(different_threshold_json);
    REQUIRE_FALSE(parameter->CheckCompatibility(different_threshold));

    auto no_reorder_json = dmq_json;
    no_reorder_json[USE_REORDER_KEY].SetBool(false);
    REQUIRE_THROWS_WITH(
        parameter->FromJson(no_reorder_json),
        Catch::Matchers::ContainsSubstring("rerank_type=dmq8 requires use_reorder=true"));

    auto incompatible_layout_json = dmq_json;
    incompatible_layout_json["rerank_layout"].SetInt(8);
    REQUIRE_THROWS_WITH(
        parameter->FromJson(incompatible_layout_json),
        Catch::Matchers::ContainsSubstring("rerank_type=dmq8 requires rerank_layout=0"));

    auto file_rerank_json = dmq_json;
    file_rerank_json["rerank_io"].SetJson(
        JsonType::Parse(R"({"type":"mmap_io","file_path":"/tmp/sindi_v2.dmq"})"));
    REQUIRE_THROWS_WITH(
        parameter->FromJson(file_rerank_json),
        Catch::Matchers::ContainsSubstring("rerank_type=dmq8 only supports block_memory_io"));
}
