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

#include "inner_string_params.h"
#include "unittest.h"

using namespace vsag;

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

TEST_CASE("SINDIV2 rerank layout accepts only top terms signature", "[ut][SINDIV2Parameter]") {
    auto param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "rerank_layout": "top_terms_signature",
        "rerank_layout_top_terms": 8,
        "term_io": {
            "type": "reader_io"
        }
    })";

    auto param = std::make_shared<vsag::SINDIV2Parameter>();
    param->FromJson(vsag::JsonType::Parse(param_str));

    REQUIRE(param->rerank_layout == "top_terms_signature");
    REQUIRE(param->rerank_layout_top_terms == 8);
    REQUIRE(param->ToJson()["rerank_layout"].GetString() == "top_terms_signature");
    REQUIRE(param->ToJson()["rerank_layout_top_terms"].GetInt() == 8);

    auto no_reorder = vsag::JsonType::Parse(param_str);
    no_reorder[USE_REORDER_KEY].SetBool(false);
    REQUIRE_THROWS_WITH(
        param->FromJson(no_reorder),
        Catch::Matchers::ContainsSubstring("SINDIV2 rerank_layout requires use_reorder=true"));

    for (const auto* layout : {"random", "minhash", "simhash"}) {
        auto unsupported = vsag::JsonType::Parse(param_str);
        unsupported["rerank_layout"].SetString(layout);
        REQUIRE_THROWS_WITH(
            param->FromJson(unsupported),
            Catch::Matchers::ContainsSubstring("unsupported SINDIV2 rerank_layout"));
    }
}
