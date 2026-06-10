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

#include "disksindi_parameter.h"

#include "inner_string_params.h"
#include "unittest.h"

using namespace vsag;

TEST_CASE("DiskSINDI default rerank io uses block memory io", "[ut][DiskSINDIParameter]") {
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

    auto param = std::make_shared<vsag::DiskSINDIParameter>();
    param->FromJson(vsag::JsonType::Parse(param_str));

    REQUIRE(param->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_BLOCK_MEMORY_IO);
}

TEST_CASE("DiskSINDI rerank memory io uses block memory io", "[ut][DiskSINDIParameter]") {
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

    auto param = std::make_shared<vsag::DiskSINDIParameter>();
    param->FromJson(vsag::JsonType::Parse(param_str));

    REQUIRE(param->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_BLOCK_MEMORY_IO);
}

TEST_CASE("DiskSINDI rerank io derives file path", "[ut][DiskSINDIParameter]") {
    auto param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "term_io": {
            "type": "mmap_io",
            "file_path": "/tmp/disksindi.index"
        },
        "rerank_io": {
            "type": "mmap_io"
        }
    })";

    auto param = std::make_shared<vsag::DiskSINDIParameter>();
    param->FromJson(vsag::JsonType::Parse(param_str));

    REQUIRE(param->rerank_io_parameter->GetTypeName() == IO_TYPE_VALUE_MMAP_IO);
    REQUIRE(param->rerank_io_parameter->ToJson()[IO_FILE_PATH_KEY].GetString() ==
            "/tmp/disksindi.index.rerank");
}

TEST_CASE("DiskSINDI parameter compatibility ignores io type", "[ut][DiskSINDIParameter]") {
    auto mmap_param_str = R"({
        "term_id_limit": 30109,
        "window_size": 60000,
        "doc_prune_ratio": 0.4,
        "use_quantization": true,
        "use_reorder": true,
        "avg_doc_term_length": 126,
        "term_io": {
            "type": "mmap_io",
            "file_path": "/tmp/disksindi.term.index"
        },
        "rerank_io": {
            "type": "mmap_io",
            "file_path": "/tmp/disksindi.rerank.index"
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
            "file_path": "/tmp/disksindi.term.index"
        },
        "rerank_io": {
            "type": "async_io",
            "file_path": "/tmp/disksindi.rerank.index"
        }
    })";

    auto mmap_param = std::make_shared<vsag::DiskSINDIParameter>();
    mmap_param->FromJson(vsag::JsonType::Parse(mmap_param_str));
    auto async_param = std::make_shared<vsag::DiskSINDIParameter>();
    async_param->FromJson(vsag::JsonType::Parse(async_param_str));

    REQUIRE(async_param->CheckCompatibility(mmap_param));
    REQUIRE(mmap_param->CheckCompatibility(async_param));
}

TEST_CASE("DiskSINDI term io rejects memory io", "[ut][DiskSINDIParameter]") {
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
            "file_path": "/tmp/disksindi.rerank.index"
        }
    })";

    auto param = std::make_shared<vsag::DiskSINDIParameter>();
    REQUIRE_THROWS_WITH(
        param->FromJson(vsag::JsonType::Parse(param_str)),
        Catch::Matchers::ContainsSubstring("DiskSINDI term_io does not support memory_io"));
}
