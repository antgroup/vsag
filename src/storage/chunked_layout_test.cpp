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

#include "chunked_layout.h"

#include <limits>
#include <nlohmann/json.hpp>
#include <string>

#include "unittest.h"
#include "vsag_exception.h"

TEST_CASE("ChunkedLayout JSON Round Trip", "[ut][chunked_layout]") {
    vsag::ChunkedLayout layout;
    layout.codec_ = "zstd";
    layout.chunk_size_ = 128;

    vsag::ComponentLayout whole;
    whole.name = "label_table";
    whole.granularity = vsag::ComponentGranularity::Whole;
    whole.offset = 0;
    whole.compressed_size = 1024;
    whole.logical_size = 4096;
    layout.components_.push_back(whole);

    vsag::ComponentLayout chunked;
    chunked.name = "base_codes";
    chunked.granularity = vsag::ComponentGranularity::Byte;
    chunked.head_offset = 1024;
    chunked.head_size = 20;
    chunked.io_size = 300;
    chunked.chunks.push_back(vsag::ChunkRecord{1044, 100});
    chunked.chunks.push_back(vsag::ChunkRecord{1144, 99});
    chunked.chunks.push_back(vsag::ChunkRecord{1243, 30});
    chunked.tail_offset = 1273;
    chunked.tail_size = 55;
    layout.components_.push_back(chunked);

    auto json = layout.ToJson();
    auto parsed = vsag::ChunkedLayout::FromJson(json);

    REQUIRE(parsed.codec_ == "zstd");
    REQUIRE(parsed.chunk_size_ == 128);
    REQUIRE(parsed.components_.size() == 2);

    const auto* parsed_whole = parsed.FindComponent("label_table");
    REQUIRE(parsed_whole != nullptr);
    REQUIRE(parsed_whole->granularity == vsag::ComponentGranularity::Whole);
    REQUIRE(parsed_whole->offset == 0);
    REQUIRE(parsed_whole->compressed_size == 1024);
    REQUIRE(parsed_whole->logical_size == 4096);

    const auto* parsed_chunked = parsed.FindComponent("base_codes");
    REQUIRE(parsed_chunked != nullptr);
    REQUIRE(parsed_chunked->granularity == vsag::ComponentGranularity::Byte);
    REQUIRE(parsed_chunked->head_offset == 1024);
    REQUIRE(parsed_chunked->head_size == 20);
    REQUIRE(parsed_chunked->io_size == 300);
    REQUIRE(parsed_chunked->chunks.size() == 3);
    REQUIRE(parsed_chunked->chunks[1].offset == 1144);
    REQUIRE(parsed_chunked->chunks[1].compressed_size == 99);
    REQUIRE(parsed_chunked->tail_offset == 1273);
    REQUIRE(parsed_chunked->tail_size == 55);

    REQUIRE(parsed.FindComponent("not_exist") == nullptr);
}

TEST_CASE("ChunkedLayout Unsupported Version", "[ut][chunked_layout]") {
    vsag::ChunkedLayout layout;
    auto json = layout.ToJson();
    json.GetInnerJson()->at("version") = 999;
    REQUIRE_THROWS_AS(vsag::ChunkedLayout::FromJson(json), vsag::VsagException);
}

TEST_CASE("ChunkedLayout Validate Accepts A Well Formed Layout", "[ut][chunked_layout]") {
    vsag::ChunkedLayout layout;
    layout.codec_ = "none";
    layout.chunk_size_ = 128;

    vsag::ComponentLayout whole;
    whole.name = "label_table";
    whole.granularity = vsag::ComponentGranularity::Whole;
    whole.offset = 0;
    whole.compressed_size = 100;
    whole.logical_size = 100;
    layout.components_.push_back(whole);

    // head [100, 120), chunks 128 + 128 + 44 from 120, tail right after
    vsag::ComponentLayout chunked;
    chunked.name = "base_codes";
    chunked.granularity = vsag::ComponentGranularity::Byte;
    chunked.head_offset = 100;
    chunked.head_size = 20;
    chunked.io_size = 300;
    chunked.chunks.push_back(vsag::ChunkRecord{120, 128});
    chunked.chunks.push_back(vsag::ChunkRecord{248, 128});
    chunked.chunks.push_back(vsag::ChunkRecord{376, 44});
    chunked.tail_offset = 420;
    chunked.tail_size = 30;
    layout.components_.push_back(chunked);

    REQUIRE_NOTHROW(layout.Validate(450));
}

TEST_CASE("ChunkedLayout Validate Rejects Tampered Layouts", "[ut][chunked_layout]") {
    // a valid uncompressed single-component base that each section corrupts
    auto make_base = []() {
        vsag::ChunkedLayout layout;
        layout.codec_ = "none";
        layout.chunk_size_ = 128;
        vsag::ComponentLayout whole;
        whole.name = "label_table";
        whole.granularity = vsag::ComponentGranularity::Whole;
        whole.offset = 0;
        whole.compressed_size = 100;
        whole.logical_size = 100;
        layout.components_.push_back(whole);
        return layout;
    };

    auto make_whole = [](const std::string& name, uint64_t offset, uint64_t size) {
        vsag::ComponentLayout comp;
        comp.name = name;
        comp.granularity = vsag::ComponentGranularity::Whole;
        comp.offset = offset;
        comp.compressed_size = size;
        comp.logical_size = size;
        return comp;
    };

    SECTION("duplicate component name") {
        // two frames under one name would let two whole-component tasks
        // deserialize into the same index member concurrently
        auto layout = make_base();
        layout.components_.push_back(make_whole("label_table", 100, 50));
        REQUIRE_THROWS_AS(layout.Validate(1024), vsag::VsagException);
    }

    SECTION("whole frame past body end") {
        auto layout = make_base();
        layout.components_[0].compressed_size = 200;
        layout.components_[0].logical_size = 200;
        REQUIRE_THROWS_AS(layout.Validate(150), vsag::VsagException);
    }

    SECTION("offset plus size overflows") {
        auto layout = make_base();
        layout.components_[0].offset = std::numeric_limits<uint64_t>::max() - 10;
        layout.components_[0].compressed_size = 100;
        layout.components_[0].logical_size = 100;
        REQUIRE_THROWS_AS(layout.Validate(1024), vsag::VsagException);
    }

    SECTION("overlapping frames") {
        auto layout = make_base();
        layout.components_.push_back(make_whole("base_codes", 50, 100));
        REQUIRE_THROWS_AS(layout.Validate(1024), vsag::VsagException);
    }

    SECTION("head extends into the tail of the same component") {
        // head and tail are separate frame entries, which is exactly why the
        // exact-tiling check catches a head_size tampered to reach past
        // tail_offset: the two frames then overlap after the sort
        vsag::ChunkedLayout layout;
        layout.codec_ = "none";
        layout.chunk_size_ = 128;
        vsag::ComponentLayout chunked;
        chunked.name = "base_codes";
        chunked.granularity = vsag::ComponentGranularity::Byte;
        chunked.head_offset = 0;
        chunked.head_size = 40;  // tampered: reaches past tail_offset below
        chunked.io_size = 128;
        chunked.chunks.push_back(vsag::ChunkRecord{32, 128});
        chunked.tail_offset = 32;
        chunked.tail_size = 8;
        layout.components_.push_back(chunked);
        REQUIRE_THROWS_AS(layout.Validate(160), vsag::VsagException);
    }

    SECTION("uncompressed size mismatch") {
        auto layout = make_base();
        layout.components_[0].logical_size = 200;
        REQUIRE_THROWS_AS(layout.Validate(1024), vsag::VsagException);
    }

    SECTION("chunk count mismatch") {
        vsag::ChunkedLayout layout;
        layout.codec_ = "none";
        layout.chunk_size_ = 128;
        vsag::ComponentLayout chunked;
        chunked.name = "base_codes";
        chunked.granularity = vsag::ComponentGranularity::Byte;
        chunked.io_size = 300;  // needs 3 chunks
        chunked.chunks.push_back(vsag::ChunkRecord{0, 128});
        layout.components_.push_back(chunked);
        REQUIRE_THROWS_AS(layout.Validate(1024), vsag::VsagException);
    }

    SECTION("uncompressed chunk size mismatch") {
        vsag::ChunkedLayout layout;
        layout.codec_ = "none";
        layout.chunk_size_ = 128;
        vsag::ComponentLayout chunked;
        chunked.name = "base_codes";
        chunked.granularity = vsag::ComponentGranularity::Byte;
        chunked.io_size = 256;
        chunked.chunks.push_back(vsag::ChunkRecord{0, 128});
        chunked.chunks.push_back(vsag::ChunkRecord{128, 100});  // must be 128
        layout.components_.push_back(chunked);
        REQUIRE_THROWS_AS(layout.Validate(1024), vsag::VsagException);
    }

    SECTION("zero chunk size with data") {
        vsag::ChunkedLayout layout;
        layout.codec_ = "none";
        layout.chunk_size_ = 0;
        vsag::ComponentLayout chunked;
        chunked.name = "base_codes";
        chunked.granularity = vsag::ComponentGranularity::Byte;
        chunked.io_size = 128;
        layout.components_.push_back(chunked);
        REQUIRE_THROWS_AS(layout.Validate(1024), vsag::VsagException);
    }

    SECTION("zero chunk size cannot smuggle an oversized io size") {
        // zeroing chunk_size to dodge the round-up overflow guard does not
        // work: the io-size-without-chunk-size check runs before it
        vsag::ChunkedLayout layout;
        layout.codec_ = "none";
        layout.chunk_size_ = 0;
        vsag::ComponentLayout chunked;
        chunked.name = "base_codes";
        chunked.granularity = vsag::ComponentGranularity::Byte;
        chunked.io_size = std::numeric_limits<uint64_t>::max();
        layout.components_.push_back(chunked);
        REQUIRE_THROWS_AS(layout.Validate(1024), vsag::VsagException);
    }

    SECTION("io size overflows the chunk count round-up") {
        vsag::ChunkedLayout layout;
        layout.codec_ = "none";
        layout.chunk_size_ = 128;
        vsag::ComponentLayout chunked;
        chunked.name = "base_codes";
        chunked.granularity = vsag::ComponentGranularity::Byte;
        chunked.io_size = std::numeric_limits<uint64_t>::max();
        layout.components_.push_back(chunked);
        REQUIRE_THROWS_AS(layout.Validate(1024), vsag::VsagException);
    }

    SECTION("gap between frames") {
        // [0, 100) then [150, 200) leaves [100, 150) unaccounted for
        auto layout = make_base();
        layout.components_.push_back(make_whole("base_codes", 150, 50));
        REQUIRE_THROWS_AS(layout.Validate(200), vsag::VsagException);
    }

    SECTION("frames stop short of the body end") {
        auto layout = make_base();
        REQUIRE_THROWS_AS(layout.Validate(200), vsag::VsagException);
    }

    SECTION("no frames but a non-empty body") {
        vsag::ChunkedLayout layout;
        layout.codec_ = "none";
        layout.chunk_size_ = 128;
        REQUIRE_THROWS_AS(layout.Validate(200), vsag::VsagException);
    }
}

TEST_CASE("ChunkedLayout Malformed Json", "[ut][chunked_layout]") {
    // a structurally broken layout must surface as the same stable error
    // class as any other corrupted binary input, not as a raw json error
    vsag::ChunkedLayout layout;
    vsag::ComponentLayout whole;
    whole.name = "label_table";
    whole.granularity = vsag::ComponentGranularity::Whole;
    whole.offset = 0;
    whole.compressed_size = 128;
    whole.logical_size = 256;
    layout.components_.push_back(whole);

    SECTION("missing top-level key") {
        auto json = layout.ToJson();
        json.GetInnerJson()->erase("chunk_size");
        REQUIRE_THROWS_AS(vsag::ChunkedLayout::FromJson(json), vsag::VsagException);
    }

    SECTION("missing component field") {
        auto json = layout.ToJson();
        json.GetInnerJson()->at("components").at(0).erase("csize");
        REQUIRE_THROWS_AS(vsag::ChunkedLayout::FromJson(json), vsag::VsagException);
    }

    SECTION("wrong field type") {
        auto json = layout.ToJson();
        json.GetInnerJson()->at("components").at(0).at("offset") = "not_a_number";
        REQUIRE_THROWS_AS(vsag::ChunkedLayout::FromJson(json), vsag::VsagException);
    }

    SECTION("unknown component type") {
        // a granularity tag from a newer format must be rejected, never
        // parsed as one of the known granularities
        auto json = layout.ToJson();
        json.GetInnerJson()->at("components").at(0).at("type") = "segmented";
        REQUIRE_THROWS_AS(vsag::ChunkedLayout::FromJson(json), vsag::VsagException);
    }
}
