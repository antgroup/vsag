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

#include <unistd.h>

#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstring>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "impl/thread_pool/default_thread_pool.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "storage/chunked_layout.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "unittest.h"
#include "vsag/deserialize_reader.h"
#include "vsag/serialize_writer.h"

namespace {

// plain writer: verbatim passthrough, codec "none"
class BufferSerializeWriter : public vsag::SerializeWriter {
public:
    void
    Write(const char* data, uint64_t size) override {
        buffer_.append(data, size);
    }

    std::string buffer_;
};

// test codec: each frame is stored as [payload size (8B)][payload], which is
// enough to exercise the compressed-frame paths without a real compressor
class FrameSerializeWriter : public vsag::SerializeWriter {
public:
    void
    Write(const char* data, uint64_t size) override {
        if (in_frame_) {
            frame_.append(data, size);
        } else {
            buffer_.append(data, size);
        }
    }

    [[nodiscard]] std::string
    GetCompressorName() const override {
        return "test";
    }

    void
    BeginCompressedFrame() override {
        in_frame_ = true;
        frame_.clear();
    }

    uint64_t
    EndCompressedFrame() override {
        in_frame_ = false;
        uint64_t payload_size = frame_.size();
        buffer_.append(reinterpret_cast<const char*>(&payload_size), sizeof(payload_size));
        buffer_.append(frame_);
        return sizeof(payload_size) + payload_size;
    }

    std::string buffer_;

private:
    bool in_frame_{false};
    std::string frame_;
};

// positioned reads over an in-memory file; no frame support
class PlainMemoryReader : public vsag::DeserializeReader {
public:
    explicit PlainMemoryReader(std::string buffer) : buffer_(std::move(buffer)) {
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return buffer_.size();
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        // called from worker threads: Catch2 assertions are not thread-safe;
        // bounds check written the way the base-class doc recommends, so the
        // wrap-around of offset + len cannot slip through
        if (len > buffer_.size() || offset > buffer_.size() - len) {
            throw std::out_of_range("read beyond the in-memory file");
        }
        std::memcpy(dest, buffer_.data() + offset, len);
    }

protected:
    std::string buffer_;
};

// decodes the [payload size (8B)][payload] frames of FrameSerializeWriter
class FrameMemoryReader : public PlainMemoryReader {
public:
    using PlainMemoryReader::PlainMemoryReader;

    void
    ReadDecompressed(uint64_t offset,
                     uint64_t compressed_size,
                     const std::function<void(std::istream&)>& consume) override {
        uint64_t payload_size = 0;
        if (compressed_size < sizeof(payload_size)) {
            throw std::invalid_argument("frame shorter than its header");
        }
        Read(offset, sizeof(payload_size), &payload_size);
        if (compressed_size != sizeof(payload_size) + payload_size) {
            throw std::invalid_argument("frame size does not match its header");
        }
        std::istringstream is(buffer_.substr(offset + sizeof(payload_size), payload_size),
                              std::ios::in | std::ios::binary);
        consume(is);
    }
};

// corrupts one frame payload by truncating / appending bytes after decoding
class CorruptingFrameReader : public FrameMemoryReader {
public:
    CorruptingFrameReader(std::string buffer, uint64_t victim_offset, int64_t delta)
        : FrameMemoryReader(std::move(buffer)), victim_offset_(victim_offset), delta_(delta) {
    }

    void
    ReadDecompressed(uint64_t offset,
                     uint64_t compressed_size,
                     const std::function<void(std::istream&)>& consume) override {
        if (offset != victim_offset_) {
            FrameMemoryReader::ReadDecompressed(offset, compressed_size, consume);
            return;
        }
        uint64_t payload_size = 0;
        Read(offset, sizeof(payload_size), &payload_size);
        auto payload = buffer_.substr(offset + sizeof(payload_size), payload_size);
        if (delta_ < 0) {
            const auto shrink = static_cast<size_t>(-delta_);
            payload.resize(shrink <= payload.size() ? payload.size() - shrink : 0);
        } else {
            payload.append(static_cast<size_t>(delta_), '\0');
        }
        std::istringstream is(payload, std::ios::in | std::ios::binary);
        consume(is);
    }

private:
    uint64_t victim_offset_;
    int64_t delta_;
};

// rewrite the chunked layout recorded in a serialized index footer, so a
// tampered layout can be fed to ParallelDeserialize
std::string
RewriteChunkedLayout(const std::string& buffer,
                     const std::function<void(nlohmann::json&)>& mutate) {
    auto read_func = [&buffer](uint64_t offset, uint64_t len, void* dest) {
        std::memcpy(dest, buffer.data() + offset, len);
    };
    vsag::ReadFuncStreamReader footer_reader(read_func, 0, buffer.size());
    auto footer = vsag::Footer::Parse(footer_reader);
    REQUIRE(footer != nullptr);

    auto metadata = footer->GetMetadata();
    auto layout_json = metadata->Get(vsag::CHUNKED_LAYOUT_KEY);
    mutate(*layout_json.GetInnerJson());
    metadata->Set(vsag::CHUNKED_LAYOUT_KEY, layout_json);

    std::ostringstream footer_out(std::ios::out | std::ios::binary);
    vsag::IOStreamWriter footer_writer(footer_out);
    footer->Write(footer_writer);

    return buffer.substr(0, buffer.size() - footer->Length()) + footer_out.str();
}

vsag::IndexCommonParam
MakeCommonParam(int64_t dim, std::shared_ptr<vsag::SafeThreadPool> thread_pool = nullptr) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    common_param.thread_pool_ = std::move(thread_pool);
    return common_param;
}

// a SafeThreadPool with a fixed worker count, for exercising the
// Resource-injected pool path of ParallelDeserialize
std::shared_ptr<vsag::SafeThreadPool>
MakeThreadPool(uint64_t threads) {
    return std::make_shared<vsag::SafeThreadPool>(new vsag::DefaultThreadPool(threads), true);
}

vsag::JsonType
MakeHGraphJson() {
    return vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1
    })");
}

vsag::JsonType
MakeSq8ReorderHGraphJson() {
    auto hgraph_json = MakeHGraphJson();
    hgraph_json["base_quantization_type"].SetString("sq8");
    hgraph_json["use_reorder"].SetBool(true);
    hgraph_json["precise_quantization_type"].SetString("fp32");
    return hgraph_json;
}

vsag::JsonType
MakeDedupHGraphJson() {
    auto hgraph_json = MakeHGraphJson();
    hgraph_json["support_duplicate"].SetBool(true);
    hgraph_json["deduplicate_storage"].SetBool(true);
    return hgraph_json;
}

// odescent graph + mci cliques so the serialized index carries the
// mci_cliques component
vsag::JsonType
MakeMciHGraphJson() {
    return vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "graph_type": "odescent",
        "max_degree": 8,
        "alpha": 1.2,
        "graph_iter_turn": 6,
        "neighbor_sample_rate": 0.3,
        "ef_construction": 32,
        "build_thread_count": 1,
        "mci_mcs": 8,
        "mci_clique_max": 4,
        "mci_incremental_clique_max": 4,
        "mci_alpha": 1.2
    })");
}

// sq8 + fp32 reorder over a specific io backend for all three chunked
// components; file-backed ios get distinct paths per tag so coexisting
// indexes never share the default file path
vsag::JsonType
MakeIoTypeHGraphJson(const std::string& io_type, const std::string& tag) {
    auto hgraph_json = MakeSq8ReorderHGraphJson();
    hgraph_json["base_io_type"].SetString(io_type);
    hgraph_json["graph_io_type"].SetString(io_type);
    hgraph_json["precise_io_type"].SetString(io_type);
    if (io_type != "block_memory_io" && io_type != "memory_io") {
        const std::string prefix =
            "/tmp/vsag_ut_iotype_" + std::to_string(::getpid()) + "_" + tag + "_" + io_type;
        hgraph_json["base_file_path"].SetString(prefix + "_base");
        hgraph_json["graph_file_path"].SetString(prefix + "_graph");
        hgraph_json["precise_file_path"].SetString(prefix + "_precise");
    }
    return hgraph_json;
}

// conjugate graph enabled so the serialized index carries the
// conjugate_graph component
vsag::JsonType
MakeConjugateHGraphJson() {
    auto hgraph_json = MakeHGraphJson();
    hgraph_json["use_conjugate_graph"].SetBool(true);
    return hgraph_json;
}

std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>
MakeHGraphIndex(const vsag::IndexCommonParam& common_param,
                const vsag::JsonType& hgraph_json = MakeHGraphJson()) {
    return std::make_shared<vsag::IndexImpl<vsag::HGraph>>(hgraph_json, common_param);
}

struct TestData {
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    int64_t dim{0};
    int64_t count{0};
};

TestData
MakeTestData(int64_t dim, int64_t count, bool with_duplicates = false) {
    TestData data;
    data.dim = dim;
    data.count = count;
    data.vectors.resize(dim * count);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& v : data.vectors) {
        v = dist(rng);
    }
    if (with_duplicates) {
        // every third vector duplicates its predecessor
        for (int64_t i = 2; i < count; i += 3) {
            std::copy(data.vectors.begin() + (i - 1) * dim,
                      data.vectors.begin() + i * dim,
                      data.vectors.begin() + i * dim);
        }
    }
    for (int64_t i = 0; i < count; ++i) {
        data.ids.push_back(i);
    }
    return data;
}

vsag::DatasetPtr
MakeDataset(TestData& data) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(data.count)
        ->Dim(data.dim)
        ->Ids(data.ids.data())
        ->Float32Vectors(data.vectors.data())
        ->Owner(false);
    return dataset;
}

// knn results of both indexes must match on every sampled base vector as query
void
RequireSameKnnResults(const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& expected,
                      const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& actual,
                      TestData& data) {
    const auto* params = R"({"hgraph": {"ef_search": 64}})";
    for (int64_t i = 0; i < data.count; i += 17) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(data.dim)
            ->Float32Vectors(data.vectors.data() + i * data.dim)
            ->Owner(false);
        auto expected_result = expected->KnnSearch(query, 5, params);
        auto actual_result = actual->KnnSearch(query, 5, params);
        REQUIRE(expected_result.has_value());
        REQUIRE(actual_result.has_value());
        REQUIRE(expected_result.value()->GetDim() == actual_result.value()->GetDim());
        for (int64_t k = 0; k < expected_result.value()->GetDim(); ++k) {
            REQUIRE(expected_result.value()->GetIds()[k] == actual_result.value()->GetIds()[k]);
        }
    }
}

}  // namespace

TEST_CASE("HGraph ParallelDeserialize Compressed Chunked Round-Trip",
          "[ut][hgraph][parallel_deserialize]") {
    auto data = MakeTestData(32, 500);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
    FrameMemoryReader reader(writer.buffer_);

    uint64_t threads = 0;
    SECTION("single thread") {
        threads = 1;
    }
    SECTION("four threads") {
        threads = 4;
    }
    SECTION("sixteen threads") {
        threads = 16;
    }

    auto pool = MakeThreadPool(threads);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
    REQUIRE(loaded->ParallelDeserialize(reader).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);
    RequireSameKnnResults(index, loaded, data);
}

TEST_CASE("HGraph ParallelDeserialize With Default Thread Pool",
          "[ut][hgraph][parallel_deserialize]") {
    auto data = MakeTestData(32, 300);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
    FrameMemoryReader reader(writer.buffer_);

    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim));
    REQUIRE(loaded->ParallelDeserialize(reader).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);
    RequireSameKnnResults(index, loaded, data);
}

TEST_CASE("HGraph ParallelDeserialize Plain Chunked Uses Positioned Reads Only",
          "[ut][hgraph][parallel_deserialize]") {
    auto data = MakeTestData(32, 300);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    BufferSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
    // the default ReadDecompressed throws, so this also proves the codec-none
    // path never touches compressed-frame reads
    PlainMemoryReader reader(writer.buffer_);

    auto pool = MakeThreadPool(4);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
    REQUIRE(loaded->ParallelDeserialize(reader).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);
    RequireSameKnnResults(index, loaded, data);
}

TEST_CASE("HGraph ParallelDeserialize Reorder Index Round-Trip",
          "[ut][hgraph][parallel_deserialize]") {
    auto data = MakeTestData(32, 500);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param, MakeSq8ReorderHGraphJson());
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
    FrameMemoryReader reader(writer.buffer_);

    auto pool = MakeThreadPool(4);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool), MakeSq8ReorderHGraphJson());
    REQUIRE(loaded->ParallelDeserialize(reader).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);
    RequireSameKnnResults(index, loaded, data);
}

TEST_CASE("HGraph ParallelDeserialize Dedup Index Falls Back to Whole Components",
          "[ut][hgraph][parallel_deserialize]") {
    auto data = MakeTestData(32, 300, /*with_duplicates=*/true);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param, MakeDedupHGraphJson());
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
    FrameMemoryReader reader(writer.buffer_);

    auto pool = MakeThreadPool(4);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool), MakeDedupHGraphJson());
    REQUIRE(loaded->ParallelDeserialize(reader).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);
    RequireSameKnnResults(index, loaded, data);
}

TEST_CASE("HGraph ParallelDeserialize File-Backed IO Round-Trip",
          "[ut][hgraph][parallel_deserialize]") {
    // file-backed ios (mmap/buffer) also serialize the three big components in
    // the chunked form and load them in parallel through ReserveIO/WriteRaw
    std::string io_type;
    SECTION("memory_io") {
        io_type = "memory_io";
    }
    SECTION("mmap_io") {
        io_type = "mmap_io";
    }
    SECTION("buffer_io") {
        io_type = "buffer_io";
    }

    auto data = MakeTestData(32, 500);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param, MakeIoTypeHGraphJson(io_type, "w"));
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
    FrameMemoryReader reader(writer.buffer_);

    auto pool = MakeThreadPool(4);
    auto loaded =
        MakeHGraphIndex(MakeCommonParam(data.dim, pool), MakeIoTypeHGraphJson(io_type, "r"));
    REQUIRE(loaded->ParallelDeserialize(reader).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);
    RequireSameKnnResults(index, loaded, data);
}

TEST_CASE("HGraph ParallelDeserialize Tiny Index All Whole Components",
          "[ut][hgraph][parallel_deserialize]") {
    auto data = MakeTestData(32, 1);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, vsag::DEFAULT_SERIALIZE_CHUNK_SIZE).has_value());
    FrameMemoryReader reader(writer.buffer_);

    auto pool = MakeThreadPool(4);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
    REQUIRE(loaded->ParallelDeserialize(reader).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);
    RequireSameKnnResults(index, loaded, data);
}

TEST_CASE("HGraph ParallelDeserialize Probes Sequential Files Without Layout",
          "[ut][hgraph][parallel_deserialize]") {
    auto data = MakeTestData(32, 300);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    // the all-in-one sequential format: body + footer, no chunked layout
    std::ostringstream out(std::ios::out | std::ios::binary);
    REQUIRE(index->Serialize(out).has_value());
    PlainMemoryReader reader(out.str());

    auto pool = MakeThreadPool(4);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
    REQUIRE(loaded->ParallelDeserialize(reader).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);
    RequireSameKnnResults(index, loaded, data);
}

// plaintext head of base_codes in the chunked form: three 4B
// FlattenInterface fields (total_count / max_capacity / code_size)
// followed by the 8B io size
constexpr uint64_t kBaseCodesHeadBytes = 3 * 4 + 8;

TEST_CASE("HGraph ParallelDeserialize Rejects Corrupted Inputs",
          "[ut][hgraph][parallel_deserialize]") {
    auto data = MakeTestData(32, 300);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());

    auto pool = MakeThreadPool(4);

    SECTION("not an index file at all") {
        PlainMemoryReader reader(std::string(1024, 'x'));
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
        REQUIRE_FALSE(loaded->ParallelDeserialize(reader).has_value());
    }

    // corrupt the first frame of the first chunked component (base_codes):
    // its payload no longer matches the chunk logical size, which must be
    // reported instead of silently accepted
    uint64_t victim_offset = 0;
    {
        FrameMemoryReader probe(writer.buffer_);
        // label_table is the first frame at offset 0; skip it to hit a chunk
        uint64_t payload_size = 0;
        probe.Read(0, sizeof(payload_size), &payload_size);
        victim_offset = sizeof(payload_size) + payload_size;
        // skip the plaintext head of base_codes
        victim_offset += kBaseCodesHeadBytes;
    }

    SECTION("decompressed chunk too short") {
        CorruptingFrameReader reader(writer.buffer_, victim_offset, -1);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
        REQUIRE_FALSE(loaded->ParallelDeserialize(reader).has_value());
    }

    SECTION("decompressed chunk too long") {
        CorruptingFrameReader reader(writer.buffer_, victim_offset, 1);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
        REQUIRE_FALSE(loaded->ParallelDeserialize(reader).has_value());
    }
}

TEST_CASE("HGraph ParallelDeserialize Rejects Tampered Component Granularity",
          "[ut][hgraph][parallel_deserialize]") {
    // label_table is always written as a single whole frame, so a chunked
    // granularity for it can only come from a tampered footer; the same holds
    // for a component name this reader does not know at all
    auto data = MakeTestData(32, 200);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());

    auto pool = MakeThreadPool(4);

    SECTION("whole-only component claims chunked granularity") {
        auto tampered = RewriteChunkedLayout(writer.buffer_, [](nlohmann::json& layout) {
            for (auto& comp : layout.at("components")) {
                if (comp.at("name").get<std::string>() != "label_table") {
                    continue;
                }
                const auto offset = comp.at("offset").get<uint64_t>();
                comp.erase("offset");
                comp.erase("csize");
                comp.erase("lsize");
                comp["type"] = "chunked";
                comp["head"] = {{"offset", offset}, {"size", 0}};
                comp["io_size"] = 0;
                comp["chunks"] = nlohmann::json::array();
                comp["tail"] = {{"offset", offset}, {"size", 0}};
                break;
            }
        });
        FrameMemoryReader reader(tampered);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
        REQUIRE_FALSE(loaded->ParallelDeserialize(reader).has_value());
    }

    SECTION("unknown component name") {
        auto tampered = RewriteChunkedLayout(writer.buffer_, [](nlohmann::json& layout) {
            nlohmann::json bogus;
            bogus["name"] = "bogus_component";
            bogus["type"] = "whole";
            bogus["offset"] = 0;
            bogus["csize"] = 0;
            bogus["lsize"] = 0;
            layout.at("components").push_back(std::move(bogus));
        });
        FrameMemoryReader reader(tampered);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
        REQUIRE_FALSE(loaded->ParallelDeserialize(reader).has_value());
    }
}

TEST_CASE("HGraph ParallelDeserialize Rejects Whole Component Byte Count Mismatch",
          "[ut][hgraph][parallel_deserialize]") {
    // a whole component must consume exactly its recorded logical size; a
    // tampered lsize or trailing frame garbage has to fail instead of
    // silently leaving the component half-read
    auto data = MakeTestData(32, 200);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());

    auto pool = MakeThreadPool(4);

    // label_table is the first whole frame in the body, at offset 0
    SECTION("whole frame payload too short") {
        CorruptingFrameReader reader(writer.buffer_, 0, -1);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
        REQUIRE_FALSE(loaded->ParallelDeserialize(reader).has_value());
    }

    SECTION("whole frame payload too long") {
        CorruptingFrameReader reader(writer.buffer_, 0, 1);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
        REQUIRE_FALSE(loaded->ParallelDeserialize(reader).has_value());
    }
}

TEST_CASE("HGraph ParallelDeserialize Default Pool Rejects Corrupted Inputs",
          "[ut][hgraph][parallel_deserialize]") {
    // the single-argument overload runs on the factory default pool, which
    // must not swallow task failures into a false success
    auto data = MakeTestData(32, 300);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());

    uint64_t victim_offset = 0;
    {
        FrameMemoryReader probe(writer.buffer_);
        // label_table is the first frame at offset 0; skip it to hit a chunk
        uint64_t payload_size = 0;
        probe.Read(0, sizeof(payload_size), &payload_size);
        victim_offset = sizeof(payload_size) + payload_size;
        // skip the plaintext head of base_codes
        victim_offset += kBaseCodesHeadBytes;
    }

    CorruptingFrameReader reader(writer.buffer_, victim_offset, -1);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim));
    REQUIRE_FALSE(loaded->ParallelDeserialize(reader).has_value());
}

TEST_CASE("HGraph ParallelDeserialize Empty Index", "[ut][hgraph][parallel_deserialize]") {
    // an index without any element serializes every component with
    // io_size == 0, so both paths must handle the all-whole form
    auto common_param = MakeCommonParam(32);
    auto index = MakeHGraphIndex(common_param);
    auto pool = MakeThreadPool(4);

    SECTION("chunked layout") {
        FrameSerializeWriter writer;
        REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
        FrameMemoryReader reader(writer.buffer_);
        auto loaded = MakeHGraphIndex(MakeCommonParam(32, pool));
        REQUIRE(loaded->ParallelDeserialize(reader).has_value());
        REQUIRE(loaded->GetNumElements() == 0);
    }
}

TEST_CASE("HGraph ParallelDeserialize Reports Configuration Mismatch",
          "[ut][hgraph][parallel_deserialize]") {
    // a reorder file loaded into a non-reorder index must fail with a
    // configuration error instead of an unknown-component one
    auto data = MakeTestData(32, 200);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param, MakeSq8ReorderHGraphJson());
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
    FrameMemoryReader reader(writer.buffer_);

    auto pool = MakeThreadPool(4);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
    auto result = loaded->ParallelDeserialize(reader);
    REQUIRE_FALSE(result.has_value());
    // either the serialized-parameter comparison or the component-level
    // guard must surface a configuration error, never an unknown component
    REQUIRE_THAT(result.error().message,
                 Catch::Matchers::ContainsSubstring("not match") ||
                     Catch::Matchers::ContainsSubstring("does not enable it"));
}

TEST_CASE("HGraph ParallelDeserialize Mci Index Round-Trip", "[ut][hgraph][parallel_deserialize]") {
    // the mci_cliques component must survive both parallel paths: as a whole
    // frame in the chunked layout and inside the probed sequential body
    auto data = MakeTestData(32, 300);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param, MakeMciHGraphJson());
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    auto pool = MakeThreadPool(4);

    SECTION("chunked layout") {
        FrameSerializeWriter writer;
        REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
        FrameMemoryReader reader(writer.buffer_);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool), MakeMciHGraphJson());
        REQUIRE(loaded->ParallelDeserialize(reader).has_value());
        REQUIRE(loaded->GetNumElements() == data.count);
        RequireSameKnnResults(index, loaded, data);
    }

    SECTION("probed body without layout") {
        std::ostringstream oss(std::ios::out | std::ios::binary);
        REQUIRE(index->Serialize(oss).has_value());
        PlainMemoryReader reader(oss.str());
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim), MakeMciHGraphJson());
        REQUIRE(loaded->ParallelDeserialize(reader).has_value());
        REQUIRE(loaded->GetNumElements() == data.count);
        RequireSameKnnResults(index, loaded, data);
    }
}

// records feedback edges so the conjugate graph is non-empty, and returns how
// many edges landed; a round trip must preserve the same enhanced results
uint32_t
FeedConjugateEdges(const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& index, TestData& data) {
    const auto* params = R"({"hgraph": {"ef_search": 64}})";
    uint32_t inserted = 0;
    for (int64_t i = 0; i < data.count; i += 41) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(data.dim)
            ->Float32Vectors(data.vectors.data() + i * data.dim)
            ->Owner(false);
        auto result = index->Feedback(query, 3, params, data.ids[i]);
        REQUIRE(result.has_value());
        inserted += result.value();
    }
    return inserted;
}

TEST_CASE("HGraph ParallelDeserialize Conjugate Graph Round-Trip",
          "[ut][hgraph][parallel_deserialize]") {
    // the conjugate graph has no io extent behind it, so it can only travel as
    // a whole frame; both parallel paths must still restore it
    auto data = MakeTestData(32, 300);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param, MakeConjugateHGraphJson());
    REQUIRE(index->Build(MakeDataset(data)).has_value());
    REQUIRE(FeedConjugateEdges(index, data) > 0);
    const auto expected_usage = index->GetMemoryUsageDetail().at("conjugate_graph");

    auto pool = MakeThreadPool(4);

    SECTION("chunked layout") {
        FrameSerializeWriter writer;
        REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
        FrameMemoryReader reader(writer.buffer_);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool), MakeConjugateHGraphJson());
        REQUIRE(loaded->ParallelDeserialize(reader).has_value());
        REQUIRE(loaded->GetNumElements() == data.count);
        REQUIRE(loaded->GetMemoryUsageDetail().at("conjugate_graph") == expected_usage);
        RequireSameKnnResults(index, loaded, data);
    }

    SECTION("uncompressed chunked layout") {
        // the component stops FOOTER_SIZE bytes short of its frame, so this
        // path only works because the frame is consumed before the component
        // sees it
        BufferSerializeWriter writer;
        REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
        PlainMemoryReader reader(writer.buffer_);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool), MakeConjugateHGraphJson());
        REQUIRE(loaded->ParallelDeserialize(reader).has_value());
        REQUIRE(loaded->GetNumElements() == data.count);
        REQUIRE(loaded->GetMemoryUsageDetail().at("conjugate_graph") == expected_usage);
        RequireSameKnnResults(index, loaded, data);
    }

    SECTION("probed body without layout") {
        std::ostringstream oss(std::ios::out | std::ios::binary);
        REQUIRE(index->Serialize(oss).has_value());
        PlainMemoryReader reader(oss.str());
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim), MakeConjugateHGraphJson());
        REQUIRE(loaded->ParallelDeserialize(reader).has_value());
        REQUIRE(loaded->GetNumElements() == data.count);
        REQUIRE(loaded->GetMemoryUsageDetail().at("conjugate_graph") == expected_usage);
        RequireSameKnnResults(index, loaded, data);
    }

    SECTION("target without conjugate graph rejects the component") {
        FrameSerializeWriter writer;
        REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
        FrameMemoryReader reader(writer.buffer_);
        auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool), MakeHGraphJson());
        auto result = loaded->ParallelDeserialize(reader);
        REQUIRE_FALSE(result.has_value());
        REQUIRE_THAT(result.error().message,
                     Catch::Matchers::ContainsSubstring("not match") ||
                         Catch::Matchers::ContainsSubstring("does not enable it"));
    }
}

// feeds each decompressed payload only a few bytes per underflow, the way a
// real streaming decompressor emits data in small increments
class DribbleFrameReader : public FrameMemoryReader {
public:
    using FrameMemoryReader::FrameMemoryReader;

    void
    ReadDecompressed(uint64_t offset,
                     uint64_t compressed_size,
                     const std::function<void(std::istream&)>& consume) override {
        uint64_t payload_size = 0;
        if (compressed_size < sizeof(payload_size)) {
            throw std::invalid_argument("frame shorter than its header");
        }
        Read(offset, sizeof(payload_size), &payload_size);
        if (compressed_size != sizeof(payload_size) + payload_size) {
            throw std::invalid_argument("frame size does not match its header");
        }
        class DribbleBuf : public std::streambuf {
        public:
            explicit DribbleBuf(std::string payload) : payload_(std::move(payload)) {
            }

        protected:
            int
            underflow() override {
                if (pos_ >= payload_.size()) {
                    return traits_type::eof();
                }
                const auto step = std::min<size_t>(7, payload_.size() - pos_);
                setg(payload_.data() + pos_, payload_.data() + pos_, payload_.data() + pos_ + step);
                pos_ += step;
                return traits_type::to_int_type(*gptr());
            }

        private:
            std::string payload_;
            size_t pos_{0};
        };
        DribbleBuf buf(buffer_.substr(offset + sizeof(payload_size), payload_size));
        std::istream is(&buf);
        consume(is);
    }
};

TEST_CASE("HGraph ParallelDeserialize Consumes Dribbling Decompressed Streams",
          "[ut][hgraph][parallel_deserialize]") {
    // chunk consumers must tolerate streams that surface data a few bytes at
    // a time and still enforce the exact per-chunk byte count
    auto data = MakeTestData(32, 300);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    FrameSerializeWriter writer;
    REQUIRE(index->Serialize(writer, /*chunk_size=*/4096).has_value());
    DribbleFrameReader reader(writer.buffer_);

    auto pool = MakeThreadPool(4);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim, pool));
    REQUIRE(loaded->ParallelDeserialize(reader).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);
    RequireSameKnnResults(index, loaded, data);
}
