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

#include <sys/stat.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "factory/serialized_index_factory.h"
#include "json_types.h"
#include "storage/serialization.h"
#include "storage/serialization_tags.h"
#include "storage/stream_writer.h"
#include "storage/streaming_serialization_test_utils.h"
#include "unittest.h"
#include "vsag/vsag.h"
#include "vsag_exception.h"

namespace {

using Json = nlohmann::json;

constexpr int64_t DIM = 8;
constexpr int64_t COUNT = 200;
constexpr int64_t QUERY_COUNT = 5;
constexpr int64_t TOPK = 10;

std::string
dense_parameter(const std::string& metric, const Json& index_param) {
    return Json{
        {"dtype", "float32"}, {"metric_type", metric}, {"dim", DIM}, {"index_param", index_param}}
        .dump();
}

std::string
hgraph_parameter(const std::string& metric = "l2") {
    return dense_parameter(metric,
                           {{"base_quantization_type", "sq8"},
                            {"max_degree", 16},
                            {"ef_construction", 100},
                            {"use_reorder", true},
                            {"precise_quantization_type", "fp32"}});
}

const Json BRUTE_FORCE_INDEX_PARAM = {{"base_quantization_type", "fp32"}};
const Json IVF_INDEX_PARAM = {{"buckets_count", 4},
                              {"base_quantization_type", "fp32"},
                              {"partition_strategy_type", "ivf"},
                              {"ivf_train_type", "kmeans"}};
const Json PYRAMID_INDEX_PARAM = {{"base_quantization_type", "fp32"},
                                  {"max_degree", 16},
                                  {"ef_construction", 100},
                                  {"index_min_size", 10},
                                  {"no_build_levels", Json::array()}};
const std::string SINDI_PARAMETER = R"({"dtype": "sparse", "metric_type": "ip",
    "index_param": {"term_id_limit": 2000, "doc_prune_ratio": 0.0, "window_size": 10000}})";

std::string
lazy_hgraph_parameter() {
    return Json{
        {"dtype", "float32"},
        {"metric_type", "l2"},
        {"dim", DIM},
        {"lazy_hgraph",
         {{"transition_threshold", 1000},
          {"hgraph",
           {{"base_quantization_type", "fp32"}, {"max_degree", 16}, {"ef_construction", 100}}}}}}
        .dump();
}

class MemoryReader : public vsag::Reader {
public:
    explicit MemoryReader(std::string bytes) : bytes_(std::move(bytes)) {
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::memcpy(dest, bytes_.data() + offset, len);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, vsag::CallBack callback) override {
        this->Read(offset, len, dest);
        callback(vsag::IOErrorCode::IO_SUCCESS, "success");
    }

    bool
    MultiRead(uint8_t* dests,
              const uint64_t* lens,
              const uint64_t* offsets,
              uint64_t count) override {
        for (uint64_t i = 0; i < count; ++i) {
            this->Read(offsets[i], lens[i], dests);
            dests += lens[i];
        }
        return true;
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return bytes_.size();
    }

private:
    std::string bytes_;
};

vsag::IndexPtr
create_index(const std::string& name, const std::string& parameter) {
    auto index = vsag::Factory::CreateIndex(name, parameter);
    REQUIRE(index.has_value());
    return index.value();
}

vsag::IndexPtr
build_dense_index(const std::string& name, const std::string& parameter, int64_t count = COUNT) {
    auto index = create_index(name, parameter);
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(count, DIM);
    auto base = vsag::Dataset::Make();
    base->NumElements(count)
        ->Dim(DIM)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    return index;
}

struct SparseData {
    std::vector<int64_t> indptr{0};
    std::vector<int32_t> indices;
    std::vector<float> values;
    std::vector<vsag::SparseVector> vectors;
    std::vector<uint32_t> ids_storage;
};

SparseData
make_sparse(int64_t rows, int seed) {
    SparseData data;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> length(3, 12);
    std::uniform_int_distribution<int> term(0, 999);
    std::uniform_real_distribution<float> value(0.1F, 1.0F);
    for (int64_t row = 0; row < rows; ++row) {
        std::set<int> terms;
        const auto target = length(rng);
        while (static_cast<int>(terms.size()) < target) {
            terms.insert(term(rng));
        }
        for (auto id : terms) {
            data.indices.push_back(id);
            data.values.push_back(value(rng));
        }
        data.indptr.push_back(static_cast<int64_t>(data.indices.size()));
    }
    data.ids_storage.assign(data.indices.begin(), data.indices.end());
    for (int64_t row = 0; row < rows; ++row) {
        vsag::SparseVector vector;
        const auto begin = data.indptr[row];
        vector.len_ = static_cast<uint32_t>(data.indptr[row + 1] - begin);
        vector.ids_ = data.ids_storage.data() + begin;
        vector.vals_ = data.values.data() + begin;
        data.vectors.push_back(vector);
    }
    return data;
}

vsag::IndexPtr
build_sindi_index(const SparseData& data,
                  const std::string& name = "sindi",
                  const std::string& parameter = SINDI_PARAMETER) {
    auto index = create_index(name, parameter);
    std::vector<int64_t> ids(data.vectors.size());
    std::iota(ids.begin(), ids.end(), 0);
    auto base = vsag::Dataset::Make();
    base->NumElements(static_cast<int64_t>(data.vectors.size()))
        ->Ids(ids.data())
        ->SparseVectors(data.vectors.data())
        ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    return index;
}

std::vector<vsag::MultiVector>
make_multi_vectors(std::vector<float>& storage, int64_t count, uint32_t tokens) {
    storage = fixtures::generate_vectors(count * tokens, DIM);
    std::vector<vsag::MultiVector> vectors(count);
    for (int64_t i = 0; i < count; ++i) {
        vectors[i].len_ = tokens;
        vectors[i].vectors_ = storage.data() + i * tokens * DIM;
    }
    return vectors;
}

vsag::IndexPtr
build_multi_vector_index(const std::string& name, const std::string& parameter) {
    auto index = create_index(name, parameter);
    std::vector<float> storage;
    auto vectors = make_multi_vectors(storage, 50, 4);
    std::vector<int64_t> ids(vectors.size());
    std::iota(ids.begin(), ids.end(), 0);
    auto base = vsag::Dataset::Make();
    base->NumElements(static_cast<int64_t>(vectors.size()))
        ->Dim(DIM)
        ->Ids(ids.data())
        ->MultiVectors(vectors.data())
        ->MultiVectorDim(DIM)
        ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    return index;
}

void
save(const vsag::IndexPtr& index, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    REQUIRE(index->Serialize(out).has_value());
}

void
save_streaming(const vsag::IndexPtr& index, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    REQUIRE(index->SerializeStreaming(out).has_value());
}

template <typename T>
void
write_values(std::ofstream& out, const std::vector<T>& values) {
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
}

void
write_dense_queries(const std::string& path, uint32_t rows, uint32_t dim) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    write_values(out, fixtures::generate_vectors(rows, dim, true, 99));
}

void
write_csr(const std::string& path, const SparseData& data, int64_t cols = 1000) {
    std::ofstream out(path, std::ios::binary);
    const int64_t rows = static_cast<int64_t>(data.indptr.size()) - 1;
    const auto nnz = static_cast<int64_t>(data.indices.size());
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));
    write_values(out, data.indptr);
    write_values(out, data.indices);
    write_values(out, data.values);
}

void
write_bytes(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string
read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

Json
analyze_ok(const vsag::AnalyzeIndexOptions& options) {
    auto result = vsag::analyze_index(options);
    if (not result.has_value()) {
        FAIL(result.error().message);
    }
    auto report = Json::parse(result.value());
    REQUIRE(report.at("schema_version") == vsag::ANALYZE_INDEX_SCHEMA_VERSION);
    for (const auto* key :
         {"metadata", "static_analysis", "dynamic_analysis", "warnings", "compatibility"}) {
        REQUIRE(report.contains(key));
    }
    return report;
}

vsag::Error
analyze_error(const vsag::AnalyzeIndexOptions& options) {
    auto result = vsag::analyze_index(options);
    REQUIRE_FALSE(result.has_value());
    return result.error();
}

bool
has_warning(const Json& report, const std::string& code) {
    for (const auto& warning : report.at("warnings")) {
        if (warning.at("code") == code) {
            return true;
        }
    }
    return false;
}

vsag::AnalyzeIndexOptions
options_for(const std::string& index_path) {
    vsag::AnalyzeIndexOptions options;
    options.index_path = index_path;
    return options;
}

// A writer that claims a compression codec but stores frames verbatim.
class FakeCompressingWriter : public vsag::SerializeWriter {
public:
    explicit FakeCompressingWriter(const std::string& path) : out_(path, std::ios::binary) {
    }

    void
    Write(const char* data, uint64_t size) override {
        out_.write(data, static_cast<std::streamsize>(size));
        frame_size_ += size;
    }

    [[nodiscard]] std::string
    GetCompressorName() const override {
        return codec_;
    }

    void
    BeginCompressedFrame() override {
        frame_size_ = 0;
    }

    uint64_t
    EndCompressedFrame() override {
        return frame_size_;
    }

    std::string codec_{"none"};

private:
    std::ofstream out_;
    uint64_t frame_size_{0};
};

}  // namespace

TEST_CASE("analyze_index reloads indexes from their recorded metadata", "[ut][analyze_index]") {
    fixtures::TempDir dir("analyze_index_metadata");
    const auto path = dir.path + "index";

    SECTION("hgraph") {
        save(build_dense_index("hgraph", hgraph_parameter()), path);
        auto report = analyze_ok(options_for(path));
        const auto& metadata = report["metadata"];
        CHECK(metadata["index_path"] == path);
        CHECK(metadata["file_size_bytes"] == std::filesystem::file_size(path));
        CHECK(metadata["index_type"] == "hgraph");
        CHECK(metadata["dim"] == DIM);
        CHECK(metadata["data_type"] == "float32");
        CHECK(metadata["metric"] == "l2");
        CHECK(metadata["num_elements"] == COUNT);
        CHECK(metadata["memory_usage_bytes"].get<int64_t>() > 0);
        CHECK(metadata["parameter_source"] == "file");
        CHECK(metadata["index_param"]["type"] == "hgraph");
        CHECK(metadata["build_parameter"].is_null());
        CHECK(metadata["basic_info"]["total_count"] == COUNT);
        CHECK_FALSE(metadata["basic_info"].contains("index_param"));
        CHECK(report["static_analysis"]["_analysis"]["analysis_type"] == "stats");
        CHECK(report["static_analysis"]["total_count"] == COUNT);
        CHECK(report["dynamic_analysis"].is_null());
        CHECK(report["warnings"].empty());
        const auto& compatibility = report["compatibility"];
        CHECK(compatibility["serialization_format"] == "footer");
        CHECK(compatibility["vsag_version"] == vsag::version());
        CHECK(compatibility["metadata_complete"] == true);
        CHECK(compatibility["lazy_hgraph_container"] == false);
    }

    SECTION("brute_force records neither metric nor data type") {
        save(build_dense_index("brute_force", dense_parameter("ip", BRUTE_FORCE_INDEX_PARAM)),
             path);
        auto report = analyze_ok(options_for(path));
        CHECK(report["metadata"]["index_type"] == "brute_force");
        CHECK(report["metadata"]["metric"] == "l2");
        CHECK(has_warning(report, "metadata_defaults"));
        CHECK(report["compatibility"]["metadata_complete"] == false);
        CHECK(report["compatibility"]["defaulted_fields"]["metric_type"] == "l2");
        CHECK(report["compatibility"]["defaulted_fields"]["dtype"] == "float32");
    }

    SECTION("ivf") {
        save(build_dense_index("ivf", dense_parameter("ip", IVF_INDEX_PARAM)), path);
        auto report = analyze_ok(options_for(path));
        CHECK(report["metadata"]["index_type"] == "ivf");
        CHECK(report["metadata"]["metric"] == "ip");
        CHECK(report["warnings"].empty());
    }

    SECTION("sindi") {
        save(build_sindi_index(make_sparse(100, 7)), path);
        auto report = analyze_ok(options_for(path));
        CHECK(report["metadata"]["index_type"] == "sindi");
        CHECK(report["metadata"]["dim"].is_null());
        CHECK(report["metadata"]["data_type"] == "sparse");
        CHECK(report["metadata"]["metric"] == "ip");
        CHECK(report["warnings"].empty());
    }

    SECTION("sindi_v2 with file-backed posting storage") {
        auto parameter = Json::parse(SINDI_PARAMETER);
        parameter["index_param"]["use_reorder"] = true;
        parameter["index_param"]["term_io"] = {{"type", "buffer_io"},
                                               {"file_path", dir.path + "terms"}};
        save(build_sindi_index(make_sparse(100, 7), "sindi_v2", parameter.dump()), path);
        std::filesystem::remove(dir.path + "terms");
        auto report = analyze_ok(options_for(path));
        CHECK(report["metadata"]["index_type"] == "sindi_v2");
        CHECK(has_warning(report, "memory_io"));
        CHECK_FALSE(std::filesystem::exists(dir.path + "terms"));

        write_csr(dir.path + "queries.csr", make_sparse(3, 13));
        auto options = options_for(path);
        options.query_path = dir.path + "queries.csr";
        options.topk = TOPK;
        auto dynamic = analyze_ok(options)["dynamic_analysis"];
        CHECK(dynamic["query_data_type"] == "sparse");
        CHECK(dynamic["result"]["_analysis"]["sample_count"] == 3);
    }

    SECTION("warp and simq") {
        save(build_multi_vector_index("warp", dense_parameter("ip", Json::object())), path);
        auto warp = analyze_ok(options_for(path));
        CHECK(warp["metadata"]["index_type"] == "warp");
        CHECK(warp["static_analysis"]["_analysis"]["index_type"] == "warp");

        const auto simq_parameter = dense_parameter("ip",
                                                    {{"base_io_type", "memory_io"},
                                                     {"init_cluster_ratio", 0.2},
                                                     {"max_cluster_size", 64},
                                                     {"split_start_idx", 32},
                                                     {"coarse_k", 8},
                                                     {"rerank_k", 100},
                                                     {"build_thread_count", 1}});
        save(build_multi_vector_index("simq", simq_parameter), path);
        auto simq = analyze_ok(options_for(path));
        CHECK(simq["metadata"]["index_type"] == "simq");

        write_dense_queries(dir.path + "queries", QUERY_COUNT, DIM);
        auto options = options_for(path);
        options.query_path = dir.path + "queries";
        CHECK(analyze_error(options).type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
    }

    SECTION("empty hgraph") {
        save(create_index("hgraph", hgraph_parameter()), path);
        auto report = analyze_ok(options_for(path));
        CHECK(report["metadata"]["num_elements"] == 0);
        CHECK(has_warning(report, "empty_index"));
    }
}

TEST_CASE("analyze_index runs dynamic analysis on query files", "[ut][analyze_index]") {
    fixtures::TempDir dir("analyze_index_dynamic");
    const auto path = dir.path + "index";
    const auto queries = dir.path + "queries";
    write_dense_queries(queries, QUERY_COUNT, DIM);
    auto options = options_for(path);
    options.query_path = queries;
    options.topk = TOPK;

    SECTION("hgraph with the default search parameter") {
        save(build_dense_index("hgraph", hgraph_parameter()), path);
        auto report = analyze_ok(options);
        const auto& dynamic = report["dynamic_analysis"];
        CHECK(dynamic["query_path"] == queries);
        CHECK(dynamic["query_data_type"] == "dense");
        CHECK(dynamic["query_count"] == QUERY_COUNT);
        CHECK(dynamic["query_dim"] == DIM);
        CHECK(dynamic["topk"] == TOPK);
        CHECK(dynamic["search_parameter"] == Json::parse(R"({"hgraph": {"ef_search": 100}})"));
        CHECK(dynamic["search_parameter_source"] == "default");
        CHECK(dynamic["result"]["_analysis"]["analysis_type"] == "search");
        CHECK(dynamic["result"]["_analysis"]["sample_count"] == QUERY_COUNT);
        CHECK(dynamic["result"].contains("recall_query"));
    }

    SECTION("hgraph with a user search parameter") {
        save(build_dense_index("hgraph", hgraph_parameter()), path);
        options.search_parameter = R"({"hgraph": {"ef_search": 40}})";
        auto report = analyze_ok(options);
        CHECK(report["dynamic_analysis"]["search_parameter"]["hgraph"]["ef_search"] == 40);
        CHECK(report["dynamic_analysis"]["search_parameter_source"] == "user");
    }

    SECTION("brute_force analyzes every query of the batch") {
        save(build_dense_index("brute_force", dense_parameter("l2", BRUTE_FORCE_INDEX_PARAM)),
             path);
        auto report = analyze_ok(options);
        const auto& result = report["dynamic_analysis"]["result"];
        CHECK(report["dynamic_analysis"]["search_parameter"] == Json::object());
        CHECK(result["result_count"] == QUERY_COUNT * TOPK);
        CHECK(result["_analysis"]["sample_count"] == QUERY_COUNT);
    }

    SECTION("ivf") {
        save(build_dense_index("ivf", dense_parameter("l2", IVF_INDEX_PARAM)), path);
        auto report = analyze_ok(options);
        CHECK(report["dynamic_analysis"]["result"]["_analysis"]["index_type"] == "ivf");
    }

    SECTION("dense query files need float32 indexes") {
        auto parameter = Json::parse(hgraph_parameter());
        parameter["dtype"] = "float16";
        auto index = create_index("hgraph", parameter.dump());
        auto codes = fixtures::generate_fp16_codes_with_floats(COUNT, DIM).first;
        std::vector<int64_t> labels(COUNT);
        std::iota(labels.begin(), labels.end(), 0);
        auto base = vsag::Dataset::Make();
        base->NumElements(COUNT)
            ->Dim(DIM)
            ->Ids(labels.data())
            ->Float16Vectors(codes.data())
            ->Owner(false);
        REQUIRE(index->Build(base).has_value());
        save(index, path);
        auto error = analyze_error(options);
        CHECK(error.type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
        CHECK(error.message.find("float16") != std::string::npos);
    }

    SECTION("int8 indexes support static analysis") {
        auto parameter = Json::parse(hgraph_parameter());
        parameter["dtype"] = "int8";
        parameter["index_param"]["base_quantization_type"] = "int8";
        parameter["index_param"]["use_reorder"] = false;
        auto index = create_index("hgraph", parameter.dump());
        auto codes = fixtures::generate_int8_codes(COUNT, DIM);
        std::vector<int64_t> labels(COUNT);
        std::iota(labels.begin(), labels.end(), 0);
        auto base = vsag::Dataset::Make();
        base->NumElements(COUNT)->Dim(DIM)->Ids(labels.data())->Int8Vectors(codes.data());
        base->Owner(false);
        REQUIRE(index->Build(base).has_value());
        save(index, path);
        auto report = analyze_ok(options_for(path));
        CHECK(report["metadata"]["data_type"] == "int8");
        CHECK(report["static_analysis"]["total_count"] == COUNT);
    }
}

TEST_CASE("analyze_index analyzes sindi with sparse inputs", "[ut][analyze_index]") {
    fixtures::TempDir dir("analyze_index_sindi");
    const auto path = dir.path + "index";
    const auto base_path = dir.path + "base.csr";
    const auto query_path = dir.path + "query.csr";
    const auto base = make_sparse(100, 7);
    save(build_sindi_index(base), path);
    write_csr(base_path, base);
    write_csr(query_path, make_sparse(4, 11));
    auto options = options_for(path);
    options.query_path = query_path;
    options.topk = TOPK;

    SECTION("auto query type") {
        auto report = analyze_ok(options);
        CHECK(report["dynamic_analysis"]["query_data_type"] == "sparse");
        CHECK(report["dynamic_analysis"]["query_count"] == 4);
        CHECK(report["dynamic_analysis"]["query_cols"] == 1000);
    }

    SECTION("generated, saved and reused ground truth") {
        const auto groundtruth = dir.path + "query.gt";
        options.base_path = base_path;
        options.save_groundtruth_path = groundtruth;
        auto generated = analyze_ok(options);
        CHECK(generated["dynamic_analysis"]["base_path"] == base_path);
        CHECK(generated["dynamic_analysis"]["result"].contains("recall_query"));
        REQUIRE(std::filesystem::exists(groundtruth));

        options.base_path.clear();
        options.save_groundtruth_path.clear();
        options.groundtruth_path = groundtruth;
        auto reused = analyze_ok(options);
        CHECK(reused["dynamic_analysis"]["groundtruth_path"] == groundtruth);
        CHECK(reused["dynamic_analysis"]["result"].contains("recall_query"));

        options.topk = 1000;
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_ARGUMENT);
        options.groundtruth_path = base_path;
        options.topk = TOPK;
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_BINARY);
    }

    SECTION("static analysis with the base dataset") {
        auto static_options = options_for(path);
        static_options.base_path = base_path;
        auto report = analyze_ok(static_options);
        CHECK(report["static_analysis"]["total_count"] == 100);
        CHECK(report["dynamic_analysis"].is_null());
    }

    SECTION("invalid sparse inputs") {
        options.query_data_type = "dense";
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_ARGUMENT);

        options.query_data_type = "sparse";
        auto broken = make_sparse(4, 11);
        broken.indptr[1] = broken.indptr[2] + 1;
        write_csr(query_path, broken);
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_BINARY);
        broken = make_sparse(4, 11);
        broken.indices[0] = 5000;
        write_csr(query_path, broken);
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_BINARY);
        write_csr(query_path, make_sparse(4, 11));
        auto bytes = read_bytes(query_path);
        write_bytes(query_path, bytes.substr(0, bytes.size() - 3));
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_BINARY);
        write_bytes(query_path, "tiny");
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_BINARY);
        write_bytes(base_path, std::string(24, '\0'));
        options.base_path = base_path;
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_BINARY);
    }
}

TEST_CASE("analyze_index supports the streaming and chunked formats", "[ut][analyze_index]") {
    fixtures::TempDir dir("analyze_index_formats");
    const auto path = dir.path + "index";

    SECTION("streaming") {
        auto index = build_dense_index("hgraph", hgraph_parameter());
        save_streaming(index, path);
        auto report = analyze_ok(options_for(path));
        CHECK(report["metadata"]["index_type"] == "hgraph");
        CHECK(report["metadata"]["num_elements"] == COUNT);
        CHECK(report["compatibility"]["serialization_format"] == "streaming");
        CHECK_FALSE(report["compatibility"]["streaming_blocks"].empty());

        auto options = options_for(path);
        options.build_parameter = hgraph_parameter();
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_ARGUMENT);

        auto bytes = read_bytes(path);
        write_bytes(path, bytes.substr(0, 40));
        CHECK(analyze_error(options_for(path)).type != vsag::ErrorType::UNKNOWN_ERROR);
    }

    SECTION("streaming files of the other index types") {
        save_streaming(
            build_dense_index("brute_force", dense_parameter("ip", BRUTE_FORCE_INDEX_PARAM)), path);
        auto brute_force = analyze_ok(options_for(path));
        CHECK(brute_force["metadata"]["index_type"] == "brute_force");
        CHECK(brute_force["metadata"]["metric"] == "ip");
        CHECK(brute_force["compatibility"]["metadata_complete"] == true);

        save_streaming(build_dense_index("ivf", dense_parameter("l2", IVF_INDEX_PARAM)), path);
        CHECK(analyze_ok(options_for(path))["metadata"]["index_type"] == "ivf");

        save_streaming(build_sindi_index(make_sparse(100, 7)), path);
        auto sindi = analyze_ok(options_for(path));
        CHECK(sindi["metadata"]["index_type"] == "sindi");
        CHECK(sindi["compatibility"]["serialization_format"] == "streaming");

        auto sindi_v2 = Json::parse(SINDI_PARAMETER);
        sindi_v2["index_param"]["use_reorder"] = true;
        save_streaming(build_sindi_index(make_sparse(100, 7), "sindi_v2", sindi_v2.dump()), path);
        CHECK(analyze_ok(options_for(path))["metadata"]["index_type"] == "sindi_v2");
    }

    SECTION("streaming pyramid records its dimension") {
        save_streaming(build_dense_index("pyramid", dense_parameter("l2", PYRAMID_INDEX_PARAM)),
                       path);
        auto report = analyze_ok(options_for(path));
        CHECK(report["metadata"]["index_type"] == "pyramid");
        CHECK(report["metadata"]["dim"] == DIM);
    }

    SECTION("chunked") {
        auto index = build_dense_index("hgraph", hgraph_parameter());
        {
            FakeCompressingWriter writer(path);
            REQUIRE(index->Serialize(writer, 4096).has_value());
        }
        auto report = analyze_ok(options_for(path));
        CHECK(report["compatibility"]["serialization_format"] == "chunked");
        CHECK(report["compatibility"]["chunked_codec"] == "none");
        CHECK(report["compatibility"]["chunk_size"] == 4096);

        {
            FakeCompressingWriter writer(path);
            writer.codec_ = "fake";
            REQUIRE(index->Serialize(writer, 4096).has_value());
        }
        auto error = analyze_error(options_for(path));
        CHECK(error.type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
        CHECK(error.message.find("\"fake\"") != std::string::npos);
    }
}

TEST_CASE("analyze_index uses build_parameter when the file is not self-describing",
          "[ut][analyze_index]") {
    fixtures::TempDir dir("analyze_index_build_parameter");
    const auto path = dir.path + "index";

    SECTION("pyramid footers do not record the dimension") {
        const auto parameter = dense_parameter("l2", PYRAMID_INDEX_PARAM);
        save(build_dense_index("pyramid", parameter), path);
        auto error = analyze_error(options_for(path));
        CHECK(error.type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
        CHECK(error.message.find("build_parameter") != std::string::npos);

        auto options = options_for(path);
        options.build_parameter = parameter;
        auto report = analyze_ok(options);
        CHECK(report["metadata"]["index_type"] == "pyramid");
        CHECK(report["metadata"]["parameter_source"] == "user");
        CHECK(report["metadata"]["build_parameter"] == Json::parse(parameter));
        CHECK(has_warning(report, "build_parameter_override"));

        write_dense_queries(dir.path + "queries", QUERY_COUNT, DIM);
        options.query_path = dir.path + "queries";
        error = analyze_error(options);
        CHECK(error.message.find("Dataset::Paths") != std::string::npos);
    }

    SECTION("lazy_hgraph") {
        save(build_dense_index("lazy_hgraph", lazy_hgraph_parameter()), path);
        auto error = analyze_error(options_for(path));
        CHECK(error.type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
        CHECK(error.message.find("lazy_hgraph") != std::string::npos);

        auto options = options_for(path);
        options.build_parameter = lazy_hgraph_parameter();
        auto report = analyze_ok(options);
        CHECK(report["metadata"]["index_type"] == "lazy_hgraph");
        CHECK(report["compatibility"]["lazy_hgraph_container"] == true);

        write_dense_queries(dir.path + "queries", QUERY_COUNT, DIM);
        options.query_path = dir.path + "queries";
        options.topk = TOPK;
        CHECK(analyze_ok(options)["dynamic_analysis"]["result"]["_analysis"]["sample_count"] ==
              QUERY_COUNT);

        options.build_parameter = Json::parse(hgraph_parameter()).dump();
        auto typed = Json::parse(options.build_parameter);
        typed["type"] = "hgraph";
        options.build_parameter = typed.dump();
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("legacy files without a footer") {
        auto parameter = Json::parse(hgraph_parameter());
        parameter["use_old_serial_format"] = true;
        save(build_dense_index("hgraph", parameter.dump()), path);
        auto error = analyze_error(options_for(path));
        CHECK(error.type == vsag::ErrorType::INVALID_BINARY);

        auto options = options_for(path);
        options.build_parameter = parameter.dump();
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_ARGUMENT);
        parameter["type"] = "hgraph";
        options.build_parameter = parameter.dump();
        auto report = analyze_ok(options);
        CHECK(report["compatibility"]["serialization_format"] == "legacy");
        CHECK(has_warning(report, "legacy_format"));
        CHECK(report["metadata"]["basic_info"].is_null());
    }

    SECTION("mismatched or invalid parameters") {
        save(build_dense_index("hgraph", hgraph_parameter()), path);
        auto options = options_for(path);
        auto parameter = Json::parse(dense_parameter("l2", IVF_INDEX_PARAM));
        parameter["type"] = "ivf";
        options.build_parameter = parameter.dump();
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_ARGUMENT);

        parameter.erase("type");
        parameter["dim"] = -1;
        options.build_parameter = parameter.dump();
        CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("analyze_index loads file-backed indexes into memory", "[ut][analyze_index]") {
    fixtures::TempDir dir("analyze_index_io");
    const auto path = dir.path + "index";
    const auto codes_path = dir.path + "codes";
    auto parameter = Json::parse(hgraph_parameter());
    parameter["index_param"]["base_io_type"] = "buffer_io";
    parameter["index_param"]["base_file_path"] = codes_path;
    save(build_dense_index("hgraph", parameter.dump()), path);
    std::filesystem::remove(codes_path);

    auto report = analyze_ok(options_for(path));
    CHECK(has_warning(report, "memory_io"));
    CHECK(report["compatibility"]["memory_io_overrides"][0]["recorded_io_type"] == "buffer_io");
    CHECK(report["metadata"]["index_param"]["base_codes"]["io_params"]["type"] == "buffer_io");
    CHECK(report["metadata"]["num_elements"] == COUNT);
    CHECK_FALSE(std::filesystem::exists(codes_path));

    SECTION("streaming hgraph") {
        save_streaming(build_dense_index("hgraph", parameter.dump()), path);
        std::filesystem::remove(codes_path);
        auto streaming = analyze_ok(options_for(path));
        CHECK(streaming["compatibility"]["serialization_format"] == "streaming");
        CHECK(has_warning(streaming, "memory_io"));
        CHECK(streaming["metadata"]["num_elements"] == COUNT);
        CHECK_FALSE(std::filesystem::exists(codes_path));
    }

    SECTION("streaming files whose loader cannot use memory IO") {
        auto sindi_v2 = Json::parse(SINDI_PARAMETER);
        sindi_v2["index_param"]["use_reorder"] = true;
        sindi_v2["index_param"]["term_io"] = {{"type", "buffer_io"},
                                              {"file_path", dir.path + "terms"}};
        save_streaming(build_sindi_index(make_sparse(100, 7), "sindi_v2", sindi_v2.dump()), path);
        std::filesystem::remove(dir.path + "terms");
        auto error = analyze_error(options_for(path));
        CHECK(error.type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
        CHECK(error.message.find("buffer_io") != std::string::npos);
        CHECK_FALSE(std::filesystem::exists(dir.path + "terms"));
    }

    SECTION("streaming hgraph snapshots of reader-backed indexes") {
        std::stringstream stream;
        REQUIRE(build_dense_index("hgraph", hgraph_parameter())->SerializeStreaming(stream));
        const auto bytes = stream.str();
        const auto block = vsag::test::FindStreamingBlock(
            bytes, vsag::StreamSerializationTag::HIGH_PRECISION_CODES);
        vsag::LoadParameters load_parameters;
        load_parameters.Set("precise_io_type", "reader_io")
            .SetReader("precise_reader",
                       std::make_shared<MemoryReader>(
                           bytes.substr(block.payload_offset, block.payload_size)));
        std::stringstream load_stream(bytes);
        auto loaded = vsag::Index::Load(load_stream, load_parameters);
        REQUIRE(loaded.has_value());
        save_streaming(loaded.value(), path);

        auto snapshot = analyze_ok(options_for(path));
        CHECK(snapshot["metadata"]["num_elements"] == COUNT);
        CHECK(snapshot["compatibility"]["memory_io_overrides"].dump().find("reader_io") !=
              std::string::npos);
    }

    SECTION("split RaBitQ supplement storage") {
        const auto supplement_path = dir.path + "supplement";
        auto split_parameter =
            Json::parse(dense_parameter("l2",
                                        {{"base_quantization_type", "rabitq"},
                                         {"precise_quantization_type", "rabitq"},
                                         {"base_io_type", "block_memory_io"},
                                         {"base_supplement_io_type", "async_io"},
                                         {"base_supplement_file_path", supplement_path},
                                         {"use_reorder", true},
                                         {"build_by_base", true},
                                         {"rabitq_bits_per_dim_base", 3},
                                         {"rabitq_bits_per_dim_precise", 5},
                                         {"rabitq_error_rate", 1.9},
                                         {"fast_encode_rabitq", true},
                                         {"max_degree", 16},
                                         {"ef_construction", 50},
                                         {"graph_storage_type", "compressed"}}));
        const auto plant_sentinel = [&]() {
            std::filesystem::remove(supplement_path);
            std::ofstream(supplement_path, std::ios::binary) << "must stay untouched";
        };
        const auto sentinel_intact = [&]() {
            std::ifstream in(supplement_path, std::ios::binary);
            return std::string(std::istreambuf_iterator<char>(in), {}) == "must stay untouched";
        };

        save(build_dense_index("hgraph", split_parameter.dump()), path);
        plant_sentinel();
        auto footer = analyze_ok(options_for(path));
        CHECK(footer["metadata"]["num_elements"] == COUNT);
        CHECK(footer["compatibility"]["memory_io_overrides"].dump().find("supplement_io_params") !=
              std::string::npos);
        CHECK(sentinel_intact());

        save_streaming(build_dense_index("hgraph", split_parameter.dump()), path);
        plant_sentinel();
        auto error = analyze_error(options_for(path));
        CHECK(error.type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
        CHECK(error.message.find("supplement_io_params") != std::string::npos);
        CHECK(sentinel_intact());
    }

    SECTION("streaming brute_force and warp cannot use memory IO") {
        const auto sentinel_path = dir.path + "recorded";
        const auto require_refused = [&](const vsag::IndexPtr& index) {
            save_streaming(index, path);
            std::filesystem::remove(sentinel_path);
            std::ofstream(sentinel_path, std::ios::binary) << "must stay untouched";
            auto error = analyze_error(options_for(path));
            CHECK(error.type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
            CHECK(error.message.find("buffer_io") != std::string::npos);
            std::ifstream in(sentinel_path, std::ios::binary);
            CHECK(std::string(std::istreambuf_iterator<char>(in), {}) == "must stay untouched");
        };
        auto brute_force = BRUTE_FORCE_INDEX_PARAM;
        brute_force["base_io_type"] = "buffer_io";
        brute_force["base_file_path"] = sentinel_path;
        require_refused(build_dense_index("brute_force", dense_parameter("l2", brute_force)));
        require_refused(build_multi_vector_index(
            "warp",
            dense_parameter("ip",
                            {{"base_io_type", "buffer_io"}, {"base_file_path", sentinel_path}})));
    }
}

TEST_CASE("analyze_index rejects invalid options and damaged files", "[ut][analyze_index]") {
    fixtures::TempDir dir("analyze_index_errors");
    const auto path = dir.path + "index";
    const auto queries = dir.path + "queries";
    save(build_dense_index("hgraph", hgraph_parameter()), path);
    write_dense_queries(queries, QUERY_COUNT, DIM);

    auto expect = [](const vsag::AnalyzeIndexOptions& options, vsag::ErrorType type) {
        auto error = analyze_error(options);
        INFO(error.message);
        CHECK(error.type == type);
        CHECK_FALSE(error.message.empty());
    };
    auto with_query = [&](const std::function<void(vsag::AnalyzeIndexOptions&)>& change) {
        auto options = options_for(path);
        options.query_path = queries;
        change(options);
        return options;
    };

    SECTION("options") {
        expect(options_for(""), vsag::ErrorType::INVALID_ARGUMENT);
        expect(with_query([](auto& o) { o.query_data_type = "csr"; }),
               vsag::ErrorType::INVALID_ARGUMENT);
        expect(with_query([](auto& o) { o.topk = 0; }), vsag::ErrorType::INVALID_ARGUMENT);
        expect(with_query([](auto& o) { o.build_parameter = "{"; }),
               vsag::ErrorType::INVALID_ARGUMENT);
        expect(with_query([](auto& o) { o.search_parameter = "[1]"; }),
               vsag::ErrorType::INVALID_ARGUMENT);
        expect(with_query([](auto& o) { o.search_parameter = std::string(300, 'x'); }),
               vsag::ErrorType::INVALID_ARGUMENT);
        expect(with_query([&](auto& o) {
                   o.groundtruth_path = queries;
                   o.save_groundtruth_path = dir.path + "gt";
               }),
               vsag::ErrorType::INVALID_ARGUMENT);
        expect(with_query([&](auto& o) { o.base_path = dir.path + "missing"; }),
               vsag::ErrorType::MISSING_FILE);
        expect(with_query([&](auto& o) { o.save_groundtruth_path = dir.path + "no/such/gt"; }),
               vsag::ErrorType::MISSING_FILE);
        expect(with_query([&](auto& o) { o.save_groundtruth_path = dir.path; }),
               vsag::ErrorType::INVALID_ARGUMENT);
        auto static_options = options_for(path);
        static_options.groundtruth_path = queries;
        expect(static_options, vsag::ErrorType::INVALID_ARGUMENT);
        static_options = options_for(path);
        static_options.save_groundtruth_path = dir.path + "gt";
        expect(static_options, vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("inputs that do not fit the index") {
        write_csr(dir.path + "base.csr", make_sparse(4, 3));
        expect(with_query([&](auto& o) { o.base_path = dir.path + "base.csr"; }),
               vsag::ErrorType::INVALID_ARGUMENT);
        expect(with_query([](auto& o) { o.query_data_type = "sparse"; }),
               vsag::ErrorType::INVALID_ARGUMENT);
        write_dense_queries(queries, QUERY_COUNT, DIM + 1);
        expect(with_query([](auto&) {}), vsag::ErrorType::INVALID_ARGUMENT);
        auto bytes = read_bytes(queries);
        write_bytes(queries, bytes.substr(0, bytes.size() - 1));
        expect(with_query([](auto&) {}), vsag::ErrorType::INVALID_BINARY);
        write_bytes(queries, "abc");
        expect(with_query([](auto&) {}), vsag::ErrorType::INVALID_BINARY);
        expect(with_query([&](auto& o) { o.query_path = dir.path + "missing"; }),
               vsag::ErrorType::MISSING_FILE);
    }

    SECTION("index files") {
        expect(options_for(dir.path + "missing"), vsag::ErrorType::MISSING_FILE);
        expect(options_for(dir.path), vsag::ErrorType::INVALID_ARGUMENT);
        const auto other = dir.path + "other";
        write_bytes(other, "");
        expect(options_for(other), vsag::ErrorType::INVALID_BINARY);
        write_bytes(other, "not an index");
        expect(options_for(other), vsag::ErrorType::INVALID_BINARY);
        const auto bytes = read_bytes(path);
        write_bytes(other, bytes.substr(0, bytes.size() / 2));
        expect(options_for(other), vsag::ErrorType::INVALID_BINARY);
        auto flipped = bytes;
        flipped[flipped.size() - 30] = static_cast<char>(flipped[flipped.size() - 30] ^ 0x5A);
        write_bytes(other, flipped);
        expect(options_for(other), vsag::ErrorType::INVALID_BINARY);
        write_bytes(other, bytes.substr(0, 1000) + bytes.substr(bytes.size() - 3000));
        CHECK_FALSE(vsag::analyze_index(options_for(other)).has_value());
    }
}

TEST_CASE("analyze_index validates crafted metadata and inputs", "[ut][analyze_index]") {
    fixtures::TempDir dir("analyze_index_crafted");
    const auto path = dir.path + "index";
    // A footer with the given metadata after a few body bytes.
    auto write_footer = [&](const Json& metadata) {
        std::ofstream out(path, std::ios::binary);
        out.write("body", 4);
        vsag::IOStreamWriter writer(out);
        vsag::Footer footer(
            std::make_shared<vsag::Metadata>(vsag::JsonType::Parse(metadata.dump())));
        footer.Write(writer);
    };
    auto expect = [&](vsag::ErrorType type) {
        auto error = analyze_error(options_for(path));
        INFO(error.message);
        CHECK(error.type == type);
    };

    write_footer({{"other", 1}});
    expect(vsag::ErrorType::INVALID_BINARY);
    write_footer({{"basic_info", {{"dim", DIM}}}});
    expect(vsag::ErrorType::UNSUPPORTED_INDEX);
    write_footer({{"basic_info", {{"index_param", "not json"}}}});
    expect(vsag::ErrorType::INVALID_BINARY);
    write_footer({{"basic_info", {{"index_param", R"({"type": "hgraph"})"}, {"metric", 9}}}});
    expect(vsag::ErrorType::INVALID_BINARY);
    write_footer({{"basic_info",
                   {{"index_param", R"({"type": "hgraph", "list": [{"io_params": {}}]})"},
                    {"dim", DIM},
                    {"metric", 0},
                    {"data_type", 0},
                    {"extra_info_size", 4}}}});
    CHECK_FALSE(vsag::analyze_index(options_for(path)).has_value());
    write_footer({{"_empty", true}});
    expect(vsag::ErrorType::INDEX_EMPTY);

    // Legacy files: the type is guessed from build_parameter when nothing names it. Only the
    // lazy_hgraph loader is used on garbage here: it checks its magic before reading sizes.
    write_bytes(path, std::string(64, 'x'));
    auto options = options_for(path);
    options.build_parameter = lazy_hgraph_parameter();
    auto error = analyze_error(options);
    CHECK(error.message.find("lazy_hgraph magic") != std::string::npos);

    // Streaming body corruption that only the loader notices.
    save_streaming(build_dense_index("hgraph", hgraph_parameter("cosine")), path);
    CHECK(analyze_ok(options_for(path))["metadata"]["metric"] == "cosine");
    auto bytes = read_bytes(path);
    for (uint64_t i = bytes.size() / 2; i < bytes.size() / 2 + 64; ++i) {
        bytes[i] = static_cast<char>(~bytes[i]);
    }
    write_bytes(path, bytes);
    CHECK_FALSE(vsag::analyze_index(options_for(path)).has_value());

    // Query, base and ground-truth files whose headers are invalid.
    save(build_dense_index("hgraph", hgraph_parameter()), path);
    const auto queries = dir.path + "queries";
    options = options_for(path);
    options.query_path = queries;
    options.query_data_type = "sparse";
    auto csr = make_sparse(2, 5);
    csr.indptr[0] = 1;
    write_csr(queries, csr);
    save(build_sindi_index(make_sparse(20, 3)), path);
    CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_BINARY);
    std::string header(24, '\0');
    const int64_t shape[3] = {1, 10, int64_t{1} << 62};
    std::memcpy(header.data(), shape, sizeof(shape));
    write_bytes(queries, header);
    CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_BINARY);
    write_csr(queries, make_sparse(2, 5));
    options.groundtruth_path = dir.path + "gt";
    write_bytes(options.groundtruth_path, "abc");
    CHECK(analyze_error(options).type == vsag::ErrorType::INVALID_BINARY);

    // Static SINDI analysis with both base_path and a search parameter.
    write_csr(dir.path + "base", make_sparse(20, 3));
    auto static_options = options_for(path);
    static_options.base_path = dir.path + "base";
    static_options.search_parameter = R"({"sindi": {}})";
    CHECK(analyze_ok(static_options)["static_analysis"]["total_count"] == 20);

    // Inputs that are not regular files.
    const auto fifo = dir.path + "fifo";
    REQUIRE(mkfifo(fifo.c_str(), 0600) == 0);
    CHECK(analyze_error(options_for(fifo)).type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("analyze_index reloads indexes with extra info", "[ut][analyze_index]") {
    fixtures::TempDir dir("analyze_index_extra_info");
    const auto path = dir.path + "index";
    auto parameter = Json::parse(hgraph_parameter());
    parameter["extra_info_size"] = 4;
    auto index = create_index("hgraph", parameter.dump());
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(COUNT, DIM);
    std::vector<char> extra_infos(COUNT * 4, 'e');
    auto base = vsag::Dataset::Make();
    base->NumElements(COUNT)
        ->Dim(DIM)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->ExtraInfos(extra_infos.data())
        ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    save(index, path);

    auto report = analyze_ok(options_for(path));
    CHECK(report["metadata"]["extra_info_size"] == 4);
    auto options = options_for(path);
    options.build_parameter = parameter.dump();
    CHECK(analyze_ok(options)["metadata"]["extra_info_size"] == 4);
}

TEST_CASE("analyze_index recreates only known index types", "[ut][analyze_index]") {
    auto common = vsag::JsonType::Parse(R"({"dtype": "float32", "metric_type": "l2", "dim": 4})");
    const auto index_param = vsag::JsonType::Parse("{}");
    CHECK_THROWS_AS(vsag::create_index_from_serialized_param("unknown", common, index_param),
                    vsag::VsagException);
}
