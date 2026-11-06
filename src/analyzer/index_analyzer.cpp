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

#include "analyzer/index_analyzer.h"

#include <fmt/format.h>

#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "algorithm/bruteforce/bruteforce_analyzer.h"
#include "data_type.h"
#include "inner_string_params.h"
#include "metric_type.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "vsag/vsag.h"

namespace vsag {

namespace {

constexpr const char* DEFAULT_BUILD_PARAM = "default";
constexpr const char* DEFAULT_SEARCH_PARAM = "default";
constexpr const char* EMPTY_QUERY_PATH = "empty";
constexpr const char* QUERY_DATA_TYPE_AUTO = "auto";
constexpr const char* QUERY_DATA_TYPE_DENSE = "dense";
constexpr const char* QUERY_DATA_TYPE_SPARSE = "sparse";

std::streamsize
ToStreamSize(uint64_t bytes) {
    return static_cast<std::streamsize>(bytes);
}

bool
IsSparseBuildParam(const JsonType& params) {
    return params.Contains("dtype") and params["dtype"].GetString() == "sparse";
}

bool
IsSindiInnerParam(const JsonType& params) {
    return params.Contains(SPARSE_TERM_ID_LIMIT) or params.Contains(SPARSE_DOC_PRUNE_RATIO) or
           params.Contains(USE_QUANTIZATION) or params.Contains(SPARSE_REMAP_TERM_IDS);
}

bool
IsSindiV2InnerParam(const JsonType& params) {
    return params.Contains("term_io") or params.Contains("rerank_layout");
}

std::string
InferIndexNameFromBuildParam(const JsonType& params) {
    if (params.Contains(TYPE_KEY)) {
        return params[TYPE_KEY].GetString();
    }
    if (IsSparseBuildParam(params)) {
        if (params.Contains(INDEX_PARAM) and IsSindiV2InnerParam(params[INDEX_PARAM])) {
            return INDEX_SINDI_V2;
        }
        return INDEX_SINDI;
    }
    if (params.Contains(INDEX_PARAM)) {
        auto index_param = params[INDEX_PARAM];
        if (index_param.Contains(TYPE_KEY)) {
            return index_param[TYPE_KEY].GetString();
        }
        if (index_param.Contains("base_quantization_type") or
            index_param.Contains("index_min_size")) {
            return INDEX_PYRAMID;
        }
        if (IsSindiV2InnerParam(index_param)) {
            return INDEX_SINDI_V2;
        }
        if (IsSindiInnerParam(index_param)) {
            return INDEX_SINDI;
        }
    }
    return "";
}

std::string
MetricTypeToString(MetricType type) {
    switch (type) {
        case MetricType::METRIC_TYPE_L2SQR:
            return "l2";
        case MetricType::METRIC_TYPE_IP:
            return "ip";
        case MetricType::METRIC_TYPE_COSINE:
            return "cosine";
    }
    return "unknown";
}

MetricType
MetricTypeFromString(const std::string& metric) {
    if (metric == "l2") {
        return MetricType::METRIC_TYPE_L2SQR;
    }
    if (metric == "ip") {
        return MetricType::METRIC_TYPE_IP;
    }
    if (metric == "cosine") {
        return MetricType::METRIC_TYPE_COSINE;
    }
    throw std::invalid_argument(fmt::format("unsupported metric_type '{}'", metric));
}

std::string
DataTypesToString(DataTypes type) {
    switch (type) {
        case DataTypes::DATA_TYPE_FLOAT:
            return "float32";
        case DataTypes::DATA_TYPE_INT8:
            return "int8";
        case DataTypes::DATA_TYPE_FP16:
            return "float16";
        case DataTypes::DATA_TYPE_BF16:
            return "bfloat16";
        case DataTypes::DATA_TYPE_SPARSE:
            return "sparse";
    }
    return "unknown";
}

JsonType
MakeStringArray(const std::vector<std::string>& values) {
    JsonType array = JsonType::Parse("[]");
    for (const auto& value : values) {
        JsonType item;
        item.SetString(value);
        array.AppendJson(item);
    }
    return array;
}

void
ReplaceWithMemoryIO(JsonType& io_params) {
    io_params[TYPE_KEY].SetString(IO_TYPE_VALUE_BLOCK_MEMORY_IO);
    io_params.Erase(IO_FILE_PATH_KEY);
}

void
IsolateDataCellStorage(JsonType& parent, const std::string& key) {
    JsonType data_cell_params = parent[key];
    if (data_cell_params.Contains(IO_PARAMS_KEY)) {
        JsonType io_params = data_cell_params[IO_PARAMS_KEY];
        ReplaceWithMemoryIO(io_params);
        data_cell_params[IO_PARAMS_KEY].SetJson(io_params);
    }
    if (data_cell_params.Contains(SUPPLEMENT_IO_PARAMS_KEY)) {
        JsonType io_params = data_cell_params[SUPPLEMENT_IO_PARAMS_KEY];
        ReplaceWithMemoryIO(io_params);
        data_cell_params[SUPPLEMENT_IO_PARAMS_KEY].SetJson(io_params);
    }
    parent[key].SetJson(data_cell_params);
}

JsonType
PrepareInternalBruteForceParam(const JsonType& serialized_param) {
    JsonType isolated_param = serialized_param;
    for (const auto* key : {BASE_CODES_KEY, PRECISE_CODES_KEY, RAW_VECTOR_KEY, EXTRA_INFO_KEY}) {
        if (isolated_param.Contains(key)) {
            IsolateDataCellStorage(isolated_param, key);
        }
    }
    return isolated_param;
}

void
IsolateExternalBruteForceParam(JsonType& build_param, bool is_warp) {
    JsonType index_param = build_param[INDEX_PARAM];
    index_param[BRUTE_FORCE_BASE_IO_TYPE].SetString(IO_TYPE_VALUE_BLOCK_MEMORY_IO);
    index_param.Erase(BRUTE_FORCE_BASE_FILE_PATH);
    if (not is_warp) {
        index_param[BRUTE_FORCE_PRECISE_IO_TYPE].SetString(IO_TYPE_VALUE_BLOCK_MEMORY_IO);
        index_param.Erase(BRUTE_FORCE_PRECISE_FILE_PATH);
    }
    build_param[INDEX_PARAM].SetJson(index_param);
}

JsonType
ParseJsonOrThrow(const std::string& text, const std::string& what) {
    try {
        return JsonType::Parse(text);
    } catch (const std::exception& error) {
        throw std::runtime_error(fmt::format("{} is not valid JSON: {}", what, error.what()));
    }
}

/**
 * @brief Parse a user-provided option string as JSON.
 *
 * Invalid arguments must be reported as std::invalid_argument (ValueError in Python) so they
 * are distinguishable from load or analysis failures.
 */
JsonType
ParseJsonArgument(const std::string& text, const std::string& what) {
    try {
        return JsonType::Parse(text);
    } catch (const std::exception& error) {
        throw std::invalid_argument(fmt::format("{} is not valid JSON: {}", what, error.what()));
    }
}

std::ifstream
OpenBinaryFile(const std::string& path, const std::string& what) {
    std::ifstream stream(path, std::ios::binary | std::ios::in);
    if (not stream.is_open()) {
        throw std::runtime_error(fmt::format("failed to open {} '{}'", what, path));
    }
    return stream;
}

DatasetPtr
LoadDenseQuery(const std::string& query_path) {
    auto stream = OpenBinaryFile(query_path, "query file");
    uint32_t rows = 0;
    uint32_t cols = 0;
    stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    stream.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (not stream.good()) {
        throw std::runtime_error(
            fmt::format("query file '{}' is truncated before its header", query_path));
    }
    if (rows == 0 or cols == 0) {
        throw std::runtime_error(
            fmt::format("query file '{}' has an invalid shape ({} x {})", query_path, rows, cols));
    }
    const uint64_t num_elements = static_cast<uint64_t>(rows) * cols;
    if (num_elements > std::numeric_limits<uint64_t>::max() / sizeof(float)) {
        throw std::runtime_error(
            fmt::format("query file '{}' declares an unsupported number of elements ({})",
                        query_path,
                        num_elements));
    }

    auto query_data = std::unique_ptr<float[]>(new float[num_elements]);
    stream.read(reinterpret_cast<char*>(query_data.get()),
                ToStreamSize(num_elements * sizeof(float)));
    if (not stream.good()) {
        throw std::runtime_error(fmt::format(
            "query file '{}' is truncated; expected {} float32 values", query_path, num_elements));
    }

    auto dataset = Dataset::Make();
    dataset->Float32Vectors(query_data.release())->Owner(true)->NumElements(rows)->Dim(cols);
    return dataset;
}

DatasetPtr
LoadSparseQuery(const std::string& query_path) {
    auto stream = OpenBinaryFile(query_path, "sparse query file");

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    stream.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    stream.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    if (not stream.good()) {
        throw std::runtime_error(
            fmt::format("sparse query file '{}' is truncated before its header", query_path));
    }
    if (rows <= 0 or cols <= 0 or nnz < 0) {
        throw std::runtime_error(
            fmt::format("sparse query file '{}' has an invalid header (rows={}, cols={}, nnz={})",
                        query_path,
                        rows,
                        cols,
                        nnz));
    }
    if (static_cast<uint64_t>(rows) > std::numeric_limits<uint64_t>::max() / sizeof(SparseVector)) {
        throw std::runtime_error(
            fmt::format("sparse query file '{}' declares too many rows ({})", query_path, rows));
    }

    std::vector<int64_t> indptr(static_cast<uint64_t>(rows) + 1);
    std::vector<int32_t> indices(static_cast<uint64_t>(nnz));
    std::vector<float> values(static_cast<uint64_t>(nnz));

    stream.read(reinterpret_cast<char*>(indptr.data()),
                ToStreamSize(sizeof(int64_t) * (static_cast<uint64_t>(rows) + 1)));
    stream.read(reinterpret_cast<char*>(indices.data()),
                ToStreamSize(sizeof(int32_t) * static_cast<uint64_t>(nnz)));
    stream.read(reinterpret_cast<char*>(values.data()),
                ToStreamSize(sizeof(float) * static_cast<uint64_t>(nnz)));
    if (not stream.good()) {
        throw std::runtime_error(fmt::format(
            "sparse query file '{}' is truncated; expected CSR arrays for {} rows and {} values",
            query_path,
            rows,
            nnz));
    }

    if (indptr[0] != 0 or indptr[static_cast<uint64_t>(rows)] != nnz) {
        throw std::runtime_error(
            fmt::format("sparse query file '{}' has inconsistent indptr bounds", query_path));
    }
    for (int64_t row = 0; row < rows; ++row) {
        const auto begin = indptr[static_cast<uint64_t>(row)];
        const auto end = indptr[static_cast<uint64_t>(row) + 1];
        if (begin < 0 or end > nnz or begin > end) {
            throw std::runtime_error(fmt::format(
                "sparse query file '{}' has non-monotonic indptr at row {}", query_path, row));
        }
    }
    for (int64_t offset = 0; offset < nnz; ++offset) {
        const auto column = indices[static_cast<uint64_t>(offset)];
        if (column < 0 or column >= cols) {
            throw std::runtime_error(
                fmt::format("sparse query file '{}' has an out-of-range column index at offset {}",
                            query_path,
                            offset));
        }
    }

    auto* sparse_vectors = new SparseVector[static_cast<uint64_t>(rows)];
    for (int64_t row = 0; row < rows; ++row) {
        const auto begin = indptr[static_cast<uint64_t>(row)];
        const auto end = indptr[static_cast<uint64_t>(row) + 1];
        const auto length = static_cast<uint64_t>(end - begin);
        sparse_vectors[row].len_ = static_cast<uint32_t>(length);
        if (length == 0) {
            continue;
        }
        sparse_vectors[row].ids_ = new uint32_t[length];
        sparse_vectors[row].vals_ = new float[length];
        for (uint64_t offset = 0; offset < length; ++offset) {
            const auto value_offset = static_cast<uint64_t>(begin) + offset;
            sparse_vectors[row].ids_[offset] = static_cast<uint32_t>(indices[value_offset]);
            sparse_vectors[row].vals_[offset] = values[value_offset];
        }
    }

    auto dataset = Dataset::Make();
    dataset->SparseVectors(sparse_vectors)->Owner(true)->NumElements(rows)->Dim(cols);
    return dataset;
}

DatasetPtr
LoadQuery(const std::string& query_path,
          DataTypes data_type,
          const std::string& query_data_type,
          std::string& effective_query_data_type) {
    if (query_data_type == QUERY_DATA_TYPE_DENSE) {
        effective_query_data_type = QUERY_DATA_TYPE_DENSE;
        return LoadDenseQuery(query_path);
    }
    if (query_data_type == QUERY_DATA_TYPE_SPARSE) {
        effective_query_data_type = QUERY_DATA_TYPE_SPARSE;
        return LoadSparseQuery(query_path);
    }
    if (data_type == DataTypes::DATA_TYPE_SPARSE) {
        effective_query_data_type = QUERY_DATA_TYPE_SPARSE;
        return LoadSparseQuery(query_path);
    }
    effective_query_data_type = QUERY_DATA_TYPE_DENSE;
    return LoadDenseQuery(query_path);
}

class IndexAnalyzerImpl {
public:
    explicit IndexAnalyzerImpl(IndexAnalysisOptions options) : options_(std::move(options)) {
    }

    JsonType
    Analyze() {
        this->ValidateOptions();
        this->LoadIndex();

        JsonType result;
        result["schema_version"].SetString(INDEX_ANALYSIS_SCHEMA_VERSION);
        result["index"].SetJson(this->BuildIndexMetadata());
        result["static_analysis"].SetJson(this->RunStaticAnalysis());
        if (this->HasQuery()) {
            result["dynamic_analysis"].SetJson(this->RunDynamicAnalysis());
        } else {
            result["dynamic_analysis"].SetJson(JsonType());
        }
        result["warnings"].SetJson(MakeStringArray(warnings_));
        result["compatibility"].SetJson(this->BuildCompatibility());
        return result;
    }

private:
    bool
    HasQuery() const {
        return not options_.query_path.empty() and options_.query_path != EMPTY_QUERY_PATH;
    }

    void
    ValidateOptions() {
        if (options_.index_path.empty()) {
            throw std::invalid_argument("index_path must not be empty");
        }
        if (options_.query_data_type != QUERY_DATA_TYPE_AUTO and
            options_.query_data_type != QUERY_DATA_TYPE_DENSE and
            options_.query_data_type != QUERY_DATA_TYPE_SPARSE) {
            throw std::invalid_argument(
                fmt::format("query_data_type must be one of [auto, dense, sparse], got '{}'",
                            options_.query_data_type));
        }
        if (options_.topk <= 0) {
            throw std::invalid_argument(
                fmt::format("topk must be a positive integer, got {}", options_.topk));
        }
        if (not this->HasQuery()) {
            if (not options_.groundtruth_path.empty()) {
                throw std::invalid_argument(
                    "groundtruth_path requires query_path; dynamic analysis was not requested");
            }
            if (not options_.save_groundtruth_path.empty()) {
                throw std::invalid_argument(
                    "save_groundtruth_path requires query_path; dynamic analysis was not "
                    "requested");
            }
        }
        if (options_.build_parameter.empty()) {
            options_.build_parameter = DEFAULT_BUILD_PARAM;
        }
        if (options_.search_parameter.empty()) {
            options_.search_parameter = DEFAULT_SEARCH_PARAM;
        }
        if (options_.search_parameter != DEFAULT_SEARCH_PARAM) {
            auto parsed = ParseJsonArgument(options_.search_parameter, "search_parameter");
            if (not parsed.IsObject()) {
                throw std::invalid_argument("search_parameter must be a JSON object");
            }
        }
    }

    void
    LoadIndex() {
        auto stream = OpenBinaryFile(options_.index_path, "index file");
        this->ParseMetadata(stream);
        this->ResolveIndexType();

        auto reopen = OpenBinaryFile(options_.index_path, "index file");
        if (options_.build_parameter != DEFAULT_BUILD_PARAM) {
            this->CreateIndexWithParam(reopen);
        } else {
            try {
                this->CreateIndexFromMetadata(reopen);
            } catch (const std::exception& error) {
                throw std::runtime_error(
                    fmt::format("failed to load index '{}' from serialized metadata: {}",
                                options_.index_path,
                                error.what()));
            }
        }

        if (not options_.base_path.empty()) {
            if (index_name_ != INDEX_SINDI and index_name_ != INDEX_SINDI_V2) {
                warnings_.emplace_back(fmt::format(
                    "base_path is only used by SINDI indexes; it is ignored for index type '{}'",
                    index_name_));
            } else if (index_name_ == INDEX_SINDI_V2 and not this->HasQuery()) {
                warnings_.emplace_back(
                    "base_path is ignored for static-only SINDI_V2 analysis; provide query_path "
                    "to use it during dynamic analysis");
            }
        }
    }

    void
    ParseMetadata(std::istream& stream) {
        FooterPtr footer;
        try {
            IOStreamReader reader(stream);
            footer = Footer::Parse(reader);
        } catch (const std::exception& error) {
            throw std::runtime_error(fmt::format(
                "failed to parse '{}' as a serialized VSAG index (file may be damaged or use an "
                "incompatible format): {}",
                options_.index_path,
                error.what()));
        }
        if (not footer) {
            throw std::runtime_error(
                fmt::format("failed to parse '{}' as a serialized VSAG index: footer is missing",
                            options_.index_path));
        }
        auto metadata = footer->GetMetadata();
        if (not metadata) {
            throw std::runtime_error(fmt::format(
                "serialized index '{}' does not contain any metadata", options_.index_path));
        }

        basic_info_ = metadata->Get(BASIC_INFO);
        if (not basic_info_.IsObject()) {
            if (metadata->EmptyIndex() and options_.build_parameter != DEFAULT_BUILD_PARAM) {
                warnings_.emplace_back(
                    "serialized empty index has no basic_info metadata; loading metadata uses "
                    "build_parameter");
                return;
            }
            throw std::runtime_error(
                fmt::format("serialized index '{}' is missing its '{}' metadata block",
                            options_.index_path,
                            BASIC_INFO));
        }

        const bool has_dim = basic_info_.Contains(DIM);
        const bool has_extra_info = basic_info_.Contains("extra_info_size");
        const bool has_data_type = basic_info_.Contains("data_type");
        const bool has_metric = basic_info_.Contains("metric");
        metadata_complete_ = has_dim and has_extra_info and has_data_type and has_metric;

        if (has_dim) {
            dim_ = basic_info_[DIM].GetInt();
        }
        if (has_extra_info) {
            extra_info_size_ = basic_info_["extra_info_size"].GetInt();
        }
        if (has_data_type) {
            data_type_ = static_cast<DataTypes>(basic_info_["data_type"].GetInt());
        }
        if (has_metric) {
            metric_type_ = static_cast<MetricType>(basic_info_["metric"].GetInt());
        }
        if (basic_info_.Contains(INDEX_PARAM)) {
            index_param_ = ParseJsonOrThrow(basic_info_[INDEX_PARAM].GetString(),
                                            "serialized index_param metadata");
        }
        if (not metadata_complete_) {
            warnings_.emplace_back(
                "serialized index metadata is incomplete; missing fields use analyzer defaults");
        }
    }

    void
    ResolveIndexType() {
        if (options_.build_parameter != DEFAULT_BUILD_PARAM) {
            auto parsed = ParseJsonArgument(options_.build_parameter, "build_parameter");
            if (not parsed.IsObject()) {
                throw std::invalid_argument("build_parameter must be a JSON object");
            }
            index_name_ = InferIndexNameFromBuildParam(parsed);
            if (IsSparseBuildParam(parsed)) {
                data_type_ = DataTypes::DATA_TYPE_SPARSE;
                metric_type_ = MetricType::METRIC_TYPE_IP;
                if (parsed.Contains(DIM)) {
                    dim_ = parsed[DIM].GetInt();
                }
            } else {
                if (parsed.Contains("metric_type")) {
                    metric_type_ = MetricTypeFromString(parsed["metric_type"].GetString());
                }
                if (dim_ == 0 and parsed.Contains(DIM)) {
                    dim_ = parsed[DIM].GetInt();
                }
            }
            if (index_name_.empty()) {
                throw std::invalid_argument(
                    "unable to infer the index type from build_parameter; provide "
                    "index_param.type");
            }
            metadata_source_ = "build_parameter";
            warnings_.emplace_back(
                "index type and loading parameters were taken from build_parameter instead of the "
                "serialized metadata");
            return;
        }

        if (not index_param_.IsObject()) {
            throw std::runtime_error(fmt::format(
                "serialized index '{}' does not embed '{}'; pass build_parameter to reload it",
                options_.index_path,
                INDEX_PARAM));
        }

        index_name_ = InferIndexNameFromBuildParam(index_param_);
        if (basic_info_.Contains("is_multi_vector") and basic_info_["is_multi_vector"].GetBool()) {
            index_name_ = INDEX_WARP;
        }
        if (index_name_.empty()) {
            index_name_ = InferIndexNameFromBuildParam(basic_info_);
        }
        if (index_name_.empty()) {
            throw std::runtime_error(fmt::format(
                "failed to infer the index type of '{}' from its serialized metadata; pass "
                "build_parameter",
                options_.index_path));
        }
        metadata_source_ = "serialized_metadata";
        if (not basic_info_.Contains("data_type")) {
            data_type_ = (index_name_ == INDEX_SINDI or index_name_ == INDEX_SINDI_V2)
                             ? DataTypes::DATA_TYPE_SPARSE
                             : DataTypes::DATA_TYPE_FLOAT;
        }
        if (not basic_info_.Contains("metric")) {
            metric_type_ = (index_name_ == INDEX_SINDI or index_name_ == INDEX_SINDI_V2)
                               ? MetricType::METRIC_TYPE_IP
                               : MetricType::METRIC_TYPE_L2SQR;
            warnings_.emplace_back(fmt::format(
                "serialized index metadata is missing 'metric'; using analyzer default '{}' for "
                "index type '{}'",
                MetricTypeToString(metric_type_),
                index_name_));
        }
    }

    void
    CreateIndexWithParam(std::istream& stream) {
        JsonType build_param = ParseJsonArgument(options_.build_parameter, "build_parameter");
        if (index_name_ == INDEX_BRUTE_FORCE or index_name_ == INDEX_WARP) {
            IsolateExternalBruteForceParam(build_param, index_name_ == INDEX_WARP);
        }
        auto create_result = Factory::CreateIndex(index_name_, build_param.Dump());
        if (not create_result.has_value()) {
            throw std::runtime_error(fmt::format(
                "failed to create index '{}': {}", index_name_, create_result.error().message));
        }
        this->Deserialize(create_result.value(), stream);
    }

    void
    CreateIndexFromMetadata(std::istream& stream) {
        JsonType build_param;
        build_param["dtype"].SetString(DataTypesToString(data_type_));
        build_param["metric_type"].SetString(MetricTypeToString(metric_type_));
        if (data_type_ != DataTypes::DATA_TYPE_SPARSE) {
            build_param["dim"].SetInt64(dim_);
        }
        if (extra_info_size_ > 0) {
            build_param["extra_info_size"].SetInt64(extra_info_size_);
        }
        if ((index_name_ == INDEX_BRUTE_FORCE or index_name_ == INDEX_WARP) and
            index_param_.Contains(BASE_CODES_KEY)) {
            build_param[INDEX_PARAM].SetJson(PrepareInternalBruteForceParam(index_param_));
            std::vector<std::string> ignored_param_keys;
            auto analyzer_index =
                CreateBruteForceIndexForAnalyzer(build_param, &ignored_param_keys);
            for (const auto& key : ignored_param_keys) {
                warnings_.emplace_back(
                    fmt::format("ignored serialized BruteForce parameter '{}'", key));
            }
            this->Deserialize(analyzer_index, stream);
            return;
        }

        build_param[INDEX_PARAM].SetJson(index_param_);
        if (index_name_ == INDEX_BRUTE_FORCE or index_name_ == INDEX_WARP) {
            IsolateExternalBruteForceParam(build_param, index_name_ == INDEX_WARP);
        }
        auto create_result = Factory::CreateIndex(index_name_, build_param.Dump());
        if (not create_result.has_value()) {
            throw std::runtime_error(
                fmt::format("failed to create index '{}' from serialized metadata: {}",
                            index_name_,
                            create_result.error().message));
        }
        this->Deserialize(create_result.value(), stream);
    }

    void
    Deserialize(const IndexPtr& index, std::istream& stream) {
        auto deserialize_result = index->Deserialize(stream);
        if (not deserialize_result.has_value()) {
            throw std::runtime_error(fmt::format("failed to deserialize index '{}' from '{}': {}",
                                                 index_name_,
                                                 options_.index_path,
                                                 deserialize_result.error().message));
        }
        index_ = index;
    }

    JsonType
    BuildIndexMetadata() const {
        // Some footers (e.g. Pyramid's) omit data_type / extra_info_size entirely, so those
        // members still hold the analyzer sentinels (float32 / 0) even though
        // CreateIndexWithParam() built the index from the explicit build_parameter. Backfill
        // the report from those effective build parameters only when the footer value is
        // absent; a footer value that is genuinely present always wins.
        std::string report_data_type = DataTypesToString(data_type_);
        int64_t report_extra_info_size = extra_info_size_;
        if (options_.build_parameter != DEFAULT_BUILD_PARAM) {
            auto parsed = ParseJsonArgument(options_.build_parameter, "build_parameter");
            if (parsed.IsObject()) {
                if (parsed.Contains("dtype") and not basic_info_.Contains("data_type")) {
                    report_data_type = parsed["dtype"].GetString();
                }
                if (parsed.Contains("extra_info_size") and
                    not basic_info_.Contains("extra_info_size")) {
                    report_extra_info_size = parsed["extra_info_size"].GetInt();
                }
            }
        }

        JsonType metadata;
        metadata["path"].SetString(options_.index_path);
        metadata["name"].SetString(index_name_);
        metadata["dim"].SetInt64(dim_);
        metadata["data_type"].SetString(report_data_type);
        metadata["metric"].SetString(MetricTypeToString(metric_type_));
        metadata["extra_info_size"].SetInt64(report_extra_info_size);
        metadata["metadata_source"].SetString(metadata_source_);
        metadata["metadata_complete"].SetBool(metadata_complete_);
        metadata["index_parameter"].SetJson(index_param_);
        return metadata;
    }

    SearchRequest
    MakeAnalyzeRequest(const std::string& search_parameter,
                       const std::string& base_path,
                       const std::string& groundtruth_path,
                       const std::string& save_groundtruth_path) const {
        SearchRequest request;
        const bool has_analyze_extras = not base_path.empty() or not groundtruth_path.empty() or
                                        not save_groundtruth_path.empty();
        const bool is_default_param = search_parameter == DEFAULT_SEARCH_PARAM;
        if (is_default_param and not has_analyze_extras) {
            // Downstream analyzers treat an empty parameter string as "use index defaults"
            // (see IVF::AnalyzeIndexBySearch and parse_sindi_search_json). Passing the
            // dumped default JSONType would yield "null", which is valid JSON but is not
            // special-cased and would break analysis for indexes requiring an object.
            request.params_str_ = "";
            return request;
        }

        JsonType params =
            is_default_param ? JsonType() : ParseJsonArgument(search_parameter, "search_parameter");
        if (not is_default_param and not params.IsObject()) {
            throw std::invalid_argument("search_parameter must be a JSON object");
        }
        if (has_analyze_extras) {
            params["analyze"].SetJson(JsonType());
            if (not base_path.empty()) {
                params["analyze"]["base_path"].SetString(base_path);
            }
            if (not groundtruth_path.empty()) {
                params["analyze"]["groundtruth_path"].SetString(groundtruth_path);
            }
            if (not save_groundtruth_path.empty()) {
                params["analyze"]["save_groundtruth_path"].SetString(save_groundtruth_path);
            }
        }
        request.params_str_ = params.Dump();
        return request;
    }

    JsonType
    RunStaticAnalysis() const {
        if (not index_->CheckFeature(IndexFeature::SUPPORT_GET_STATS)) {
            throw std::runtime_error(fmt::format(
                "index type '{}' does not support static analysis (GetStats)", index_name_));
        }

        std::string statistics;
        try {
            if (options_.base_path.empty() or index_name_ != INDEX_SINDI) {
                statistics = index_->GetStats();
            } else {
                if (not index_->CheckFeature(IndexFeature::SUPPORT_ANALYZE_INDEX_BY_SEARCH)) {
                    throw std::runtime_error(fmt::format(
                        "index type '{}' does not support base_path static analysis", index_name_));
                }
                SearchRequest request =
                    this->MakeAnalyzeRequest(options_.search_parameter, options_.base_path, "", "");
                request.topk_ = 0;
                statistics = index_->AnalyzeIndexBySearch(request);
            }
        } catch (const std::exception& error) {
            throw std::runtime_error(fmt::format(
                "static analysis failed for index '{}': {}", index_name_, error.what()));
        }

        JsonType section;
        section["supported"].SetBool(true);
        section["statistics"].SetJson(ParseJsonOrThrow(statistics, "static analysis result"));
        return section;
    }

    JsonType
    RunDynamicAnalysis() {
        if (not index_->CheckFeature(IndexFeature::SUPPORT_ANALYZE_INDEX_BY_SEARCH)) {
            throw std::runtime_error(fmt::format(
                "index type '{}' does not support dynamic analysis (AnalyzeIndexBySearch)",
                index_name_));
        }

        std::string effective_query_data_type;
        DatasetPtr query;
        try {
            query = LoadQuery(options_.query_path,
                              data_type_,
                              options_.query_data_type,
                              effective_query_data_type);
        } catch (const std::exception& error) {
            throw std::runtime_error(fmt::format(
                "failed to load query dataset '{}': {}", options_.query_path, error.what()));
        }

        std::string analysis;
        try {
            if (index_name_ == INDEX_SINDI_V2 and query->GetNumElements() != 1) {
                throw std::runtime_error(fmt::format(
                    "index type '{}' requires exactly one query row for dynamic analysis, got {}",
                    index_name_,
                    query->GetNumElements()));
            }
            SearchRequest request = this->MakeAnalyzeRequest(options_.search_parameter,
                                                             options_.base_path,
                                                             options_.groundtruth_path,
                                                             options_.save_groundtruth_path);
            request.query_ = query;
            request.topk_ = options_.topk;
            analysis = index_->AnalyzeIndexBySearch(request);
        } catch (const std::exception& error) {
            throw std::runtime_error(fmt::format(
                "dynamic analysis failed for index '{}': {}", index_name_, error.what()));
        }

        JsonType section;
        section["supported"].SetBool(true);
        section["query_path"].SetString(options_.query_path);
        section["query_data_type"].SetString(effective_query_data_type);
        section["topk"].SetInt64(options_.topk);
        section["search_parameter"].SetString(options_.search_parameter);
        section["results"].SetJson(ParseJsonOrThrow(analysis, "dynamic analysis result"));
        return section;
    }

    JsonType
    BuildCompatibility() const {
        JsonType compatibility;
        compatibility["schema_version"].SetString(INDEX_ANALYSIS_SCHEMA_VERSION);
        compatibility["serialization_format"].SetString("vsag-serialized-index");
        compatibility["static_analysis_supported"].SetBool(true);
        compatibility["dynamic_analysis_supported"].SetBool(
            index_->CheckFeature(IndexFeature::SUPPORT_ANALYZE_INDEX_BY_SEARCH));
        compatibility["metadata_complete"].SetBool(metadata_complete_);
        return compatibility;
    }

private:
    IndexAnalysisOptions options_;
    IndexPtr index_{nullptr};
    JsonType basic_info_;
    JsonType index_param_;
    std::string index_name_;
    std::string metadata_source_{"serialized_metadata"};
    int64_t dim_{0};
    int64_t extra_info_size_{0};
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};
    MetricType metric_type_{MetricType::METRIC_TYPE_L2SQR};
    bool metadata_complete_{false};
    std::vector<std::string> warnings_;
};

}  // namespace

JsonType
AnalyzeIndex(const IndexAnalysisOptions& options) {
    IndexAnalyzerImpl analyzer(options);
    return analyzer.Analyze();
}

}  // namespace vsag
