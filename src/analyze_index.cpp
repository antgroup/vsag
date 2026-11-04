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

#include "vsag/analyze_index.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "algorithm/lazy_hgraph/lazy_hgraph_format.h"
#include "data_type.h"
#include "factory/serialized_index_factory.h"
#include "inner_string_params.h"
#include "json_types.h"
#include "metric_type.h"
#include "storage/chunked_manifest.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "vsag/constants.h"
#include "vsag/dataset.h"
#include "vsag/factory.h"
#include "vsag/index.h"
#include "vsag/load_parameters.h"
#include "vsag/search_request.h"
#include "vsag/vsag.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

using Json = nlohmann::ordered_json;

constexpr const char* DEFAULT_PARAMETER = "default";
constexpr const char* QUERY_DATA_TYPE_AUTO = "auto";
constexpr const char* QUERY_DATA_TYPE_DENSE = "dense";
constexpr const char* QUERY_DATA_TYPE_SPARSE = "sparse";
constexpr const char* PARAMETER_SOURCE_FILE = "file";
constexpr const char* PARAMETER_SOURCE_USER = "user";
constexpr const char* ANALYZE_PARAMETER_KEY = "analyze";
constexpr const char* UNCOMPRESSED_CODEC = "none";
constexpr uint64_t MAGIC_SIZE = 8;
constexpr uint64_t DENSE_HEADER_SIZE = 2 * sizeof(uint32_t);
constexpr uint64_t CSR_HEADER_SIZE = 3 * sizeof(int64_t);
constexpr uint64_t GROUND_TRUTH_HEADER_SIZE = 2 * sizeof(uint32_t);
constexpr uint64_t MAX_ECHOED_OPTION_LENGTH = 200;
constexpr int64_t DEFAULT_ANALYSIS_EF_SEARCH = 100;

enum class FileFormat { FOOTER, CHUNKED, STREAMING, LEGACY };

const char*
format_name(FileFormat format) {
    switch (format) {
        case FileFormat::FOOTER:
            return "footer";
        case FileFormat::CHUNKED:
            return "chunked";
        case FileFormat::STREAMING:
            return "streaming";
        case FileFormat::LEGACY:
            return "legacy";
    }
    return "unknown";
}

const char*
metric_name(MetricType metric) {
    switch (metric) {
        case MetricType::METRIC_TYPE_L2SQR:
            return METRIC_L2;
        case MetricType::METRIC_TYPE_IP:
            return METRIC_IP;
        case MetricType::METRIC_TYPE_COSINE:
            return METRIC_COSINE;
    }
    return "unknown";
}

[[noreturn]] void
fail(ErrorType type, std::string message) {
    throw VsagException(type, std::move(message));
}

bool
is_default(const std::string& value) {
    return value.empty() or value == DEFAULT_PARAMETER;
}

std::string
echo_option(const std::string& value) {
    if (value.size() <= MAX_ECHOED_OPTION_LENGTH) {
        return value;
    }
    return value.substr(0, MAX_ECHOED_OPTION_LENGTH) + "...";
}

Json
parse_object_option(const std::string& value, const char* option) {
    auto parsed = Json::parse(value, nullptr, false);
    if (parsed.is_discarded() or not parsed.is_object()) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format(
                 "{} must be a JSON object or \"default\", got: {}", option, echo_option(value)));
    }
    return parsed;
}

bool
is_sparse_index(const std::string& index_type) {
    return index_type == INDEX_SINDI or index_type == INDEX_SINDI_V2;
}

bool
contains_any(const Json& json, std::initializer_list<const char*> keys) {
    return json.is_object() and std::any_of(keys.begin(), keys.end(), [&json](const char* key) {
               return json.contains(key);
           });
}

std::string
string_member(const Json& json, const char* key) {
    if (json.is_object()) {
        auto iter = json.find(key);
        if (iter != json.end() and iter->is_string()) {
            return iter->get<std::string>();
        }
    }
    return "";
}

// Recognizes the index type of an index_param object. HGraph, IVF, BruteForce, SIMQ and
// LazyHGraph record "type"; Pyramid, SINDI and SINDI_V2 are recognized by keys only they use.
std::string
infer_type_from_index_param(const Json& index_param) {
    auto type = string_member(index_param, TYPE_KEY);
    if (not type.empty()) {
        return type;
    }
    if (contains_any(index_param, {"term_io", "rerank_layout"})) {
        return INDEX_SINDI_V2;
    }
    if (contains_any(index_param,
                     {SPARSE_TERM_ID_LIMIT,
                      SPARSE_DOC_PRUNE_RATIO,
                      USE_QUANTIZATION,
                      SPARSE_REMAP_TERM_IDS})) {
        return INDEX_SINDI;
    }
    if (contains_any(index_param, {INDEX_MIN_SIZE})) {
        return INDEX_PYRAMID;
    }
    return "";
}

// lazy_hgraph takes its parameters from "lazy_hgraph" when "index_param" is absent.
const Json&
user_index_param(const Json& build_parameter) {
    static const Json empty = Json::object();
    for (const auto* key : {INDEX_PARAM, INDEX_LAZY_HGRAPH}) {
        auto iter = build_parameter.find(key);
        if (iter != build_parameter.end() and iter->is_object()) {
            return *iter;
        }
    }
    return empty;
}

// The type named by build_parameter itself: "type" or "index_param.type".
std::string
explicit_user_type(const Json& build_parameter) {
    auto type = string_member(build_parameter, TYPE_KEY);
    return type.empty() ? string_member(user_index_param(build_parameter), TYPE_KEY) : type;
}

// Used only when neither the file nor build_parameter names the type.
std::string
guess_user_type(const Json& build_parameter) {
    if (build_parameter.contains(INDEX_LAZY_HGRAPH)) {
        return INDEX_LAZY_HGRAPH;
    }
    const auto& index_param = user_index_param(build_parameter);
    if (string_member(build_parameter, PARAMETER_DTYPE) == DATATYPE_SPARSE) {
        return contains_any(index_param, {"term_io", "rerank_layout"}) ? INDEX_SINDI_V2
                                                                       : INDEX_SINDI;
    }
    return infer_type_from_index_param(index_param);
}

std::optional<DataTypes>
parse_data_type(const std::string& name) {
    for (auto type : {DataTypes::DATA_TYPE_FLOAT,
                      DataTypes::DATA_TYPE_INT8,
                      DataTypes::DATA_TYPE_FP16,
                      DataTypes::DATA_TYPE_BF16,
                      DataTypes::DATA_TYPE_SPARSE}) {
        if (name == ToString(type)) {
            return type;
        }
    }
    return std::nullopt;
}

std::optional<MetricType>
parse_metric(const std::string& name) {
    for (auto metric : {MetricType::METRIC_TYPE_L2SQR,
                        MetricType::METRIC_TYPE_IP,
                        MetricType::METRIC_TYPE_COSINE}) {
        if (name == metric_name(metric)) {
            return metric;
        }
    }
    return std::nullopt;
}

// Status checks give precise errors before any parser sees the file.
uint64_t
require_input_file(const std::string& path, const char* option) {
    std::error_code error;
    const auto status = std::filesystem::status(path, error);
    if (status.type() == std::filesystem::file_type::not_found) {
        fail(ErrorType::MISSING_FILE, fmt::format("{} does not exist: {}", option, path));
    }
    if (error) {
        fail(ErrorType::READ_ERROR,
             fmt::format("cannot access {} {}: {}", option, path, error.message()));
    }
    if (std::filesystem::is_directory(status)) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("{} must be a file, but {} is a directory", option, path));
    }
    if (not std::filesystem::is_regular_file(status)) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("{} must be a regular file: {}", option, path));
    }
    const auto size = std::filesystem::file_size(path, error);
    if (error) {
        fail(ErrorType::READ_ERROR,
             fmt::format("cannot read the size of {} {}: {}", option, path, error.message()));
    }
    std::ifstream probe(path, std::ios::binary);
    if (not probe.is_open()) {
        fail(ErrorType::READ_ERROR, fmt::format("cannot open {} for reading: {}", option, path));
    }
    return static_cast<uint64_t>(size);
}

void
require_output_file(const std::string& path, const char* option) {
    std::error_code error;
    const std::filesystem::path target(path);
    if (std::filesystem::is_directory(target, error)) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("{} must be a file path, but {} is a directory", option, path));
    }
    auto parent = target.parent_path();
    if (parent.empty()) {
        parent = ".";
    }
    if (not std::filesystem::is_directory(parent, error)) {
        fail(ErrorType::MISSING_FILE,
             fmt::format("the directory of {} does not exist: {}", option, parent.string()));
    }
}

void
read_exact(std::istream& in, void* data, uint64_t size, const std::string& path) {
    in.read(static_cast<char*>(data), static_cast<std::streamsize>(size));
    if (static_cast<uint64_t>(in.gcount()) != size) {
        fail(ErrorType::READ_ERROR, fmt::format("unexpected end of file while reading {}", path));
    }
}

// Returns false when a * b + c overflows uint64_t.
bool
checked_multiply_add(uint64_t a, uint64_t b, uint64_t c, uint64_t& result) {
    if (a != 0 and b > (std::numeric_limits<uint64_t>::max() - c) / a) {
        return false;
    }
    result = a * b + c;
    return true;
}

struct CsrShape {
    int64_t rows{0};
    int64_t cols{0};
    int64_t nnz{0};
};

CsrShape
read_csr_header(std::istream& in, uint64_t file_size, const std::string& path, const char* option) {
    if (file_size < CSR_HEADER_SIZE) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("{} {} is not a sparse CSR file: it has {} bytes, fewer than the "
                         "{}-byte header (int64 rows, int64 cols, int64 nnz)",
                         option,
                         path,
                         file_size,
                         CSR_HEADER_SIZE));
    }
    CsrShape shape;
    read_exact(in, &shape.rows, sizeof(shape.rows), path);
    read_exact(in, &shape.cols, sizeof(shape.cols), path);
    read_exact(in, &shape.nnz, sizeof(shape.nnz), path);
    if (shape.rows <= 0 or shape.cols <= 0 or shape.nnz < 0) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("{} {} has an invalid sparse CSR header: rows={}, cols={}, nnz={}",
                         option,
                         path,
                         shape.rows,
                         shape.cols,
                         shape.nnz));
    }
    uint64_t expected = 0;
    const auto rows = static_cast<uint64_t>(shape.rows);
    const auto nnz = static_cast<uint64_t>(shape.nnz);
    bool valid = checked_multiply_add(rows + 1, sizeof(int64_t), CSR_HEADER_SIZE, expected);
    valid =
        valid and checked_multiply_add(nnz, sizeof(int32_t) + sizeof(float), expected, expected);
    if (not valid or expected != file_size) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("{} {} has {} bytes, but its sparse CSR header (rows={}, cols={}, nnz={}) "
                         "describes {} bytes; the file is truncated or is not a VSAG sparse CSR "
                         "file",
                         option,
                         path,
                         file_size,
                         shape.rows,
                         shape.cols,
                         shape.nnz,
                         valid ? std::to_string(expected) : std::string("more than 2^64")));
    }
    return shape;
}

struct QuerySet {
    DatasetPtr dataset;
    int64_t rows{0};
    int64_t cols{0};
};

QuerySet
load_dense_queries(const std::string& path, uint64_t file_size) {
    std::ifstream in(path, std::ios::binary);
    if (file_size < DENSE_HEADER_SIZE) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("query_path {} is not a dense query file: it has {} bytes, fewer than "
                         "the {}-byte header (uint32 rows, uint32 dim)",
                         path,
                         file_size,
                         DENSE_HEADER_SIZE));
    }
    uint32_t rows = 0;
    uint32_t dim = 0;
    read_exact(in, &rows, sizeof(rows), path);
    read_exact(in, &dim, sizeof(dim), path);
    uint64_t expected = 0;
    const auto count = static_cast<uint64_t>(rows) * dim;
    const bool valid = checked_multiply_add(count, sizeof(float), DENSE_HEADER_SIZE, expected);
    if (rows == 0 or dim == 0 or not valid or expected != file_size) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("query_path {} has {} bytes, but its dense header (rows={}, dim={}) "
                         "describes {} bytes; expected uint32 rows, uint32 dim and "
                         "float32 data[rows * dim]",
                         path,
                         file_size,
                         rows,
                         dim,
                         valid ? std::to_string(expected) : std::string("more than 2^64")));
    }
    auto values = std::make_unique<float[]>(count);
    read_exact(in, values.get(), count * sizeof(float), path);
    QuerySet queries;
    queries.rows = rows;
    queries.cols = dim;
    queries.dataset = Dataset::Make();
    queries.dataset->NumElements(rows)->Dim(dim)->Float32Vectors(values.release())->Owner(true);
    return queries;
}

QuerySet
load_sparse_queries(const std::string& path, uint64_t file_size) {
    std::ifstream in(path, std::ios::binary);
    const auto shape = read_csr_header(in, file_size, path, "query_path");
    const auto rows = static_cast<uint64_t>(shape.rows);
    const auto nnz = static_cast<uint64_t>(shape.nnz);
    std::vector<int64_t> indptr(rows + 1);
    std::vector<int32_t> indices(nnz);
    std::vector<float> values(nnz);
    read_exact(in, indptr.data(), indptr.size() * sizeof(int64_t), path);
    read_exact(in, indices.data(), indices.size() * sizeof(int32_t), path);
    read_exact(in, values.data(), values.size() * sizeof(float), path);
    if (indptr.front() != 0 or indptr.back() != shape.nnz) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("query_path {} has invalid CSR row offsets: indptr[0]={} and "
                         "indptr[rows]={} must be 0 and nnz={}",
                         path,
                         indptr.front(),
                         indptr.back(),
                         shape.nnz));
    }
    for (uint64_t row = 0; row < rows; ++row) {
        if (indptr[row] > indptr[row + 1]) {
            fail(ErrorType::INVALID_BINARY,
                 fmt::format("query_path {} has decreasing CSR row offsets at row {}", path, row));
        }
    }
    for (uint64_t offset = 0; offset < nnz; ++offset) {
        if (indices[offset] < 0 or indices[offset] >= shape.cols) {
            fail(ErrorType::INVALID_BINARY,
                 fmt::format("query_path {} has column index {} outside [0, {})",
                             path,
                             indices[offset],
                             shape.cols));
        }
    }

    // The dataset owns the vectors and their arrays, exactly like the sparse datasets created by
    // the SINDI analyzer. It owns them before they are filled, so that a failed allocation
    // releases the rows that are already done.
    QuerySet queries;
    queries.rows = shape.rows;
    queries.cols = shape.cols;
    queries.dataset = Dataset::Make();
    auto* vectors = new SparseVector[rows]();
    queries.dataset->SparseVectors(vectors)->NumElements(shape.rows)->Dim(shape.cols)->Owner(true);
    for (uint64_t row = 0; row < rows; ++row) {
        const auto begin = static_cast<uint64_t>(indptr[row]);
        const auto length = static_cast<uint64_t>(indptr[row + 1]) - begin;
        if (length > std::numeric_limits<uint32_t>::max()) {
            fail(ErrorType::INVALID_BINARY,
                 fmt::format("query_path {} has {} values in row {}, more than a sparse vector "
                             "can hold",
                             path,
                             length,
                             row));
        }
        vectors[row].ids_ = new uint32_t[length];
        vectors[row].vals_ = new float[length];
        vectors[row].len_ = static_cast<uint32_t>(length);
        for (uint64_t i = 0; i < length; ++i) {
            vectors[row].ids_[i] = static_cast<uint32_t>(indices[begin + i]);
            vectors[row].vals_[i] = values[begin + i];
        }
    }
    return queries;
}

void
check_ground_truth_file(const std::string& path,
                        uint64_t file_size,
                        int64_t query_count,
                        int64_t required_topk) {
    std::ifstream in(path, std::ios::binary);
    if (file_size < GROUND_TRUTH_HEADER_SIZE) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("groundtruth_path {} is not a ground-truth file: it has {} bytes, fewer "
                         "than the {}-byte header (uint32 rows, uint32 topk)",
                         path,
                         file_size,
                         GROUND_TRUTH_HEADER_SIZE));
    }
    uint32_t rows = 0;
    uint32_t topk = 0;
    read_exact(in, &rows, sizeof(rows), path);
    read_exact(in, &topk, sizeof(topk), path);
    uint64_t expected = 0;
    const bool valid = checked_multiply_add(static_cast<uint64_t>(rows) * topk,
                                            sizeof(int32_t) + sizeof(float),
                                            GROUND_TRUTH_HEADER_SIZE,
                                            expected);
    if (not valid or expected != file_size) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("groundtruth_path {} has {} bytes, but its header (rows={}, topk={}) "
                         "describes {} bytes; expected uint32 rows, uint32 topk, "
                         "int32 ids[rows * topk] and float32 distances[rows * topk]",
                         path,
                         file_size,
                         rows,
                         topk,
                         valid ? std::to_string(expected) : std::string("more than 2^64")));
    }
    if (static_cast<int64_t>(rows) != query_count or static_cast<int64_t>(topk) < required_topk) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("groundtruth_path {} holds {} neighbors for each of {} queries, but the "
                         "analysis needs {} neighbors for each of the {} queries in query_path",
                         path,
                         topk,
                         rows,
                         required_topk,
                         query_count));
    }
}

struct FileInspection {
    FileFormat format{FileFormat::LEGACY};
    bool lazy_container{false};
    bool empty_index{false};
    Json metadata = Json::object();
    std::string chunked_codec;
    uint64_t chunk_size{0};
    std::vector<std::string> streaming_blocks;
};

FileInspection
inspect_index_file(const std::string& path, uint64_t file_size) {
    FileInspection inspection;
    std::ifstream in(path, std::ios::binary);
    if (file_size >= MAGIC_SIZE) {
        char magic[MAGIC_SIZE] = {};
        read_exact(in, magic, MAGIC_SIZE, path);
        if (std::memcmp(magic, SERIAL_STREAM_MAGIC, MAGIC_SIZE) == 0) {
            in.seekg(0);
            auto streaming = Index::GetStreamingMetadata(in);
            if (not streaming.has_value()) {
                fail(streaming.error().type,
                     fmt::format("{} starts with the VSAG streaming header, but the header or the "
                                 "block layout is invalid: {}",
                                 path,
                                 streaming.error().message));
            }
            inspection.format = FileFormat::STREAMING;
            inspection.metadata = Json::parse(streaming->metadata_json, nullptr, false);
            for (const auto& block : streaming->blocks) {
                inspection.streaming_blocks.emplace_back(block.name);
            }
            inspection.empty_index = inspection.metadata.value("_empty", false);
            return inspection;
        }
        uint64_t container_magic = 0;
        std::memcpy(&container_magic, magic, sizeof(container_magic));
        inspection.lazy_container = container_magic == LAZY_HGRAPH_MAGIC;
    }

    in.clear();
    in.seekg(0);
    IOStreamReader reader(in);
    FooterPtr footer{nullptr};
    try {
        footer = Footer::Parse(reader);
    } catch (const std::exception& error) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("the footer of {} is damaged: {}", path, error.what()));
    }
    if (footer == nullptr) {
        return inspection;
    }
    auto footer_metadata = footer->GetMetadata();
    inspection.metadata = Json::parse(footer_metadata->ToString(), nullptr, false);
    if (not inspection.metadata.is_object()) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("the footer metadata of {} is not a JSON object", path));
    }
    inspection.empty_index = footer_metadata->EmptyIndex();
    inspection.format = FileFormat::FOOTER;
    auto layout = inspection.metadata.find(CHUNKED_LAYOUT_KEY);
    if (layout != inspection.metadata.end() and layout->is_object()) {
        inspection.format = FileFormat::CHUNKED;
        try {
            auto manifest = ChunkedManifest::FromJson(footer_metadata->Get(CHUNKED_LAYOUT_KEY));
            inspection.chunked_codec = manifest.codec_;
            inspection.chunk_size = manifest.chunk_size_;
        } catch (const std::exception& error) {
            fail(ErrorType::INVALID_BINARY,
                 fmt::format(
                     "the chunked layout recorded in {} is invalid: {}", path, error.what()));
        }
    }
    return inspection;
}

struct ResolvedIndex {
    std::string type;
    std::string source{PARAMETER_SOURCE_FILE};
    Json index_param = Json::object();
    Json build_parameter;  // user-supplied build_parameter, null when loaded from the file
    Json basic_info;       // recorded basic_info without index_param, null for legacy files
    DataTypes data_type{DataTypes::DATA_TYPE_FLOAT};
    MetricType metric{MetricType::METRIC_TYPE_L2SQR};
    int64_t dim{0};
    int64_t extra_info_size{0};
    Json defaulted_fields = Json::object();
    Json load_index_param = Json::object();  // index_param with memory IO, used for loading
    Json rewritten_io = Json::array();
};

const Json*
recorded_basic_info(const FileInspection& inspection) {
    auto iter = inspection.metadata.find(BASIC_INFO);
    if (iter == inspection.metadata.end() or not iter->is_object()) {
        return nullptr;
    }
    return &(*iter);
}

Json
recorded_index_param(const Json& basic_info, const std::string& path) {
    auto iter = basic_info.find(INDEX_PARAM);
    if (iter == basic_info.end()) {
        return Json::object();
    }
    Json index_param;
    if (iter->is_string()) {
        index_param = Json::parse(iter->get<std::string>(), nullptr, false);
    }
    if (not index_param.is_object()) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("the recorded basic_info.index_param of {} is not a JSON object", path));
    }
    return index_param;
}

// Type recorded by the file, or an empty string when the file does not identify it.
std::string
recorded_index_type(const FileInspection& inspection, const std::string& path) {
    if (inspection.lazy_container) {
        return INDEX_LAZY_HGRAPH;
    }
    if (inspection.format == FileFormat::STREAMING) {
        return string_member(inspection.metadata, "index_name");
    }
    const auto* basic_info = recorded_basic_info(inspection);
    if (basic_info == nullptr) {
        return "";
    }
    if (basic_info->value("is_multi_vector", false)) {
        return INDEX_WARP;
    }
    return infer_type_from_index_param(recorded_index_param(*basic_info, path));
}

template <typename Enum>
std::optional<Enum>
recorded_enum(const Json& basic_info, const char* key, int64_t max_value) {
    auto iter = basic_info.find(key);
    if (iter == basic_info.end()) {
        return std::nullopt;
    }
    if (not iter->is_number_integer() or iter->get<int64_t>() < 0 or
        iter->get<int64_t>() > max_value) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("the recorded basic_info.{} value {} is not valid", key, iter->dump()));
    }
    return static_cast<Enum>(iter->get<int64_t>());
}

bool
is_io_parameter_key(const std::string& key) {
    // "io_params" configures the datacells and "supplement_io_params" the supplement of split
    // RaBitQ codes; SINDI_V2 configures its posting and rerank storage with "term_io" and
    // "rerank_io".
    return key == IO_PARAMS_KEY or key == SUPPLEMENT_IO_PARAMS_KEY or key == "term_io" or
           key == "rerank_io";
}

// Split RaBitQ codes accept a memory supplement only with the same IO type as the codes next to it.
const char*
supplement_memory_io(const Json& parent) {
    const auto codes_io = parent.find(IO_PARAMS_KEY);
    const bool block_memory = codes_io != parent.end() and
                              string_member(*codes_io, TYPE_KEY) == IO_TYPE_VALUE_BLOCK_MEMORY_IO;
    return block_memory ? IO_TYPE_VALUE_BLOCK_MEMORY_IO : IO_TYPE_VALUE_MEMORY_IO;
}

// File-backed IO (buffer_io, async_io, mmap_io, uring_io) recorded in the parameters would make
// Deserialize write the index data to the recorded file paths, and reader_io would wait for an
// external reader. Serialized files always carry the data inline, so memory IO is used instead.
// Most streaming loaders bind reader_io themselves, so only file-backed IO is replaced for them;
// the HGraph loader needs a reader from the caller, and the analyzer has none to offer.
void
use_memory_io(Json& json, const std::string& pointer, bool replace_reader_io, Json& rewritten) {
    if (json.is_array()) {
        for (uint64_t i = 0; i < json.size(); ++i) {
            use_memory_io(json[i], pointer + "/" + std::to_string(i), replace_reader_io, rewritten);
        }
        return;
    }
    if (not json.is_object()) {
        return;
    }
    for (const auto& [key, value] : json.items()) {
        auto child = pointer;
        child.append("/").append(key);
        if (is_io_parameter_key(key) and value.is_object()) {
            const auto io_type = string_member(value, TYPE_KEY);
            const bool in_memory = io_type.empty() or io_type == IO_TYPE_VALUE_MEMORY_IO or
                                   io_type == IO_TYPE_VALUE_BLOCK_MEMORY_IO;
            if (not in_memory and (replace_reader_io or io_type != IO_TYPE_VALUE_READER_IO)) {
                rewritten.push_back({{"parameter", child}, {"recorded_io_type", io_type}});
                value[TYPE_KEY] = key == SUPPLEMENT_IO_PARAMS_KEY ? supplement_memory_io(json)
                                                                  : IO_TYPE_VALUE_MEMORY_IO;
                value.erase(IO_FILE_PATH_KEY);
            }
            continue;
        }
        use_memory_io(value, child, replace_reader_io, rewritten);
    }
}

ResolvedIndex
resolve_from_file(const FileInspection& inspection, const std::string& path) {
    const auto* basic_info = recorded_basic_info(inspection);
    if (basic_info == nullptr) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("the metadata of {} has no basic_info object, so the index cannot be "
                         "reloaded from it; pass build_parameter with the original build "
                         "parameters",
                         path));
    }
    ResolvedIndex resolved;
    resolved.type = recorded_index_type(inspection, path);
    if (resolved.type.empty()) {
        fail(ErrorType::UNSUPPORTED_INDEX,
             fmt::format("cannot determine the index type of {} from its metadata; pass "
                         "build_parameter with the original build parameters and \"type\"",
                         path));
    }
    resolved.index_param = recorded_index_param(*basic_info, path);
    resolved.basic_info = *basic_info;
    resolved.basic_info.erase(INDEX_PARAM);
    const bool sparse = is_sparse_index(resolved.type);

    auto data_type = recorded_enum<DataTypes>(
        *basic_info, "data_type", static_cast<int64_t>(DataTypes::DATA_TYPE_SPARSE));
    if (data_type.has_value()) {
        resolved.data_type = data_type.value();
    } else if (sparse) {
        resolved.data_type = DataTypes::DATA_TYPE_SPARSE;
    } else {
        resolved.defaulted_fields[PARAMETER_DTYPE] = DATATYPE_FLOAT32;
    }
    auto metric = recorded_enum<MetricType>(
        *basic_info, "metric", static_cast<int64_t>(MetricType::METRIC_TYPE_COSINE));
    if (metric.has_value()) {
        resolved.metric = metric.value();
    } else if (sparse) {
        resolved.metric = MetricType::METRIC_TYPE_IP;
    } else {
        resolved.defaulted_fields[PARAMETER_METRIC_TYPE] = METRIC_L2;
    }
    auto dim = basic_info->find(DIM);
    if (dim != basic_info->end() and dim->is_number_integer()) {
        resolved.dim = dim->get<int64_t>();
    } else if (not sparse) {
        fail(ErrorType::UNSUPPORTED_INDEX_OPERATION,
             fmt::format("{} does not record the vector dimension of its {} index ({} files "
                         "written by Serialize(std::ostream&) keep only the index parameters), so "
                         "it cannot be reloaded from its metadata; pass build_parameter with the "
                         "original build parameters",
                         path,
                         resolved.type,
                         resolved.type));
    }
    // Without a recorded size the index has no extra info; a wrong guess would fail to load
    // rather than skew the analysis, so it is not reported as an assumption.
    auto extra_info_size = basic_info->find(EXTRA_INFO_SIZE);
    if (extra_info_size != basic_info->end() and extra_info_size->is_number_integer()) {
        resolved.extra_info_size = extra_info_size->get<int64_t>();
    }
    resolved.load_index_param = resolved.index_param;
    use_memory_io(resolved.load_index_param,
                  "/" + std::string(INDEX_PARAM),
                  inspection.format != FileFormat::STREAMING or resolved.type == INDEX_HGRAPH,
                  resolved.rewritten_io);
    return resolved;
}

ResolvedIndex
resolve_from_user(const FileInspection& inspection,
                  const std::string& path,
                  const std::string& build_parameter) {
    if (inspection.format == FileFormat::STREAMING) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("build_parameter cannot be used with {}: streaming files always embed "
                         "their build parameters and are loaded with them; remove "
                         "build_parameter",
                         path));
    }
    ResolvedIndex resolved;
    resolved.source = PARAMETER_SOURCE_USER;
    resolved.build_parameter = parse_object_option(build_parameter, "build_parameter");
    const auto user_type = explicit_user_type(resolved.build_parameter);
    const auto file_type = recorded_index_type(inspection, path);
    if (not user_type.empty() and not file_type.empty() and user_type != file_type) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("build_parameter names a {} index, but {} contains a {} index",
                         user_type,
                         path,
                         file_type));
    }
    resolved.type = not user_type.empty() ? user_type : file_type;
    if (resolved.type.empty()) {
        resolved.type = guess_user_type(resolved.build_parameter);
    }
    if (resolved.type.empty()) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("cannot determine the index type: {} does not record it and "
                         "build_parameter has neither \"type\" nor \"index_param.type\"; add "
                         "\"type\": \"<index type>\" to build_parameter",
                         path));
    }
    if (const auto* basic_info = recorded_basic_info(inspection)) {
        resolved.basic_info = *basic_info;
        resolved.basic_info.erase(INDEX_PARAM);
    }
    resolved.index_param = user_index_param(resolved.build_parameter);
    const bool sparse = is_sparse_index(resolved.type);
    auto data_type = parse_data_type(string_member(resolved.build_parameter, PARAMETER_DTYPE));
    resolved.data_type =
        data_type.value_or(sparse ? DataTypes::DATA_TYPE_SPARSE : DataTypes::DATA_TYPE_FLOAT);
    auto metric = parse_metric(string_member(resolved.build_parameter, PARAMETER_METRIC_TYPE));
    resolved.metric =
        metric.value_or(sparse ? MetricType::METRIC_TYPE_IP : MetricType::METRIC_TYPE_L2SQR);
    auto dim = resolved.build_parameter.find(PARAMETER_DIM);
    if (dim != resolved.build_parameter.end() and dim->is_number_integer()) {
        resolved.dim = dim->get<int64_t>();
    }
    auto extra_info_size = resolved.build_parameter.find(EXTRA_INFO_SIZE);
    if (extra_info_size != resolved.build_parameter.end() and
        extra_info_size->is_number_integer()) {
        resolved.extra_info_size = extra_info_size->get<int64_t>();
    }
    return resolved;
}

// Runs one step of the analysis and names it in the error message.
template <typename Function>
auto
run_step(const std::string& step, Function&& function) {
    try {
        return function();
    } catch (const VsagException& error) {
        throw VsagException(error.error_.type, fmt::format("{}: {}", step, error.error_.message));
    } catch (const std::bad_alloc&) {
        throw;
    } catch (const std::exception& error) {
        throw VsagException(ErrorType::UNKNOWN_ERROR, fmt::format("{}: {}", step, error.what()));
    }
}

// Index::Load keeps the IO recorded in a streaming file unless the load parameters override it.
// Only HGraph applies overrides before it creates the storage of its codes and raw vectors. The
// BruteForce and WARP load parameters only select a load plan: their storage is still created
// from the recorded parameters and would write to the recorded path, so those files are refused.
LoadParameters
streaming_load_parameters(const ResolvedIndex& resolved, const std::string& path) {
    const std::vector<std::pair<std::string, const char*>> overrides = {
        {fmt::format("/{}/{}/{}", INDEX_PARAM, BASE_CODES_KEY, IO_PARAMS_KEY), HGRAPH_BASE_IO_TYPE},
        {fmt::format("/{}/{}/{}", INDEX_PARAM, PRECISE_CODES_KEY, IO_PARAMS_KEY),
         HGRAPH_PRECISE_IO_TYPE},
        {fmt::format("/{}/{}/{}", INDEX_PARAM, RAW_VECTOR_KEY, IO_PARAMS_KEY), RAW_VECTOR_IO_TYPE},
    };
    LoadParameters parameters;
    for (const auto& rewritten : resolved.rewritten_io) {
        const auto parameter = rewritten.at("parameter").get<std::string>();
        const char* key = nullptr;
        for (const auto& [pointer, override_key] : overrides) {
            if (resolved.type == INDEX_HGRAPH and parameter == pointer) {
                key = override_key;
            }
        }
        if (key == nullptr) {
            fail(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                 fmt::format("{} records {} for {}; loading it would use the storage recorded "
                             "in the file, and the {} streaming loader cannot load that "
                             "component into memory instead; load it with Index::Load and "
                             "suitable load parameters, or re-serialize it with memory IO",
                             path,
                             rewritten.at("recorded_io_type").get<std::string>(),
                             parameter,
                             resolved.type));
        }
        parameters.Set(key, IO_TYPE_VALUE_MEMORY_IO);
    }
    return parameters;
}

IndexPtr
load_index(const std::string& path, const FileInspection& inspection, ResolvedIndex& resolved) {
    if (inspection.format == FileFormat::CHUNKED and
        inspection.chunked_codec != UNCOMPRESSED_CODEC) {
        fail(ErrorType::UNSUPPORTED_INDEX_OPERATION,
             fmt::format("{} is a chunked index file compressed with codec \"{}\"; decompressing "
                         "it needs the DeserializeReader of the application that wrote it, so it "
                         "cannot be analyzed from the file alone; re-serialize the index without "
                         "compression",
                         path,
                         inspection.chunked_codec));
    }
    std::ifstream in(path, std::ios::binary);
    if (inspection.format == FileFormat::STREAMING) {
        auto loaded = Index::Load(in, streaming_load_parameters(resolved, path));
        if (not loaded.has_value()) {
            fail(loaded.error().type,
                 fmt::format("failed to load the streaming index file {}: {}",
                             path,
                             loaded.error().message));
        }
        return loaded.value();
    }

    IndexPtr index{nullptr};
    if (resolved.source == PARAMETER_SOURCE_USER) {
        auto parameters = resolved.build_parameter;
        parameters.erase(TYPE_KEY);
        auto created = Factory::CreateIndex(resolved.type, parameters.dump());
        if (not created.has_value()) {
            fail(created.error().type,
                 fmt::format("cannot create a {} index from build_parameter: {}",
                             resolved.type,
                             created.error().message));
        }
        index = created.value();
    } else {
        Json common = {{PARAMETER_DTYPE, ToString(resolved.data_type)},
                       {PARAMETER_METRIC_TYPE, metric_name(resolved.metric)}};
        if (not is_sparse_index(resolved.type)) {
            common[PARAMETER_DIM] = resolved.dim;
        }
        if (resolved.extra_info_size > 0) {
            common[EXTRA_INFO_SIZE] = resolved.extra_info_size;
        }
        auto common_param = JsonType::Parse(common.dump());
        const auto index_param = JsonType::Parse(resolved.load_index_param.dump());
        index = run_step(
            fmt::format("cannot recreate the {} index recorded in {}", resolved.type, path), [&]() {
                return create_index_from_serialized_param(resolved.type, common_param, index_param);
            });
    }

    auto loaded = index->Deserialize(in);
    if (not loaded.has_value()) {
        fail(loaded.error().type,
             fmt::format("failed to deserialize {} as a {} index: {}; the file may be truncated "
                         "or damaged, or the build parameters do not match the ones used to "
                         "write it",
                         path,
                         resolved.type,
                         loaded.error().message));
    }
    return index;
}

Json
parse_report_section(const std::string& text, const char* section) {
    auto parsed = Json::parse(text, nullptr, false);
    if (not parsed.is_object()) {
        fail(ErrorType::INTERNAL_ERROR, fmt::format("{} did not return a JSON object", section));
    }
    return parsed;
}

// Search parameters of dynamic analysis. HGraph and Pyramid searches require ef_search and
// SINDI_V2 searches require their own member, so their defaults spell those out; the other index
// types search with their own defaults for {}.
Json
analysis_search_parameter(const ResolvedIndex& resolved, const AnalyzeIndexOptions& options) {
    if (not is_default(options.search_parameter)) {
        return parse_object_option(options.search_parameter, "search_parameter");
    }
    const Json ef_search = {
        {HGRAPH_PARAMETER_EF_RUNTIME, std::max<int64_t>(DEFAULT_ANALYSIS_EF_SEARCH, options.topk)}};
    if (resolved.type == INDEX_HGRAPH or resolved.type == INDEX_LAZY_HGRAPH) {
        return {{INDEX_HGRAPH, ef_search}};
    }
    if (resolved.type == INDEX_PYRAMID) {
        return {{INDEX_PYRAMID, ef_search}};
    }
    if (resolved.type == INDEX_SINDI_V2) {
        return {{INDEX_SINDI_V2, Json::object()}};
    }
    return Json::object();
}

// SINDI reads its base_path, groundtruth_path and save_groundtruth_path inputs from the
// "analyze" member of the search parameters.
std::string
with_analyze_inputs(const Json& search_parameter, const AnalyzeIndexOptions& options) {
    Json parameter = search_parameter.is_object() ? search_parameter : Json::object();
    Json inputs = Json::object();
    if (not options.base_path.empty()) {
        inputs["base_path"] = options.base_path;
    }
    if (not options.groundtruth_path.empty()) {
        inputs["groundtruth_path"] = options.groundtruth_path;
    }
    if (not options.save_groundtruth_path.empty()) {
        inputs["save_groundtruth_path"] = options.save_groundtruth_path;
    }
    parameter[ANALYZE_PARAMETER_KEY] = inputs;
    return parameter.dump();
}

Json
run_static_analysis(const IndexPtr& index,
                    const ResolvedIndex& resolved,
                    const AnalyzeIndexOptions& options) {
    return run_step("static analysis failed", [&]() {
        if (resolved.type == INDEX_SINDI and not options.base_path.empty()) {
            // SINDI computes the metrics that need original vectors from the base dataset passed
            // through the analyze parameters of a query-less request.
            SearchRequest request;
            request.topk_ = 0;
            request.params_str_ = with_analyze_inputs(
                is_default(options.search_parameter)
                    ? Json::object()
                    : parse_object_option(options.search_parameter, "search_parameter"),
                options);
            return parse_report_section(index->AnalyzeIndexBySearch(request),
                                        "AnalyzeIndexBySearch()");
        }
        return parse_report_section(index->GetStats(), "GetStats()");
    });
}

struct DynamicAnalysis {
    DatasetPtr queries;
    int64_t topk{0};
    std::string search_parameter;
    std::string step;
    Json report = Json::object();
};

// Validates the dynamic-analysis inputs and loads the queries before the static analysis, which
// can take a long time on large indexes, so that invalid inputs fail fast.
void
prepare_dynamic_analysis(const IndexPtr& index,
                         const ResolvedIndex& resolved,
                         const AnalyzeIndexOptions& options,
                         uint64_t query_file_size,
                         DynamicAnalysis& dynamic) {
    const bool sparse_index = resolved.data_type == DataTypes::DATA_TYPE_SPARSE;
    std::string query_type = options.query_data_type;
    if (query_type == QUERY_DATA_TYPE_AUTO) {
        query_type = sparse_index ? QUERY_DATA_TYPE_SPARSE : QUERY_DATA_TYPE_DENSE;
    }
    if (query_type == QUERY_DATA_TYPE_SPARSE and not sparse_index) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("query_data_type \"sparse\" needs a sparse index, but the {} index "
                         "stores {} vectors",
                         resolved.type,
                         ToString(resolved.data_type)));
    }
    if (query_type == QUERY_DATA_TYPE_DENSE and sparse_index) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("the {} index stores sparse vectors; pass a sparse CSR query file with "
                         "query_data_type \"sparse\" or \"auto\"",
                         resolved.type));
    }
    if (resolved.type == INDEX_WARP or resolved.type == INDEX_SIMQ) {
        fail(ErrorType::UNSUPPORTED_INDEX_OPERATION,
             fmt::format("dynamic analysis of {} indexes needs multi-vector queries, which the "
                         "query file formats cannot express; static analysis is available",
                         resolved.type));
    }
    if (not sparse_index and resolved.data_type != DataTypes::DATA_TYPE_FLOAT) {
        fail(ErrorType::UNSUPPORTED_INDEX_OPERATION,
             fmt::format("dense query files hold float32 vectors, but the {} index stores {} "
                         "vectors; dynamic analysis supports float32 dense indexes and sparse "
                         "indexes",
                         resolved.type,
                         ToString(resolved.data_type)));
    }
    const bool analyze_inputs = not options.base_path.empty() or
                                not options.groundtruth_path.empty() or
                                not options.save_groundtruth_path.empty();
    if (analyze_inputs and resolved.type != INDEX_SINDI) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("base_path, groundtruth_path and save_groundtruth_path are only used by "
                         "sindi indexes; the {} index would ignore them",
                         resolved.type));
    }

    auto queries = query_type == QUERY_DATA_TYPE_DENSE
                       ? load_dense_queries(options.query_path, query_file_size)
                       : load_sparse_queries(options.query_path, query_file_size);
    if (query_type == QUERY_DATA_TYPE_DENSE and queries.cols != resolved.dim) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("query_path {} holds {}-dimensional vectors, but the index dimension is "
                         "{}",
                         options.query_path,
                         queries.cols,
                         resolved.dim));
    }
    if (not options.groundtruth_path.empty()) {
        const auto required_topk = std::min(options.topk, index->GetNumElements());
        check_ground_truth_file(options.groundtruth_path,
                                require_input_file(options.groundtruth_path, "groundtruth_path"),
                                queries.rows,
                                required_topk);
    }

    const auto search_parameter = analysis_search_parameter(resolved, options);
    dynamic.queries = queries.dataset;
    dynamic.topk = options.topk;
    dynamic.search_parameter =
        analyze_inputs ? with_analyze_inputs(search_parameter, options) : search_parameter.dump();
    // Query files carry vectors only; Pyramid needs per-query hierarchy paths when the root of
    // the selected hierarchy is not built, and reports that as a missing "query_path".
    dynamic.step = resolved.type == INDEX_PYRAMID
                       ? "dynamic analysis failed (query files cannot attach Pyramid hierarchy "
                         "paths to queries; analyze such hierarchies through the C++ API with "
                         "Dataset::Paths)"
                       : "dynamic analysis failed";
    dynamic.report = {
        {"query_path", options.query_path},
        {"query_data_type", query_type},
        {"query_count", queries.rows},
        {query_type == QUERY_DATA_TYPE_DENSE ? "query_dim" : "query_cols", queries.cols},
        {"topk", options.topk},
        {"search_parameter", search_parameter},
        {"search_parameter_source",
         is_default(options.search_parameter) ? DEFAULT_PARAMETER : PARAMETER_SOURCE_USER}};
    if (not options.base_path.empty()) {
        dynamic.report["base_path"] = options.base_path;
    }
    if (not options.groundtruth_path.empty()) {
        dynamic.report["groundtruth_path"] = options.groundtruth_path;
    }
    if (not options.save_groundtruth_path.empty()) {
        dynamic.report["save_groundtruth_path"] = options.save_groundtruth_path;
    }
}

Json
run_dynamic_analysis(const IndexPtr& index, DynamicAnalysis& dynamic) {
    SearchRequest request;
    request.query_ = dynamic.queries;
    request.topk_ = dynamic.topk;
    request.params_str_ = dynamic.search_parameter;
    dynamic.report["result"] = run_step(dynamic.step, [&]() {
        return parse_report_section(index->AnalyzeIndexBySearch(request), "AnalyzeIndexBySearch()");
    });
    return std::move(dynamic.report);
}

void
validate_options(const AnalyzeIndexOptions& options) {
    if (options.index_path.empty()) {
        fail(ErrorType::INVALID_ARGUMENT, "index_path is required");
    }
    if (options.query_data_type != QUERY_DATA_TYPE_AUTO and
        options.query_data_type != QUERY_DATA_TYPE_DENSE and
        options.query_data_type != QUERY_DATA_TYPE_SPARSE) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format(R"(query_data_type must be "auto", "dense" or "sparse", got "{}")",
                         echo_option(options.query_data_type)));
    }
    if (not is_default(options.build_parameter)) {
        parse_object_option(options.build_parameter, "build_parameter");
    }
    if (not is_default(options.search_parameter)) {
        parse_object_option(options.search_parameter, "search_parameter");
    }
    const bool dynamic = not options.query_path.empty();
    if (dynamic and options.topk <= 0) {
        fail(ErrorType::INVALID_ARGUMENT,
             fmt::format("topk must be positive, got {}", options.topk));
    }
    if (not dynamic and not options.groundtruth_path.empty()) {
        fail(ErrorType::INVALID_ARGUMENT,
             "groundtruth_path is only used by dynamic analysis; pass query_path as well");
    }
    if (not dynamic and not options.save_groundtruth_path.empty()) {
        fail(ErrorType::INVALID_ARGUMENT,
             "save_groundtruth_path is only used by dynamic analysis; pass query_path as well");
    }
    if (not options.groundtruth_path.empty() and not options.save_groundtruth_path.empty()) {
        fail(ErrorType::INVALID_ARGUMENT,
             "save_groundtruth_path saves generated ground truth, which is not generated when "
             "groundtruth_path is given; pass only one of them");
    }
}

void
add_warning(Json& warnings, const char* code, const std::string& message) {
    warnings.push_back({{"code", code}, {"message", message}});
}

Json
analyze(const AnalyzeIndexOptions& options) {
    validate_options(options);
    const auto file_size = require_input_file(options.index_path, "index_path");
    if (file_size == 0) {
        fail(ErrorType::INVALID_BINARY, fmt::format("index file {} is empty", options.index_path));
    }
    // Check every input before the potentially long index load.
    uint64_t query_file_size = 0;
    if (not options.query_path.empty()) {
        query_file_size = require_input_file(options.query_path, "query_path");
    }
    if (not options.base_path.empty()) {
        const auto base_size = require_input_file(options.base_path, "base_path");
        std::ifstream base(options.base_path, std::ios::binary);
        read_csr_header(base, base_size, options.base_path, "base_path");
    }
    if (not options.groundtruth_path.empty()) {
        require_input_file(options.groundtruth_path, "groundtruth_path");
    }
    if (not options.save_groundtruth_path.empty()) {
        require_output_file(options.save_groundtruth_path, "save_groundtruth_path");
    }

    const auto inspection = inspect_index_file(options.index_path, file_size);
    Json warnings = Json::array();
    ResolvedIndex resolved;
    if (not is_default(options.build_parameter)) {
        resolved = resolve_from_user(inspection, options.index_path, options.build_parameter);
        if (inspection.format == FileFormat::LEGACY) {
            add_warning(warnings,
                        "legacy_format",
                        "the file has no VSAG footer; it was loaded as a legacy file with the "
                        "given build_parameter");
        } else if (recorded_basic_info(inspection) != nullptr) {
            add_warning(warnings,
                        "build_parameter_override",
                        "the index was loaded with the given build_parameter instead of the "
                        "parameters recorded in the file");
        }
    } else if (inspection.format == FileFormat::LEGACY) {
        fail(ErrorType::INVALID_BINARY,
             fmt::format("{} has neither a VSAG footer nor a VSAG streaming header: it is not a "
                         "VSAG index file, it is truncated or damaged, or it was written by a "
                         "VSAG version older than 0.15; for such legacy files pass "
                         "build_parameter with the original build parameters",
                         options.index_path));
    } else if (inspection.lazy_container) {
        fail(ErrorType::UNSUPPORTED_INDEX_OPERATION,
             fmt::format("{} was written by a lazy_hgraph index, which does not record its "
                         "lazy_hgraph parameters (transition_threshold and hgraph); pass "
                         "build_parameter with the original build parameters",
                         options.index_path));
    } else if (inspection.empty_index and inspection.format != FileFormat::STREAMING and
               recorded_basic_info(inspection) == nullptr) {
        fail(ErrorType::INDEX_EMPTY,
             fmt::format("{} holds an index that was serialized while empty, which records no "
                         "index type or parameters; pass build_parameter to analyze it",
                         options.index_path));
    } else {
        resolved = resolve_from_file(inspection, options.index_path);
    }
    if (not resolved.defaulted_fields.empty()) {
        add_warning(warnings,
                    "metadata_defaults",
                    fmt::format("the file does not record {}; the analyzer assumed these values, "
                                "so pass build_parameter if they differ from the original build",
                                resolved.defaulted_fields.dump()));
    }
    if (not resolved.rewritten_io.empty()) {
        add_warning(warnings,
                    "memory_io",
                    "the recorded parameters use file-backed IO; the index was loaded into memory "
                    "instead and no recorded file path was opened");
    }

    auto index = load_index(options.index_path, inspection, resolved);
    const auto num_elements = index->GetNumElements();
    if (num_elements == 0) {
        add_warning(warnings, "empty_index", "the index contains no vectors");
    }

    Json report = Json::object();
    report["schema_version"] = ANALYZE_INDEX_SCHEMA_VERSION;
    Json metadata = {{"index_path", options.index_path},
                     {"file_size_bytes", file_size},
                     {"index_type", resolved.type},
                     {"dim", is_sparse_index(resolved.type) ? Json(nullptr) : Json(resolved.dim)},
                     {"data_type", ToString(resolved.data_type)},
                     {"metric", metric_name(resolved.metric)},
                     {"extra_info_size", resolved.extra_info_size},
                     {"num_elements", num_elements},
                     {"memory_usage_bytes", index->GetMemoryUsage()},
                     {"parameter_source", resolved.source},
                     {"index_param", resolved.index_param},
                     {"build_parameter", resolved.build_parameter},
                     {"basic_info", resolved.basic_info}};
    report["metadata"] = std::move(metadata);
    const bool has_queries = not options.query_path.empty();
    DynamicAnalysis dynamic;
    if (has_queries) {
        prepare_dynamic_analysis(index, resolved, options, query_file_size, dynamic);
    }
    report["static_analysis"] = run_static_analysis(index, resolved, options);
    report["dynamic_analysis"] = has_queries ? run_dynamic_analysis(index, dynamic) : Json(nullptr);
    report["warnings"] = std::move(warnings);

    Json compatibility = {{"vsag_version", version()},
                          {"serialization_format", format_name(inspection.format)},
                          {"lazy_hgraph_container", inspection.lazy_container},
                          {"metadata_complete", resolved.defaulted_fields.empty()},
                          {"defaulted_fields", resolved.defaulted_fields},
                          {"memory_io_overrides", resolved.rewritten_io}};
    if (inspection.format == FileFormat::CHUNKED) {
        compatibility["chunked_codec"] = inspection.chunked_codec;
        compatibility["chunk_size"] = inspection.chunk_size;
    }
    if (inspection.format == FileFormat::STREAMING) {
        compatibility["streaming_blocks"] = inspection.streaming_blocks;
    }
    report["compatibility"] = std::move(compatibility);
    return report;
}

}  // namespace

tl::expected<std::string, Error>
analyze_index(const AnalyzeIndexOptions& options) {
    try {
        return analyze(options).dump(-1, ' ', false, Json::error_handler_t::replace);
    } catch (const VsagException& error) {
        return tl::unexpected(error.error_);
    } catch (const std::bad_alloc& error) {
        return tl::unexpected(Error(
            ErrorType::NO_ENOUGH_MEMORY,
            fmt::format("not enough memory to analyze {}: {}", options.index_path, error.what())));
    } catch (const std::exception& error) {
        return tl::unexpected(
            Error(ErrorType::UNKNOWN_ERROR,
                  fmt::format("failed to analyze {}: {}", options.index_path, error.what())));
    }
}

}  // namespace vsag
