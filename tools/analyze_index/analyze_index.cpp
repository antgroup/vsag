
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

#include <vsag/vsag.h>

#include <argparse/argparse.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

#include "algorithm/hgraph.h"
#include "algorithm/ivf.h"
#include "algorithm/pyramid.h"
#include "algorithm/pyramid_zparameters.h"
#include "algorithm/sindi/sindi.h"
#include "algorithm/sindi/sindi_parameter.h"
#include "index/index_impl.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"

using namespace vsag;

constexpr static const char* DEFAULT_BUILD_PARAM = "default";
constexpr static const char* DEFAULT_SEARCH_PARAM = "default";
constexpr static const char* EMPTY_QUERY_PATH = "empty";
constexpr static const char* EMPTY_OPTIONAL_PATH = "";
constexpr static const char* QUERY_DATA_TYPE_AUTO = "auto";

std::streamsize
to_stream_size(size_t bytes) {
    return static_cast<std::streamsize>(bytes);
}

bool
is_sparse_build_param(const JsonType& params) {
    return params.Contains("dtype") && params["dtype"].GetString() == "sparse";
}

bool
is_sindi_inner_param(const JsonType& params) {
    return params.Contains(SPARSE_TERM_ID_LIMIT) || params.Contains(SPARSE_DOC_PRUNE_RATIO) ||
           params.Contains(USE_QUANTIZATION) || params.Contains(SPARSE_REMAP_TERM_IDS);
}

std::string
infer_index_name_from_build_param(const JsonType& params) {
    if (params.Contains(TYPE_KEY)) {
        return params[TYPE_KEY].GetString();
    }
    if (is_sparse_build_param(params)) {
        return INDEX_SINDI;
    }
    if (params.Contains(INDEX_PARAM)) {
        auto index_param = params[INDEX_PARAM];
        if (index_param.Contains(TYPE_KEY)) {
            return index_param[TYPE_KEY].GetString();
        }
        if (index_param.Contains("base_quantization_type") ||
            index_param.Contains("index_min_size")) {
            return INDEX_PYRAMID;
        }
        if (is_sindi_inner_param(index_param)) {
            return INDEX_SINDI;
        }
    }
    return "";
}

inline const std::string
MetricTypeToString(MetricType type) {
    switch (type) {
        case MetricType::METRIC_TYPE_L2SQR:
            return "l2";
        case MetricType::METRIC_TYPE_IP:
            return "ip";
        case MetricType::METRIC_TYPE_COSINE:
            return "cosine";
        default:
            return "unknown";
    }
}

std::string
DataTypesToString(DataTypes type) {
    switch (type) {
        case DataTypes::DATA_TYPE_FLOAT:
            return "float";
        case DataTypes::DATA_TYPE_INT8:
            return "int8";
        case DataTypes::DATA_TYPE_FP16:
            return "fp16";
        case DataTypes::DATA_TYPE_SPARSE:
            return "sparse";
        default:
            return "unknown";
    }
}

void
parse_args(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument<std::string>("--index_path", "-i")
        .required()
        .help("The index path for load or save");

    parser.add_argument<std::string>("--build_parameter", "-bp")
        .default_value(DEFAULT_BUILD_PARAM)
        .help(
            "The parameter for build index, "
            "if not set, will use default parameter in index file");

    parser.add_argument<std::string>("--query_path", "-qp")
        .default_value(EMPTY_QUERY_PATH)
        .help("The query dataset path, if not set, will not do query analysis");

    parser.add_argument<std::string>("--query_data_type")
        .default_value(std::string(QUERY_DATA_TYPE_AUTO))
        .help("Query dataset type: auto, dense, sparse");

    parser.add_argument<std::string>("--base_path")
        .default_value(std::string(EMPTY_OPTIONAL_PATH))
        .help("Optional sparse base dataset path for SINDI analyze");

    parser.add_argument<std::string>("--groundtruth_path")
        .default_value(std::string(EMPTY_OPTIONAL_PATH))
        .help("Optional ground truth path for analyze");

    parser.add_argument<std::string>("--save_groundtruth_path")
        .default_value(std::string(EMPTY_OPTIONAL_PATH))
        .help("Optional output path for generated ground truth");

    parser.add_argument<std::string>("--search_parameter", "-sp")
        .default_value(DEFAULT_SEARCH_PARAM)
        .help("The parameter for search, if not set, will use default parameter");

    parser.add_argument("--topk", "-k")
        .default_value(100)
        .help("The topk for search")
        .scan<'i', int>();

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
    }
}

DatasetPtr
load_dense_query(const std::string& query_path) {
    std::fstream in_stream(query_path, std::ios::binary | std::ios::in);
    if (not in_stream.is_open()) {
        logger::error("Failed to open query file: {}", query_path);
        return nullptr;
    }
    uint32_t rows, cols;
    in_stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in_stream.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    size_t num_elements = static_cast<size_t>(rows) * cols;
    auto dataset = Dataset::Make();
    auto query_data = new float[num_elements];
    dataset->Float32Vectors(query_data)->Owner(true)->NumElements(rows)->Dim(cols);
    in_stream.read(reinterpret_cast<char*>(query_data), num_elements * sizeof(float));
    return dataset;
}

DatasetPtr
load_sparse_query(const std::string& query_path) {
    std::fstream in_stream(query_path, std::ios::binary | std::ios::in);
    if (not in_stream.is_open()) {
        logger::error("Failed to open sparse query file: {}", query_path);
        return nullptr;
    }

    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    in_stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in_stream.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    in_stream.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    if (rows < 0 || cols < 0 || nnz < 0) {
        logger::error("Invalid sparse query header in file: {}", query_path);
        return nullptr;
    }

    std::vector<int64_t> indptr(static_cast<size_t>(rows) + 1);
    std::vector<int32_t> indices(static_cast<size_t>(nnz));
    std::vector<float> values(static_cast<size_t>(nnz));
    auto* sparse_vectors = new SparseVector[static_cast<size_t>(rows)];

    in_stream.read(reinterpret_cast<char*>(indptr.data()),
                   to_stream_size(sizeof(int64_t) * (static_cast<size_t>(rows) + 1)));
    in_stream.read(reinterpret_cast<char*>(indices.data()),
                   to_stream_size(sizeof(int32_t) * static_cast<size_t>(nnz)));
    in_stream.read(reinterpret_cast<char*>(values.data()),
                   to_stream_size(sizeof(float) * static_cast<size_t>(nnz)));
    if (not in_stream.good()) {
        logger::error("Failed to read sparse query CSR arrays from file: {}", query_path);
        delete[] sparse_vectors;
        return nullptr;
    }

    if (indptr[0] != 0 || indptr[static_cast<size_t>(rows)] != nnz) {
        logger::error("Invalid sparse query indptr bounds in file: {}", query_path);
        delete[] sparse_vectors;
        return nullptr;
    }
    for (int64_t row = 0; row < rows; ++row) {
        if (indptr[static_cast<size_t>(row)] > indptr[static_cast<size_t>(row + 1)] ||
            indptr[static_cast<size_t>(row)] < 0 || indptr[static_cast<size_t>(row + 1)] > nnz) {
            logger::error("Invalid sparse query indptr monotonicity in file: {}", query_path);
            delete[] sparse_vectors;
            return nullptr;
        }
    }
    for (int64_t offset = 0; offset < nnz; ++offset) {
        if (indices[static_cast<size_t>(offset)] < 0 ||
            indices[static_cast<size_t>(offset)] >= cols) {
            logger::error("Invalid sparse query column index in file: {}", query_path);
            delete[] sparse_vectors;
            return nullptr;
        }
    }

    for (int64_t row = 0; row < rows; ++row) {
        auto begin = indptr[static_cast<size_t>(row)];
        auto end = indptr[static_cast<size_t>(row + 1)];
        auto len = end - begin;
        sparse_vectors[row].len_ = static_cast<uint32_t>(len);
        sparse_vectors[row].ids_ = new uint32_t[static_cast<size_t>(len)];
        sparse_vectors[row].vals_ = new float[static_cast<size_t>(len)];
        for (int64_t offset = 0; offset < len; ++offset) {
            auto value_offset = static_cast<size_t>(begin + offset);
            sparse_vectors[row].ids_[offset] = static_cast<uint32_t>(indices[value_offset]);
            sparse_vectors[row].vals_[offset] = values[value_offset];
        }
    }

    auto dataset = Dataset::Make();
    dataset->SparseVectors(sparse_vectors)->Owner(true)->NumElements(rows)->Dim(cols);
    return dataset;
}

DatasetPtr
load_query(const std::string& query_path, DataTypes data_type, const std::string& query_data_type) {
    if (query_data_type == "dense") {
        return load_dense_query(query_path);
    }
    if (query_data_type == "sparse") {
        return load_sparse_query(query_path);
    }
    if (data_type == DataTypes::DATA_TYPE_SPARSE) {
        return load_sparse_query(query_path);
    }
    return load_dense_query(query_path);
}

class AnalyzedIndex {
public:
    AnalyzedIndex(const std::string& build_param) : build_param_(build_param) {
    }

    bool
    LoadIndex(const std::string& index_path) {
        index_path_ = index_path;
        std::fstream in_stream(index_path, std::ios::binary | std::ios::in);
        IOStreamReader reader(in_stream);
        if (not parse_reader(reader)) {
            return false;
        }
        if (build_param_ != DEFAULT_BUILD_PARAM) {
            std::fstream new_stream(index_path_, std::ios::binary | std::ios::in);
            if (not create_index_with_param(index_name_, new_stream)) {
                return false;
            }
            return true;
        }
        return create_index_without_param(index_name_, reader);
    }

    void
    AnalyzeQuery(const DatasetPtr& query_dataset,
                 int64_t topk,
                 const std::string& search_param,
                 const std::string& base_path,
                 const std::string& groundtruth_path,
                 const std::string& save_groundtruth_path) {
        if (not index_) {
            logger::error("Index not loaded");
            return;
        }
        if (not query_dataset) {
            logger::error("Query dataset is null");
            return;
        }
        SearchRequest query_request;
        query_request.query_ = query_dataset;
        query_request.topk_ = topk;
        if (base_path.empty() && groundtruth_path.empty() && save_groundtruth_path.empty()) {
            query_request.params_str_ = search_param;
        } else {
            JsonType params =
                search_param == DEFAULT_SEARCH_PARAM ? JsonType() : JsonType::Parse(search_param);
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
            query_request.params_str_ = params.Dump();
        }
        auto search_result = index_->AnalyzeIndexBySearch(query_request);
        logger::info("Search Analyze: {}", search_result);
    }

    void
    ShowIndexProperty(const std::string& search_param, const std::string& base_path) const {
        if (base_path.empty() || index_name_ != INDEX_SINDI) {
            logger::info("index inner property: {}", index_->GetStats());
            return;
        }
        SearchRequest stats_request;
        stats_request.topk_ = 0;
        JsonType params =
            search_param == DEFAULT_SEARCH_PARAM ? JsonType() : JsonType::Parse(search_param);
        params["analyze"].SetJson(JsonType());
        params["analyze"]["base_path"].SetString(base_path);
        stats_request.params_str_ = params.Dump();
        logger::info("index inner property: {}", index_->AnalyzeIndexBySearch(stats_request));
    }

    DataTypes
    GetDataType() const {
        return data_type_;
    }

private:
    bool
    parse_reader(StreamReader& reader) {
        logger::info("[parse_reader] Start parsing reader");
        auto footer = Footer::Parse(reader);
        if (not footer) {
            logger::error("[parse_reader] Failed to parse footer");
            return false;
        }
        logger::info("[parse_reader] Footer parsed successfully");
        auto meta_data = footer->GetMetadata();
        logger::info("[parse_reader] Got metadata");
        auto basic_info = meta_data->Get(BASIC_INFO);
        logger::info("[parse_reader] Got basic_info");

        if (build_param_ != DEFAULT_BUILD_PARAM) {
            auto parsed = JsonType::Parse(build_param_);
            index_name_ = infer_index_name_from_build_param(parsed);
            if (is_sparse_build_param(parsed)) {
                dim_ = parsed.Contains(DIM) ? parsed[DIM].GetInt() : 0;
                extra_info_size_ = 0;
                data_type_ = DataTypes::DATA_TYPE_SPARSE;
                metric_type_ = MetricType::METRIC_TYPE_IP;
            }

            if (not index_name_.empty()) {
                logger::info("index name (from explicit build_param): {}", index_name_);
                return true;
            }
            logger::error("Failed to infer index type from explicit build_param");
            return false;
        }

        if (not basic_info.Contains(INDEX_PARAM)) {
            logger::error("Index parameter not found in metadata and no build_param provided");
            return false;
        }

        logger::info("[parse_reader] INDEX_PARAM found in basic_info");
        // parse basic info
        std::string inner_param = basic_info[INDEX_PARAM].GetString();
        index_param_ = JsonType::Parse(inner_param);
        index_name_ = infer_index_name_from_build_param(index_param_);
        if (index_name_.empty()) {
            index_name_ = infer_index_name_from_build_param(basic_info);
        }
        if (index_name_.empty()) {
            logger::error("Failed to infer index type from metadata");
            return false;
        }

        dim_ = basic_info.Contains(DIM) ? basic_info[DIM].GetInt() : 0;
        extra_info_size_ =
            basic_info.Contains("extra_info_size") ? basic_info["extra_info_size"].GetInt() : 0;
        if (basic_info.Contains("data_type")) {
            data_type_ = static_cast<DataTypes>(basic_info["data_type"].GetInt());
        } else {
            data_type_ = index_name_ == INDEX_SINDI ? DataTypes::DATA_TYPE_SPARSE
                                                    : DataTypes::DATA_TYPE_FLOAT;
        }
        if (basic_info.Contains("metric")) {
            metric_type_ = static_cast<MetricType>(basic_info["metric"].GetInt());
        } else {
            metric_type_ = index_name_ == INDEX_SINDI ? MetricType::METRIC_TYPE_IP
                                                      : MetricType::METRIC_TYPE_L2SQR;
        }
        if (not basic_info.Contains(DIM) || not basic_info.Contains("extra_info_size") ||
            not basic_info.Contains("data_type") || not basic_info.Contains("metric")) {
            logger::warn("Index metadata is incomplete; missing fields will use analyzer defaults");
        }
        logger::info("index name: {}", index_name_);
        logger::info("index dim: {}", dim_);
        logger::info("index data type: {}", DataTypesToString(data_type_));
        logger::info("index metric: {}", MetricTypeToString(metric_type_));
        logger::info("index param: {}", index_param_.Dump(4));
        return true;
    }

    bool
    create_index_with_param(const std::string& index_name, std::istream& in_stream) {
        auto create_result = Factory::CreateIndex(index_name, build_param_);
        if (not create_result.has_value()) {
            logger::error("Failed to create index with name: {}, due to: {}",
                          index_name,
                          create_result.error().message);
            return false;
        }
        index_ = create_result.value();
        auto deserialize_result = index_->Deserialize(in_stream);
        if (not deserialize_result.has_value()) {
            logger::error("Failed to deserialize index from file, due to: {}",
                          deserialize_result.error().message);
            return false;
        }
        return true;
    }

    bool
    create_index_without_param(const std::string& index_name, StreamReader& reader) {
        // create index common parameters
        IndexCommonParam index_common_params;
        index_common_params.dim_ = dim_;
        index_common_params.metric_ = metric_type_;
        index_common_params.allocator_ = Engine::CreateDefaultAllocator();
        index_common_params.data_type_ = data_type_;
        index_common_params.extra_info_size_ = extra_info_size_;
        // create index and deserialize
        if (index_name_ == INDEX_HGRAPH) {
            auto hgraph_parameter = std::make_shared<HGraphParameter>();
            hgraph_parameter->FromJson(index_param_);
            hgraph_parameter->data_type = data_type_;
            auto inner_index = std::make_shared<HGraph>(hgraph_parameter, index_common_params);
            inner_index->Deserialize(reader);
            index_ = std::make_shared<IndexImpl<HGraph>>(inner_index, index_common_params);
            return true;
        } else if (index_name_ == INDEX_IVF) {
            auto ivf_parameter = std::make_shared<IVFParameter>();
            ivf_parameter->FromJson(index_param_);
            auto inner_index = std::make_shared<IVF>(ivf_parameter, index_common_params);
            inner_index->Deserialize(reader);
            index_ = std::make_shared<IndexImpl<IVF>>(inner_index, index_common_params);
            return true;
        } else if (index_name_ == INDEX_PYRAMID) {
            auto pyramid_parameter = std::make_shared<PyramidParameters>();
            pyramid_parameter->FromJson(index_param_);
            auto inner_index = std::make_shared<Pyramid>(pyramid_parameter, index_common_params);
            inner_index->Deserialize(reader);
            index_ = std::make_shared<IndexImpl<Pyramid>>(inner_index, index_common_params);
            return true;
        } else if (index_name_ == INDEX_SINDI) {
            auto sindi_parameter = std::make_shared<SINDIParameter>();
            sindi_parameter->FromJson(index_param_);
            auto inner_index = std::make_shared<SINDI>(sindi_parameter, index_common_params);
            inner_index->Deserialize(reader);
            index_ = std::make_shared<IndexImpl<SINDI>>(inner_index, index_common_params);
            return true;
        } else {
            logger::error("Index type {} not supported", index_name_);
            return false;
        }
    }

private:
    IndexPtr index_{nullptr};
    std::string build_param_;
    std::string index_path_;
    int64_t dim_{0};
    int64_t extra_info_size_{0};
    DataTypes data_type_{DataTypes::DATA_TYPE_FLOAT};
    MetricType metric_type_{MetricType::METRIC_TYPE_L2SQR};
    std::string index_name_;
    std::string inner_param_;
    JsonType index_param_;
};

int
main(int argc, char** argv) {
    argparse::ArgumentParser parser("analyze_index");
    parse_args(parser, argc, argv);
    std::string index_path = parser.get<std::string>("--index_path");
    std::string build_param = parser.get<std::string>("--build_parameter");
    std::string search_param = parser.get<std::string>("--search_parameter");
    std::string base_path = parser.get<std::string>("--base_path");
    // parse index
    AnalyzedIndex index(build_param);
    if (not index.LoadIndex(index_path)) {
        logger::error("Failed to load index from {}", index_path);
        return -1;
    }
    // get index property
    index.ShowIndexProperty(search_param, base_path);
    // analyze query
    std::string query_path = parser.get<std::string>("--query_path");
    std::string query_data_type = parser.get<std::string>("--query_data_type");
    std::string groundtruth_path = parser.get<std::string>("--groundtruth_path");
    std::string save_groundtruth_path = parser.get<std::string>("--save_groundtruth_path");
    int64_t topk = parser.get<int>("--topk");
    if (query_path != EMPTY_QUERY_PATH) {
        auto querys = load_query(query_path, index.GetDataType(), query_data_type);
        index.AnalyzeQuery(
            querys, topk, search_param, base_path, groundtruth_path, save_groundtruth_path);
    }
}
