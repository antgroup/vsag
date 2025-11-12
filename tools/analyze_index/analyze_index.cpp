
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
#include <fstream>
#include <iostream>

#include "algorithm/hgraph.h"
#include "algorithm/ivf.h"
#include "index/index_impl.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"

using namespace vsag;

constexpr static const char* DEFAULT_BUILD_PARAM = "default";
constexpr static const char* DEFAULT_SEARCH_PARAM = "default";
constexpr static const char* EMPTY_QUERY_PATH = "empty";

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
        .default_value(DEFAULT_SEARCH_PARAM)
        .help("The query dataset path, if not set, will not do query analysis");

    parser.add_argument<std::string>("--search_parameter", "-sp")
        .default_value(EMPTY_QUERY_PATH)
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
load_query(const std::string& query_path) {
    std::fstream in_stream(query_path);
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

class AnalyzedIndex {
public:
    AnalyzedIndex(const std::string& build_param) : build_param_(build_param) {
    }

    bool
    LoadIndex(const std::string& index_path) {
        std::fstream in_stream(index_path);
        IOStreamReader reader(in_stream);
        if (not parse_reader(reader)) {
            return false;
        }
        if (build_param_ != DEFAULT_BUILD_PARAM) {
            if (not create_index_with_param(index_name_, in_stream)) {
                return false;
            }
            return true;
        }
        return create_index_without_param(index_name_, reader);
    }

    void
    AnalyzeQuery(const DatasetPtr& query_dataset, int64_t topk, const std::string& search_param) {
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
        query_request.params_str_ = search_param;
        auto search_result = index_->AnalyzeIndexBySearch(query_request);
        logger::info("Search Analyze: {}", search_result);
    }

    void
    ShowIndexProperty() const {
        logger::info("index inner property: {}", index_->GetStats());
    }

private:
    bool
    parse_reader(StreamReader& reader) {
        auto footer = Footer::Parse(reader);
        if (not footer) {
            logger::error("Failed to parse footer");
            return false;
        }
        auto meta_data = footer->GetMetadata();
        auto basic_info = meta_data->Get(BASIC_INFO);
        if (not basic_info.Contains(INDEX_PARAM)) {
            logger::error("Index parameter not found in metadata");
            return false;
        }
        // parse basic info
        dim_ = basic_info[DIM].GetInt();
        extra_info_size_ = basic_info["extra_info_size"].GetInt();
        data_type_ = static_cast<DataTypes>(basic_info["data_type"].GetInt());
        metric_type_ = static_cast<MetricType>(basic_info["metric"].GetInt());
        std::string inner_param = basic_info[INDEX_PARAM].GetString();
        index_param_ = JsonType::Parse(inner_param);
        index_name_ = index_param_[TYPE_KEY].GetString();
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
        } else {
            logger::error("Index type {} not supported", index_name_);
            return false;
        }
    }

private:
    IndexPtr index_{nullptr};
    std::string build_param_;
    int64_t dim_;
    int64_t extra_info_size_;
    DataTypes data_type_;
    MetricType metric_type_;
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
    // parse index
    AnalyzedIndex index(build_param);
    index.LoadIndex(index_path);
    // get index property
    index.ShowIndexProperty();
    // analyze query
    std::string query_path = parser.get<std::string>("--query_path");
    std::string search_param = parser.get<std::string>("--search_parameter");
    int64_t topk = parser.get<int>("--topk");
    if (query_path != EMPTY_QUERY_PATH) {
        auto querys = load_query(query_path);
        index.AnalyzeQuery(querys, topk, search_param);
    }
}
