
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
#include "index/index_impl.h"
#include "inner_string_params.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"

using namespace vsag;

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

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
    }
}

IndexPtr
get_index(const std::string& index_path) {
    std::fstream in_stream(index_path);
    IOStreamReader reader(in_stream);
    auto footer = Footer::Parse(reader);
    if (not footer) {
        logger::error("Failed to parse footer");
        exit(-1);
    }
    auto meta_data = footer->GetMetadata();
    auto basic_info = meta_data->Get(BASIC_INFO);
    if (not basic_info.contains(INDEX_PARAM)) {
        logger::error("Index parameter not found in metadata");
        exit(-1);
    }
    // parse basic info
    int64_t dim = basic_info[DIM];
    int64_t extra_info_size = basic_info["extra_info_size"];
    DataTypes data_type = static_cast<DataTypes>(basic_info["data_type"].get<int64_t>());
    MetricType metric_type = static_cast<MetricType>(basic_info["metric"].get<int64_t>());
    std::string param_str = basic_info[INDEX_PARAM];
    auto index_param = JsonType::parse(param_str);
    std::string index_name = index_param[INDEX_TYPE];
    logger::info("index name: {}", index_name);
    logger::info("index dim: {}", dim);
    logger::info("index data type: {}", DataTypesToString(data_type));
    logger::info("index metric: {}", MetricTypeToString(metric_type));
    logger::info("index param: {}", index_param.dump(4));

    // create index common parameters
    IndexCommonParam index_common_params;
    index_common_params.dim_ = dim;
    index_common_params.metric_ = metric_type;
    index_common_params.allocator_ = Engine::CreateDefaultAllocator();
    index_common_params.data_type_ = data_type;
    index_common_params.extra_info_size_ = extra_info_size;
    // create index and deserialize
    if (index_name == INDEX_HGRAPH) {
        auto hgraph_parameter = std::make_shared<HGraphParameter>();
        hgraph_parameter->FromJson(index_param);
        hgraph_parameter->data_type = data_type;
        auto inner_index = std::make_shared<HGraph>(hgraph_parameter, index_common_params);
        inner_index->Deserialize(reader);
        return std::make_shared<IndexImpl<HGraph>>(inner_index, index_common_params);
    }
    logger::error("Index type {} not supported", index_name);
    exit(-1);
}

int
main(int argc, char** argv) {
    argparse::ArgumentParser program("analyze_index");
    parse_args(program, argc, argv);
    std::string index_path = program.get<std::string>("--index_path");
    // parse index
    auto index = get_index(index_path);
    // get index property
    logger::info("index inner property: {}", index->GetStats());
}
