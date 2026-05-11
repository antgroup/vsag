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

#include "hgraph_serializer.h"

#include "hgraph.h"
#include "storage/serialization.h"

namespace vsag {

#define TO_JSON_BASE64(json_obj, var) json_obj[#var].SetString(base64_encode_obj(graph.var##_));

JsonType
HGraphSerializer::SerializeBasicInfo(const HGraph& graph) {
    JsonType jsonify_basic_info;
    jsonify_basic_info["use_reorder"].SetBool(graph.use_reorder_);
    jsonify_basic_info["dim"].SetInt(graph.dim_);
    jsonify_basic_info["metric"].SetInt(static_cast<int64_t>(graph.metric_));
    jsonify_basic_info["entry_point_id"].SetInt(graph.entry_point_id_);
    jsonify_basic_info["ef_construct"].SetInt(graph.ef_construct_);
    jsonify_basic_info["extra_info_size"].SetInt(graph.extra_info_size_);
    jsonify_basic_info["data_type"].SetInt(static_cast<int64_t>(graph.data_type_));
    TO_JSON_BASE64(jsonify_basic_info, mult);
    jsonify_basic_info["max_capacity"].SetInt(graph.max_capacity_.load());
    jsonify_basic_info["max_level"].SetInt(graph.route_graphs_.size());
    jsonify_basic_info[INDEX_PARAM].SetString(graph.create_param_ptr_->ToString());

    return jsonify_basic_info;
}

#undef TO_JSON_BASE64

#define FROM_JSON(json_obj, var, type)                   \
    do {                                                 \
        if ((json_obj).Contains(#var)) {                 \
            graph.var##_ = (json_obj)[#var].Get##type(); \
        }                                                \
    } while (0)

#define FROM_JSON_BASE64(json_obj, var) \
    base64_decode_obj((json_obj)[#var].GetString(), graph.var##_);

void
HGraphSerializer::DeserializeBasicInfo(HGraph& graph, const JsonType& jsonify_basic_info) {
    logger::debug("jsonify_basic_info: {}", jsonify_basic_info.Dump());
    FROM_JSON(jsonify_basic_info, use_reorder, Bool);
    FROM_JSON(jsonify_basic_info, dim, Int);
    if (jsonify_basic_info.Contains("metric")) {
        graph.metric_ = static_cast<MetricType>(jsonify_basic_info["metric"].GetInt());
    }
    FROM_JSON(jsonify_basic_info, entry_point_id, Int);
    FROM_JSON(jsonify_basic_info, ef_construct, Int);
    FROM_JSON(jsonify_basic_info, extra_info_size, Int);
    if (jsonify_basic_info.Contains("data_type")) {
        graph.data_type_ = static_cast<DataTypes>(jsonify_basic_info["data_type"].GetInt());
    }
    FROM_JSON_BASE64(jsonify_basic_info, mult);
    graph.max_capacity_.store(jsonify_basic_info["max_capacity"].GetInt());

    auto max_level = jsonify_basic_info["max_level"].GetInt();
    for (int64_t i = 0; i < max_level; ++i) {
        graph.route_graphs_.emplace_back(graph.generate_one_route_graph());
    }
    if (jsonify_basic_info.Contains(INDEX_PARAM)) {
        std::string index_param_string = jsonify_basic_info[INDEX_PARAM].GetString();
        HGraphParameterPtr index_param = std::make_shared<HGraphParameter>();
        index_param->data_type = graph.data_type_;
        index_param->FromString(index_param_string);
        if (not graph.create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("HGraph index parameter not match, current: {}, new: {}",
                                       graph.create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }
}

#undef FROM_JSON
#undef FROM_JSON_BASE64

void
HGraphSerializer::SerializeLabelInfo(const HGraph& graph, StreamWriter& writer) {
    if (graph.support_duplicate_) {
        graph.label_table_->Serialize(writer);
        return;
    }

    StreamWriter::WriteVector(writer, graph.label_table_->label_table_);
    uint64_t size = graph.label_table_->GetRemapSize();
    StreamWriter::WriteObj(writer, size);
    graph.label_table_->ForEachRemap([&writer](LabelType key, InnerIdType value) {
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, value);
    });
}

void
HGraphSerializer::DeserializeLabelInfo(const HGraph& graph, StreamReader& reader) {
    if (graph.support_duplicate_) {
        graph.label_table_->Deserialize(reader);
        return;
    }

    StreamReader::ReadVector(reader, graph.label_table_->label_table_);
    uint64_t size;
    StreamReader::ReadObj(reader, size);
    graph.label_table_->ResetRemap(size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        graph.label_table_->InsertRemap(key, value);
    }
    graph.label_table_->total_count_.store(static_cast<int64_t>(size));
}

void
HGraphSerializer::SerializeBasicInfoV0_14(const HGraph& graph, StreamWriter& writer) {
    StreamWriter::WriteObj(writer, graph.use_reorder_);
    StreamWriter::WriteObj(writer, graph.dim_);
    StreamWriter::WriteObj(writer, graph.metric_);
    uint64_t max_level = graph.route_graphs_.size();
    StreamWriter::WriteObj(writer, max_level);
    StreamWriter::WriteObj(writer, graph.entry_point_id_);
    StreamWriter::WriteObj(writer, graph.ef_construct_);
    StreamWriter::WriteObj(writer, graph.mult_);
    auto capacity = graph.max_capacity_.load();
    StreamWriter::WriteObj(writer, capacity);
    StreamWriter::WriteVector(writer, graph.label_table_->label_table_);

    uint64_t size = graph.label_table_->GetRemapSize();
    StreamWriter::WriteObj(writer, size);
    graph.label_table_->ForEachRemap([&writer](LabelType key, InnerIdType value) {
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, value);
    });
}

void
HGraphSerializer::DeserializeBasicInfoV0_14(HGraph& graph, StreamReader& reader) {
    StreamReader::ReadObj(reader, graph.use_reorder_);
    StreamReader::ReadObj(reader, graph.dim_);
    StreamReader::ReadObj(reader, graph.metric_);
    uint64_t max_level;
    StreamReader::ReadObj(reader, max_level);
    for (uint64_t i = 0; i < max_level; ++i) {
        graph.route_graphs_.emplace_back(graph.generate_one_route_graph());
    }
    StreamReader::ReadObj(reader, graph.entry_point_id_);
    StreamReader::ReadObj(reader, graph.ef_construct_);
    StreamReader::ReadObj(reader, graph.mult_);
    InnerIdType capacity;
    StreamReader::ReadObj(reader, capacity);
    graph.max_capacity_.store(capacity);
    StreamReader::ReadVector(reader, graph.label_table_->label_table_);

    uint64_t size;
    StreamReader::ReadObj(reader, size);
    graph.label_table_->ResetRemap(size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        graph.label_table_->InsertRemap(key, value);
    }
    graph.label_table_->total_count_.store(static_cast<int64_t>(size));
}

void
HGraphSerializer::Serialize(const HGraph& graph, StreamWriter& writer) {
    if (graph.ignore_reorder_) {
        graph.use_reorder_ = false;
    }

    if (graph.use_old_serial_format_) {
        HGraphSerializer::SerializeBasicInfoV0_14(graph, writer);
        graph.basic_flatten_codes_->Serialize(writer);
        graph.bottom_graph_->Serialize(writer);
        if (graph.use_reorder_) {
            graph.high_precise_codes_->Serialize(writer);
        }
        for (const auto& route_graph : graph.route_graphs_) {
            route_graph->Serialize(writer);
        }
        if (graph.extra_info_size_ > 0 && graph.extra_infos_ != nullptr) {
            graph.extra_infos_->Serialize(writer);
        }
        if (graph.use_attribute_filter_ and graph.attr_filter_index_ != nullptr) {
            graph.attr_filter_index_->Serialize(writer);
        }
        return;
    }

    HGraphSerializer::SerializeLabelInfo(graph, writer);
    graph.basic_flatten_codes_->Serialize(writer);
    graph.bottom_graph_->Serialize(writer);
    if (graph.use_reorder_) {
        graph.high_precise_codes_->Serialize(writer);
    }
    for (const auto& route_graph : graph.route_graphs_) {
        route_graph->Serialize(writer);
    }
    if (graph.extra_info_size_ > 0 && graph.extra_infos_ != nullptr) {
        graph.extra_infos_->Serialize(writer);
    }
    if (graph.use_attribute_filter_ and graph.attr_filter_index_ != nullptr) {
        graph.attr_filter_index_->Serialize(writer);
    }
    if (graph.create_new_raw_vector_) {
        graph.raw_vector_->Serialize(writer);
    }

    auto jsonify_basic_info = HGraphSerializer::SerializeBasicInfo(graph);
    auto metadata = std::make_shared<Metadata>();
    metadata->Set(BASIC_INFO, jsonify_basic_info);
    if (graph.support_duplicate_) {
        metadata->Set("duplicate_format_version", 1);
    }
    logger::debug(jsonify_basic_info.Dump());

    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

void
HGraphSerializer::Deserialize(HGraph& graph, StreamReader& reader) {
    auto footer = Footer::Parse(reader);

    if (footer == nullptr) {
        logger::debug("parse with v0.14 version format");

        HGraphSerializer::DeserializeBasicInfoV0_14(graph, reader);

        graph.basic_flatten_codes_->Deserialize(reader);
        graph.bottom_graph_->Deserialize(reader);
        if (graph.use_reorder_) {
            graph.high_precise_codes_->Deserialize(reader);
        }

        for (auto& route_graph : graph.route_graphs_) {
            route_graph->Deserialize(reader);
        }
        auto new_size = graph.max_capacity_.load();
        graph.neighbors_mutex_->Resize(new_size);

        graph.pool_ =
            std::make_shared<VisitedListPool>(1, graph.allocator_, new_size, graph.allocator_);

        if (graph.extra_info_size_ > 0 && graph.extra_infos_ != nullptr) {
            graph.extra_infos_->Deserialize(reader);
        }
        graph.total_count_ = graph.basic_flatten_codes_->TotalCount();

        if (graph.use_attribute_filter_ and graph.attr_filter_index_ != nullptr) {
            graph.attr_filter_index_->Deserialize(reader);
        }
    } else {
        logger::debug("parse with new version format");

        BufferStreamReader buffer_reader(
            &reader, std::numeric_limits<uint64_t>::max(), graph.allocator_);

        auto metadata = footer->GetMetadata();
        HGraphSerializer::DeserializeBasicInfo(graph, metadata->Get(BASIC_INFO));

        int64_t dup_version = 0;
        if (metadata->Get("duplicate_format_version").IsNumberInteger()) {
            dup_version = metadata->Get("duplicate_format_version").GetInt();
        }
        graph.label_table_->is_legacy_duplicate_format_ = (dup_version == 0);

        HGraphSerializer::DeserializeLabelInfo(graph, buffer_reader);

        graph.basic_flatten_codes_->Deserialize(buffer_reader);
        graph.bottom_graph_->Deserialize(buffer_reader);
        if (graph.use_reorder_) {
            graph.high_precise_codes_->Deserialize(buffer_reader);
        }

        for (auto& route_graph : graph.route_graphs_) {
            route_graph->Deserialize(buffer_reader);
        }
        auto new_size = graph.max_capacity_.load();
        graph.neighbors_mutex_->Resize(new_size);

        graph.pool_ =
            std::make_shared<VisitedListPool>(1, graph.allocator_, new_size, graph.allocator_);

        if (graph.extra_info_size_ > 0 && graph.extra_infos_ != nullptr) {
            graph.extra_infos_->Deserialize(buffer_reader);
        }
        graph.total_count_ = graph.basic_flatten_codes_->TotalCount();

        if (graph.use_attribute_filter_ and graph.attr_filter_index_ != nullptr) {
            graph.attr_filter_index_->Deserialize(buffer_reader);
        }

        if (graph.create_new_raw_vector_) {
            graph.raw_vector_->Deserialize(buffer_reader);
        }
        if (graph.raw_vector_ != nullptr) {
            graph.has_raw_vector_ = true;
        }
    }
    graph.cal_memory_usage();

    if (graph.use_elp_optimizer_) {
        graph.elp_optimize();
    }
}

std::string
HGraphSerializer::GetMemoryUsageDetail(const HGraph& graph) {
    JsonType memory_usage;
    if (graph.ignore_reorder_) {
        graph.use_reorder_ = false;
    }
    memory_usage["basic_flatten_codes"].SetInt(graph.basic_flatten_codes_->CalcSerializeSize());
    memory_usage["bottom_graph"].SetInt(graph.bottom_graph_->CalcSerializeSize());
    if (graph.use_reorder_) {
        memory_usage["high_precise_codes"].SetInt(graph.high_precise_codes_->CalcSerializeSize());
    }
    uint64_t route_graph_size = 0;
    for (const auto& route_graph : graph.route_graphs_) {
        route_graph_size += route_graph->CalcSerializeSize();
    }
    memory_usage["route_graph"].SetInt(route_graph_size);
    if (graph.extra_info_size_ > 0 && graph.extra_infos_ != nullptr) {
        memory_usage["extra_infos"].SetInt(graph.extra_infos_->CalcSerializeSize());
    }
    memory_usage["__total_size__"].SetInt(graph.CalSerializeSize());
    return memory_usage.Dump();
}

}  // namespace vsag
