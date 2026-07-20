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

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <limits>

#include "datacell/sparse_graph_datacell.h"
#include "hgraph.h"  // IWYU pragma: keep
#include "impl/heap/standard_heap.h"
#include "impl/odescent/odescent_graph_builder.h"
#include "impl/pruning_strategy.h"
#include "storage/serialization.h"
#include "storage/serialization_tags.h"
#include "storage/stream_reader.h"
#include "storage/tlv_section.h"
#include "typing.h"
#include "utils/util_functions.h"
#include "vsag/options.h"

namespace vsag {

namespace {

std::string
dump_basic_info_for_log(const JsonType& basic_info) {
    JsonType log_basic_info = basic_info;  // NOLINT(performance-unnecessary-copy-initialization)
    if (log_basic_info.Contains(INDEX_PARAM) and log_basic_info[INDEX_PARAM].IsString()) {
        auto index_param = JsonType::Parse(log_basic_info[INDEX_PARAM].GetString(), false);
        if (not index_param.IsDiscarded() and index_param.IsObject()) {
            log_basic_info[INDEX_PARAM].SetJson(index_param);
        }
    }
    return log_basic_info.Dump(4);
}

class PayloadChecksumStreamReader : public StreamReader {
public:
    PayloadChecksumStreamReader(StreamReader& reader, const StreamBlockHeader& header)
        : StreamReader(header.value_len),
          reader_(reader),
          expected_checksum_(header.payload_checksum) {
    }

    void
    Read(char* data, uint64_t size) override {
        const auto cursor = reader_.GetCursor();
        if (cursor > length_ || size > length_ - cursor) {
            throw VsagException(ErrorType::READ_ERROR,
                                "checksum stream reader exceeds payload boundary");
        }
        if (cursor > checksum_cursor_) {
            checksum_range(checksum_cursor_, cursor - checksum_cursor_);
        }

        reader_.Read(data, size);
        if (cursor + size > checksum_cursor_) {
            const auto checksum_offset = checksum_cursor_ > cursor ? checksum_cursor_ - cursor : 0;
            const auto checksum_size = size - checksum_offset;
            crc_ = StreamHeader::UpdateChecksum(
                crc_, std::string_view(data + checksum_offset, checksum_size));
            checksum_cursor_ += checksum_size;
        }
    }

    void
    Seek(uint64_t cursor) override {
        if (cursor > length_) {
            throw VsagException(ErrorType::READ_ERROR,
                                "checksum stream reader seek exceeds payload boundary");
        }
        reader_.Seek(cursor);
    }

    [[nodiscard]] uint64_t
    GetCursor() const override {
        return reader_.GetCursor();
    }

    void
    Validate() {
        if (checksum_cursor_ < length_) {
            checksum_range(checksum_cursor_, length_ - checksum_cursor_);
        }
        if (StreamHeader::FinalizeChecksum(crc_) != expected_checksum_) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                "streaming block payload checksum mismatch");
        }
    }

private:
    void
    checksum_range(uint64_t offset, uint64_t size) {
        constexpr uint64_t k_buffer_size = 8192;
        std::array<char, k_buffer_size> buffer{};
        const auto original_cursor = reader_.GetCursor();
        reader_.Seek(offset);
        uint64_t remaining = size;
        while (remaining > 0) {
            const auto read_size = std::min<uint64_t>(remaining, k_buffer_size);
            reader_.Read(buffer.data(), read_size);
            crc_ = StreamHeader::UpdateChecksum(crc_, std::string_view(buffer.data(), read_size));
            remaining -= read_size;
        }
        checksum_cursor_ += size;
        reader_.Seek(original_cursor);
    }

private:
    StreamReader& reader_;
    uint32_t expected_checksum_{0};
    uint32_t crc_{StreamHeader::InitialChecksum()};
    uint64_t checksum_cursor_{0};
};

}  // namespace

void
HGraph::serialize_basic_info_v0_14(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, this->use_reorder_);
    StreamWriter::WriteObj(writer, this->dim_);
    StreamWriter::WriteObj(writer, this->metric_);
    uint64_t max_level = this->route_graphs_.size();
    StreamWriter::WriteObj(writer, max_level);
    StreamWriter::WriteObj(writer, this->entry_point_id_);
    StreamWriter::WriteObj(writer, this->ef_construct_);
    StreamWriter::WriteObj(writer, this->mult_);
    auto capacity = this->max_capacity_.load();
    StreamWriter::WriteObj(writer, capacity);
    StreamWriter::WriteVector(writer, this->label_table_->label_table_);

    uint64_t size = this->label_table_->GetRemapSize();
    StreamWriter::WriteObj(writer, size);
    this->label_table_->ForEachRemap([&writer](LabelType key, InnerIdType value) {
        StreamWriter::WriteObj(writer, key);
        StreamWriter::WriteObj(writer, value);
    });
}

void
HGraph::deserialize_basic_info_v0_14(StreamReader& reader) {
    StreamReader::ReadObj(reader, this->use_reorder_);
    StreamReader::ReadObj(reader, this->dim_);
    StreamReader::ReadObj(reader, this->metric_);
    uint64_t max_level;
    StreamReader::ReadObj(reader, max_level);
    for (uint64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
    StreamReader::ReadObj(reader, this->entry_point_id_);
    StreamReader::ReadObj(reader, this->ef_construct_);
    StreamReader::ReadObj(reader, this->mult_);
    InnerIdType capacity;
    StreamReader::ReadObj(reader, capacity);
    this->max_capacity_.store(capacity);
    StreamReader::ReadVector(reader, this->label_table_->label_table_);

    uint64_t size;
    StreamReader::ReadObj(reader, size);
    this->label_table_->ResetRemap(size);
    for (uint64_t i = 0; i < size; ++i) {
        LabelType key;
        StreamReader::ReadObj(reader, key);
        InnerIdType value;
        StreamReader::ReadObj(reader, value);
        this->label_table_->InsertRemap(key, value);
    }
    // Restore total_count from label_remap size
    this->label_table_->total_count_.store(static_cast<int64_t>(size));
}

#define TO_JSON_BASE64(json_obj, var) json_obj[#var].SetString(base64_encode_obj(this->var##_));

JsonType
HGraph::serialize_basic_info() const {
    JsonType jsonify_basic_info;
    jsonify_basic_info["use_reorder"].SetBool(this->use_reorder_);
    jsonify_basic_info["reorder_by_base"].SetBool(this->reorder_by_base_);
    jsonify_basic_info["dim"].SetInt(this->dim_);
    jsonify_basic_info["metric"].SetInt(static_cast<int64_t>(this->metric_));
    jsonify_basic_info["entry_point_id"].SetInt(this->entry_point_id_);
    jsonify_basic_info["ef_construct"].SetUint64(this->ef_construct_);
    jsonify_basic_info["extra_info_size"].SetUint64(this->extra_info_size_);
    jsonify_basic_info["data_type"].SetInt(static_cast<int64_t>(this->data_type_));
    jsonify_basic_info["persist_source_id"].SetBool(this->persist_source_id_);
    // logger::debug("mult: {}", this->mult_);
    TO_JSON_BASE64(jsonify_basic_info, mult);
    jsonify_basic_info["max_capacity"].SetUint64(this->max_capacity_.load());
    jsonify_basic_info["max_level"].SetUint64(this->route_graphs_.size());
    jsonify_basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());

    return jsonify_basic_info;
}

#define FROM_JSON(json_obj, var, type)                   \
    do {                                                 \
        if ((json_obj).Contains(#var)) {                 \
            this->var##_ = (json_obj)[#var].Get##type(); \
        }                                                \
    } while (0)

#define FROM_JSON_BASE64(json_obj, var) \
    base64_decode_obj((json_obj)[#var].GetString(), this->var##_);

void
HGraph::deserialize_basic_info(const JsonType& jsonify_basic_info) {
    logger::debug("jsonify_basic_info:\n{}", dump_basic_info_for_log(jsonify_basic_info));
    FROM_JSON(jsonify_basic_info, use_reorder, Bool);
    this->reorder_by_base_ = false;
    FROM_JSON(jsonify_basic_info, reorder_by_base, Bool);
    FROM_JSON(jsonify_basic_info, dim, Int);
    if (jsonify_basic_info.Contains("metric")) {
        this->metric_ = static_cast<MetricType>(jsonify_basic_info["metric"].GetInt());
    }
    FROM_JSON(jsonify_basic_info, entry_point_id, Int);
    FROM_JSON(jsonify_basic_info, ef_construct, Uint64);
    FROM_JSON(jsonify_basic_info, extra_info_size, Uint64);
    if (jsonify_basic_info.Contains("data_type")) {
        this->data_type_ = static_cast<DataTypes>(jsonify_basic_info["data_type"].GetInt());
    }
    if (jsonify_basic_info.Contains("persist_source_id")) {
        this->persist_source_id_ = jsonify_basic_info["persist_source_id"].GetBool();
    }
    FROM_JSON_BASE64(jsonify_basic_info, mult);
    // logger::debug("mult: {}", this->mult_);
    auto max_capacity = jsonify_basic_info["max_capacity"].GetUint64();
    if (max_capacity > std::numeric_limits<InnerIdType>::max()) {
        throw VsagException(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("HGraph max_capacity {} exceeds InnerIdType limit", max_capacity));
    }
    this->max_capacity_.store(static_cast<InnerIdType>(max_capacity));

    auto max_level = jsonify_basic_info["max_level"].GetUint64();
    for (uint64_t i = 0; i < max_level; ++i) {
        this->route_graphs_.emplace_back(this->generate_one_route_graph());
    }
    if (jsonify_basic_info.Contains(INDEX_PARAM)) {
        std::string index_param_string = jsonify_basic_info[INDEX_PARAM].GetString();
        HGraphParameterPtr index_param = std::make_shared<HGraphParameter>();
        index_param->data_type = this->data_type_;
        index_param->FromString(index_param_string);
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("HGraph index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }
}

// Magic header used to mark presence of an appended source_id_table block,
// enabling backward-compatible Deserialize against legacy index files written
// before source_id_table_ persistence was implemented.
static constexpr uint64_t SOURCE_ID_TABLE_MAGIC = 0x534F555243454944ULL;  // "SOURCEID"

void
HGraph::serialize_label_info(StreamWriter& writer) const {
    if (this->support_duplicate_) {
        this->label_table_->Serialize(writer);
    } else {
        StreamWriter::WriteVector(writer, this->label_table_->label_table_);
        uint64_t size = this->label_table_->GetRemapSize();
        StreamWriter::WriteObj(writer, size);
        this->label_table_->ForEachRemap([&writer](LabelType key, InnerIdType value) {
            StreamWriter::WriteObj(writer, key);
            StreamWriter::WriteObj(writer, value);
        });
    }

    // Append source_id_table_ block: [magic][count][str0][str1]...
    // Only persist when persist_source_id_ is enabled. Even an empty
    // source_id_table_ still emits the magic + count==0 so the reader can
    // unambiguously detect (and skip) the block.
    if (this->persist_source_id_) {
        const auto& sid_table = this->label_table_->GetSourceIdTableRef();
        StreamWriter::WriteObj(writer, SOURCE_ID_TABLE_MAGIC);
        uint64_t sid_count = sid_table.size();
        StreamWriter::WriteObj(writer, sid_count);
        for (uint64_t i = 0; i < sid_count; ++i) {
            StreamWriter::WriteString(writer, sid_table[i]);
        }
    }
}

void
HGraph::deserialize_label_info(StreamReader& reader) const {
    if (this->support_duplicate_) {
        this->label_table_->Deserialize(reader);
    } else {
        StreamReader::ReadVector(reader, this->label_table_->label_table_);
        uint64_t size;
        StreamReader::ReadObj(reader, size);
        this->label_table_->ResetRemap(size);
        for (uint64_t i = 0; i < size; ++i) {
            LabelType key;
            StreamReader::ReadObj(reader, key);
            InnerIdType value;
            StreamReader::ReadObj(reader, value);
            this->label_table_->InsertRemap(key, value);
        }
        this->label_table_->total_count_.store(static_cast<int64_t>(size));
    }

    // Optional source_id_table_ block. If the next 8 bytes don't match
    // SOURCE_ID_TABLE_MAGIC, the stream is from a legacy writer; rewind so
    // the parent reader can continue with the next field.
    const uint64_t cursor_before = reader.GetCursor();
    if (reader.Length() >= cursor_before + sizeof(uint64_t)) {
        uint64_t magic = 0;
        StreamReader::ReadObj(reader, magic);
        if (magic == SOURCE_ID_TABLE_MAGIC) {
            uint64_t sid_count = 0;
            StreamReader::ReadObj(reader, sid_count);
            // Defensive validation against corrupted / maliciously crafted
            // streams: an unchecked resize on a huge sid_count would cause
            // OOM / DoS. sid_count must not exceed the just-deserialized label
            // table size. It may be smaller (e.g. partial source_id assignment).
            const uint64_t label_table_size = this->label_table_->label_table_.size();
            if (sid_count > label_table_size) {
                throw VsagException(ErrorType::INVALID_ARGUMENT,
                                    fmt::format("corrupted index: source_id_table sid_count ({}) "
                                                "exceeds label_table size ({})",
                                                sid_count,
                                                label_table_size));
            }
            Vector<std::string> sid_table(sid_count, std::string{}, allocator_);
            for (uint64_t i = 0; i < sid_count; ++i) {
                sid_table[i] = StreamReader::ReadString(reader);
            }
            this->label_table_->ReplaceSourceIdTable(std::move(sid_table));
        } else {
            reader.Seek(cursor_before);
        }
    }
}

void
HGraph::Serialize(StreamWriter& writer) const {
    if (this->ignore_reorder_) {
        this->use_reorder_ = false;
    }

    // FIXME(wxyu): this option is used for special purposes, like compatibility testing
    if (this->use_old_serial_format_) {
        this->serialize_basic_info_v0_14(writer);
        this->basic_flatten_codes_->Serialize(writer);
        this->bottom_graph_->Serialize(writer);
        if (this->has_precise_reorder()) {
            this->high_precise_codes_->Serialize(writer);
        }
        for (const auto& route_graph : this->route_graphs_) {
            route_graph->Serialize(writer);
        }
        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Serialize(writer);
        }
        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Serialize(writer);
        }
        return;
    }

    this->serialize_label_info(writer);
    this->basic_flatten_codes_->Serialize(writer);
    this->bottom_graph_->Serialize(writer);
    if (this->has_precise_reorder()) {
        this->high_precise_codes_->Serialize(writer);
    }
    for (const auto& route_graph : this->route_graphs_) {
        route_graph->Serialize(writer);
    }
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        this->extra_infos_->Serialize(writer);
    }
    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Serialize(writer);
    }
    if (create_new_raw_vector_) {
        this->raw_vector_->Serialize(writer);
    }

    // serialize footer (introduced since v0.15)
    auto jsonify_basic_info = this->serialize_basic_info();
    auto metadata = std::make_shared<Metadata>();
    metadata->Set(BASIC_INFO, jsonify_basic_info);
    if (this->support_duplicate_) {
        metadata->Set("duplicate_format_version", 1);
    }
    logger::debug(jsonify_basic_info.Dump());

    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

MetadataPtr
HGraph::collect_streaming_header() const {
    auto metadata = std::make_shared<Metadata>();
    metadata->Set("format", "vsag_stream_v1");
    metadata->Set("index_name", this->GetName());

    auto jsonify_basic_info = this->serialize_basic_info();
    const bool include_precise_codes = this->has_precise_reorder() && !this->ignore_reorder_;
    if (this->ignore_reorder_) {
        jsonify_basic_info["use_reorder"].SetBool(false);
    }
    metadata->Set(BASIC_INFO, jsonify_basic_info);
    if (this->support_duplicate_) {
        metadata->Set("duplicate_format_version", 1);
    }

    JsonType manifest;
    auto append_manifest = [&manifest](uint32_t tag, bool critical) {
        AppendStreamingManifestBlock(
            manifest, tag, StreamSerializationBlockCurrentVersion(tag), critical);
    };
    auto label_table_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);
    auto base_codes_tag = static_cast<uint32_t>(StreamSerializationTag::BASE_CODES);
    auto bottom_graph_tag = static_cast<uint32_t>(StreamSerializationTag::BOTTOM_GRAPH);
    auto route_graphs_tag = static_cast<uint32_t>(StreamSerializationTag::ROUTE_GRAPHS);
    append_manifest(label_table_tag, StreamSerializationTagCritical(label_table_tag));
    append_manifest(base_codes_tag, StreamSerializationTagCritical(base_codes_tag));
    append_manifest(bottom_graph_tag, StreamSerializationTagCritical(bottom_graph_tag));
    if (include_precise_codes) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::HIGH_PRECISION_CODES);
        append_manifest(tag, StreamSerializationTagCritical(tag));
    }
    append_manifest(route_graphs_tag, StreamSerializationTagCritical(route_graphs_tag));
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::EXTRA_INFO);
        append_manifest(tag, StreamSerializationTagCritical(tag));
    }
    if (this->use_attribute_filter_ && this->attr_filter_index_ != nullptr) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::ATTRIBUTE_FILTER);
        append_manifest(tag, StreamSerializationTagCritical(tag));
    }
    if (create_new_raw_vector_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::RAW_VECTOR);
        append_manifest(tag, StreamSerializationTagCritical(tag));
    }
    metadata->Set("block_manifest", manifest);
    metadata->SetEmptyIndex(this->GetNumElements() == 0);
    return metadata;
}

void
HGraph::serialize_streaming_body(StreamWriter& writer) const {
    const bool include_precise_codes = this->has_precise_reorder() && !this->ignore_reorder_;

    auto label_table_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);
    auto base_codes_tag = static_cast<uint32_t>(StreamSerializationTag::BASE_CODES);
    auto bottom_graph_tag = static_cast<uint32_t>(StreamSerializationTag::BOTTOM_GRAPH);
    auto route_graphs_tag = static_cast<uint32_t>(StreamSerializationTag::ROUTE_GRAPHS);

    WriteStreamingBlock(
        writer,
        label_table_tag,
        StreamSerializationTagCritical(label_table_tag),
        [this](StreamWriter& block_writer) { this->serialize_label_info(block_writer); });
    WriteStreamingBlock(writer,
                        base_codes_tag,
                        StreamSerializationTagCritical(base_codes_tag),
                        [this](StreamWriter& block_writer) {
                            this->basic_flatten_codes_->Serialize(block_writer);
                        });
    WriteStreamingBlock(
        writer,
        bottom_graph_tag,
        StreamSerializationTagCritical(bottom_graph_tag),
        [this](StreamWriter& block_writer) { this->bottom_graph_->Serialize(block_writer); });
    if (include_precise_codes) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::HIGH_PRECISION_CODES);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& block_writer) {
                this->high_precise_codes_->Serialize(block_writer);
            });
    }
    WriteStreamingBlock(writer,
                        route_graphs_tag,
                        StreamSerializationTagCritical(route_graphs_tag),
                        [this](StreamWriter& block_writer) {
                            for (const auto& route_graph : this->route_graphs_) {
                                route_graph->Serialize(block_writer);
                            }
                        });
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::EXTRA_INFO);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& block_writer) {
                this->extra_infos_->Serialize(block_writer);
            });
    }
    if (this->use_attribute_filter_ && this->attr_filter_index_ != nullptr) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::ATTRIBUTE_FILTER);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& block_writer) {
                this->attr_filter_index_->Serialize(block_writer);
            });
    }
    if (create_new_raw_vector_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::RAW_VECTOR);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& block_writer) {
                this->raw_vector_->Serialize(block_writer);
            });
    }
}

void
HGraph::deserialize_label_info_streaming(StreamReader& reader) const {
    if (this->support_duplicate_) {
        this->label_table_->Deserialize(reader);
    } else {
        StreamReader::ReadVector(reader, this->label_table_->label_table_);
        uint64_t size;
        StreamReader::ReadObj(reader, size);
        this->label_table_->ResetRemap(size);
        for (uint64_t i = 0; i < size; ++i) {
            LabelType key;
            StreamReader::ReadObj(reader, key);
            InnerIdType value;
            StreamReader::ReadObj(reader, value);
            this->label_table_->InsertRemap(key, value);
        }
        this->label_table_->total_count_.store(static_cast<int64_t>(size));
    }

    if (this->persist_source_id_) {
        uint64_t magic = 0;
        StreamReader::ReadObj(reader, magic);
        if (magic != SOURCE_ID_TABLE_MAGIC) {
            throw VsagException(ErrorType::READ_ERROR, "missing HGraph source_id_table marker");
        }
        uint64_t sid_count = 0;
        StreamReader::ReadObj(reader, sid_count);
        const uint64_t label_table_size = this->label_table_->label_table_.size();
        if (sid_count > label_table_size) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                fmt::format("corrupted index: source_id_table sid_count ({}) "
                                            "exceeds label_table size ({})",
                                            sid_count,
                                            label_table_size));
        }
        Vector<std::string> sid_table(sid_count, std::string{}, allocator_);
        for (uint64_t i = 0; i < sid_count; ++i) {
            sid_table[i] = StreamReader::ReadString(reader);
        }
        this->label_table_->ReplaceSourceIdTable(std::move(sid_table));
    }
}

void
HGraph::deserialize_streaming_body(StreamReader& reader, const MetadataPtr& metadata) {
    this->read_streaming_body(reader, metadata);
}

void
HGraph::load_streaming_body(StreamReader& reader,
                            const MetadataPtr& metadata,
                            const LoadParameters& parameters) {
    this->read_streaming_body(reader, metadata, &parameters);
}

void
HGraph::read_streaming_body(StreamReader& reader,
                            const MetadataPtr& metadata,
                            const LoadParameters* load_parameters) {
    auto basic_info = metadata->Get(BASIC_INFO);
    this->deserialize_basic_info(basic_info);

    int64_t dup_version = 0;
    if (metadata->Get("duplicate_format_version").IsNumberInteger()) {
        dup_version = metadata->Get("duplicate_format_version").GetInt();
    }
    this->label_table_->is_legacy_duplicate_format_ = (dup_version == 0);

    bool loaded_label_table = false;
    bool loaded_base_codes = false;
    bool loaded_bottom_graph = false;
    bool loaded_high_precision_codes = false;
    bool loaded_route_graphs = false;
    bool loaded_extra_info = false;
    bool loaded_attribute_filter = false;
    bool loaded_raw_vector = false;

    while (true) {
        auto block_header = StreamBlockHeader::Read(reader);
        if (block_header.IsSectionEnd()) {
            break;
        }
        BoundedForwardReader block_reader(&reader, block_header.value_len);
        if (!StreamSerializationBlockVersionSupported(block_header.tag,
                                                      block_header.block_version)) {
            if (block_header.IsCritical()) {
                throw VsagException(
                    ErrorType::UNSUPPORTED_INDEX_OPERATION,
                    fmt::format("unsupported HGraph streaming block version: tag={}, name={}, "
                                "version={}, flags={}, value_len={}",
                                block_header.tag,
                                StreamSerializationTagName(block_header.tag),
                                block_header.block_version,
                                block_header.flags,
                                block_header.value_len));
            }
            block_reader.SkipRemaining();
            continue;
        }

        switch (static_cast<StreamSerializationTag>(block_header.tag)) {
            case StreamSerializationTag::LABEL_TABLE:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->deserialize_label_info_streaming(block);
                });
                loaded_label_table = true;
                break;
            case StreamSerializationTag::BASE_CODES:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->basic_flatten_codes_->Deserialize(block);
                });
                loaded_base_codes = true;
                break;
            case StreamSerializationTag::BOTTOM_GRAPH:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->bottom_graph_->Deserialize(block);
                });
                loaded_bottom_graph = true;
                break;
            case StreamSerializationTag::HIGH_PRECISION_CODES:
                if (this->has_precise_reorder()) {
                    constexpr const char* k_precise_reader = "precise_reader";
                    if (load_parameters != nullptr &&
                        load_parameters->HasReader(k_precise_reader)) {
                        auto reader_ptr = load_parameters->GetReader(k_precise_reader);
                        if (reader_ptr == nullptr) {
                            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                                "precise_reader is null");
                        }
                        if (reader_ptr->Size() != block_header.value_len) {
                            throw VsagException(
                                ErrorType::INVALID_ARGUMENT,
                                fmt::format("precise_reader size {} does not match streaming block "
                                            "payload size {}",
                                            reader_ptr->Size(),
                                            block_header.value_len));
                        }
                        auto read_func = [reader_ptr](uint64_t offset, uint64_t len, void* dest) {
                            reader_ptr->Read(offset, len, dest);
                        };
                        block_reader.SkipRemaining();
                        this->SetPreciseCodesIO(reader_ptr);
                        ReadFuncStreamReader external_reader(read_func, 0, reader_ptr->Size());
                        PayloadChecksumStreamReader checksum_reader(external_reader, block_header);
                        this->high_precise_codes_->Deserialize(checksum_reader);
                        checksum_reader.Validate();
                    } else {
                        ReadSeekableBlockPayload(
                            block_reader, block_header, [this](StreamReader& block) {
                                this->high_precise_codes_->Deserialize(block);
                            });
                    }
                    loaded_high_precision_codes = true;
                }
                break;
            case StreamSerializationTag::ROUTE_GRAPHS:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    for (auto& route_graph : this->route_graphs_) {
                        route_graph->Deserialize(block);
                    }
                });
                loaded_route_graphs = true;
                break;
            case StreamSerializationTag::EXTRA_INFO:
                if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            this->extra_infos_->Deserialize(block);
                        });
                    loaded_extra_info = true;
                }
                break;
            case StreamSerializationTag::ATTRIBUTE_FILTER:
                if (this->use_attribute_filter_ && this->attr_filter_index_ != nullptr) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            this->attr_filter_index_->Deserialize(block);
                        });
                    loaded_attribute_filter = true;
                }
                break;
            case StreamSerializationTag::RAW_VECTOR:
                if (create_new_raw_vector_) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            this->raw_vector_->Deserialize(block);
                        });
                    loaded_raw_vector = true;
                }
                break;
            default:
                if (block_header.IsCritical()) {
                    throw VsagException(
                        ErrorType::UNSUPPORTED_INDEX_OPERATION,
                        fmt::format("unknown HGraph streaming serialization block: tag={}, "
                                    "name={}, version={}, flags={}, value_len={}",
                                    block_header.tag,
                                    StreamSerializationTagName(block_header.tag),
                                    block_header.block_version,
                                    block_header.flags,
                                    block_header.value_len));
                }
                break;
        }
        block_reader.SkipRemaining();
    }

    if (!loaded_label_table || !loaded_base_codes || !loaded_bottom_graph || !loaded_route_graphs) {
        throw VsagException(ErrorType::READ_ERROR,
                            "HGraph streaming serialization required block is missing");
    }
    if (this->has_precise_reorder() && !loaded_high_precision_codes) {
        throw VsagException(ErrorType::READ_ERROR,
                            "HGraph streaming serialization high precision block is missing");
    }
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr && !loaded_extra_info) {
        throw VsagException(ErrorType::READ_ERROR,
                            "HGraph streaming serialization extra info block is missing");
    }
    if (this->use_attribute_filter_ && this->attr_filter_index_ != nullptr &&
        !loaded_attribute_filter) {
        throw VsagException(ErrorType::READ_ERROR,
                            "HGraph streaming serialization attribute filter block is missing");
    }
    if (create_new_raw_vector_ && !loaded_raw_vector) {
        throw VsagException(ErrorType::READ_ERROR,
                            "HGraph streaming serialization raw vector block is missing");
    }

    auto new_size = max_capacity_.load();
    this->neighbors_mutex_->Resize(new_size);
    pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);
    this->total_count_ = this->basic_flatten_codes_->TotalCount();
    if (this->raw_vector_ != nullptr) {
        this->has_raw_vector_ = true;
    }
    this->cal_memory_usage();

    if (use_elp_optimizer_) {
        elp_optimize();
    }
}

void
HGraph::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        logger::debug("parse with v0.14 version format");

        this->deserialize_basic_info_v0_14(reader);

        this->basic_flatten_codes_->Deserialize(reader);
        this->bottom_graph_->Deserialize(reader);
        if (this->has_precise_reorder()) {
            this->high_precise_codes_->Deserialize(reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(reader);
        }
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        BufferStreamReader buffer_reader(
            &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

        auto metadata = footer->GetMetadata();
        // metadata should NOT be nullptr if footer is not nullptr
        this->deserialize_basic_info(metadata->Get(BASIC_INFO));

        int64_t dup_version = 0;
        if (metadata->Get("duplicate_format_version").IsNumberInteger()) {
            dup_version = metadata->Get("duplicate_format_version").GetInt();
        }
        this->label_table_->is_legacy_duplicate_format_ = (dup_version == 0);

        this->deserialize_label_info(buffer_reader);

        this->basic_flatten_codes_->Deserialize(buffer_reader);
        this->bottom_graph_->Deserialize(buffer_reader);
        if (this->has_precise_reorder()) {
            this->high_precise_codes_->Deserialize(buffer_reader);
        }

        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(buffer_reader);
        }
        auto new_size = max_capacity_.load();
        this->neighbors_mutex_->Resize(new_size);

        pool_ = std::make_shared<VisitedListPool>(1, allocator_, new_size, allocator_);

        if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
            this->extra_infos_->Deserialize(buffer_reader);
        }
        this->total_count_ = this->basic_flatten_codes_->TotalCount();

        if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
            this->attr_filter_index_->Deserialize(buffer_reader);
        }

        if (create_new_raw_vector_) {
            this->raw_vector_->Deserialize(buffer_reader);
        }
        if (this->raw_vector_ != nullptr) {
            this->has_raw_vector_ = true;
        }
    }
    this->cal_memory_usage();

    // post serialize procedure
    if (use_elp_optimizer_) {
        elp_optimize();
    }
}

std::unordered_map<std::string, uint64_t>
HGraph::GetMemoryUsageDetail() const {
    std::unordered_map<std::string, uint64_t> memory_usage;
    memory_usage["neighbors_mutex"] = this->neighbors_mutex_->GetMemoryUsage();
    memory_usage["pool"] = this->pool_->GetMemoryUsage();
    memory_usage["label_table"] = this->label_table_->GetMemoryUsage();
    memory_usage["basic_flatten_codes"] = this->basic_flatten_codes_->GetMemoryUsage();
    memory_usage["bottom_graph"] = this->bottom_graph_->GetMemoryUsage();
    uint64_t route_graph_memory = 0;
    for (const auto& route_graph : this->route_graphs_) {
        route_graph_memory += route_graph->GetMemoryUsage();
    }
    memory_usage["route_graph"] = route_graph_memory;
    if (this->has_precise_reorder()) {
        memory_usage["high_precise_codes"] = this->high_precise_codes_->GetMemoryUsage();
    }
    if (this->extra_info_size_ > 0 && this->extra_infos_ != nullptr) {
        memory_usage["extra_infos"] = this->extra_infos_->GetMemoryUsage();
    }
    if (this->create_new_raw_vector_ && this->raw_vector_ != nullptr) {
        memory_usage["raw_vector"] = this->raw_vector_->GetMemoryUsage();
    }
    return memory_usage;
}

}  // namespace vsag
