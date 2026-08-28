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

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "datacell/graph_datacell_parameter.h"
#include "datacell/graph_interface_parameter.h"
#include "graph_bucket_searcher.h"
#include "inner_string_params.h"
#include "io/reader_io/reader_io_parameter.h"
#include "ivf.h"  // IWYU pragma: keep
#include "storage/serialization.h"
#include "storage/serialization_tags.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "storage/tlv_section.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {

static GraphInterfaceParamPtr
get_ivf_graph_param(const std::string& graph_param_string) {
    auto graph_json = JsonType::Parse(graph_param_string);
    if (!graph_json.Contains(IO_PARAMS_KEY) || !graph_json.Contains(GRAPH_STORAGE_TYPE_KEY) ||
        graph_json[GRAPH_STORAGE_TYPE_KEY].GetString() != GRAPH_STORAGE_TYPE_VALUE_FLAT) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "bucket graph requires flat storage with memory-backed IO");
    }

    auto param = std::dynamic_pointer_cast<GraphDataCellParameter>(
        GraphInterfaceParameter::GetGraphParameterByJson(
            GraphStorageTypes::GRAPH_STORAGE_TYPE_VALUE_FLAT, graph_json));
    if (param == nullptr || param->io_parameter_ == nullptr || param->support_remove_) {
        throw VsagException(ErrorType::INVALID_BINARY, "invalid bucket graph parameters");
    }

    const auto io_type = param->io_parameter_->GetTypeName();
    if (io_type != IO_TYPE_VALUE_MEMORY_IO && io_type != IO_TYPE_VALUE_BLOCK_MEMORY_IO) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            fmt::format("unsupported bucket graph IO type: {}", io_type));
    }
    return param;
}

static JsonType
serialize_ivf_graph_param(const GraphInterfaceParamPtr& graph_param) {
    auto graph_json = graph_param->ToJson();
    graph_json[GRAPH_STORAGE_TYPE_KEY].SetString(GRAPH_STORAGE_TYPE_VALUE_FLAT);
    return graph_json;
}

static bool
has_any_fresh_bucket_graph(const BucketInterfacePtr& bucket,
                           const Vector<GraphInterfacePtr>& bucket_graphs) {
    for (BucketIdType b = 0; b < static_cast<BucketIdType>(bucket_graphs.size()); ++b) {
        if (bucket_graphs[b] != nullptr &&
            bucket_graphs[b]->TotalCount() == bucket->GetBucketSize(b)) {
            return true;
        }
    }
    return false;
}

#define WRITE_DATACELL_WITH_NAME(writer, name, datacell)            \
    datacell_offsets[(name)].SetInt(offset);                        \
    auto datacell##_start = (writer).GetCursor();                   \
    (datacell)->Serialize(writer);                                  \
    auto datacell##_size = (writer).GetCursor() - datacell##_start; \
    datacell_sizes[(name)].SetInt(datacell##_size);                 \
    offset += datacell##_size;
void
IVF::Serialize(StreamWriter& writer) const {
    JsonType datacell_offsets;
    JsonType datacell_sizes;
    uint64_t offset = 0;

    WRITE_DATACELL_WITH_NAME(writer, "bucket", bucket_);
    WRITE_DATACELL_WITH_NAME(writer, "partition_strategy", partition_strategy_);
    WRITE_DATACELL_WITH_NAME(writer, "label_table", label_table_);

    if (use_reorder_) {
        if (precise_bucket_ != nullptr) {
            WRITE_DATACELL_WITH_NAME(writer, "precise_bucket", precise_bucket_);
        } else {
            WRITE_DATACELL_WITH_NAME(writer, "reorder_codes", reorder_codes_);
        }
    }

    if (use_attribute_filter_) {
        WRITE_DATACELL_WITH_NAME(writer, "attr_filter_index", attr_filter_index_);
    }

    if (graph_build_threshold_ > 0 && graph_param_ != nullptr &&
        has_any_fresh_bucket_graph(bucket_, bucket_graphs_)) {
        datacell_offsets["bucket_graphs"].SetInt(offset);
        auto bucket_graphs_start = writer.GetCursor();
        StreamWriter::WriteObj(writer, graph_build_threshold_);
        StreamWriter::WriteString(writer, serialize_ivf_graph_param(graph_param_).Dump());
        int64_t graph_count = 0;
        for (BucketIdType b = 0; b < static_cast<BucketIdType>(bucket_graphs_.size()); ++b) {
            if (bucket_graphs_[b] != nullptr &&
                bucket_graphs_[b]->TotalCount() == bucket_->GetBucketSize(b)) {
                ++graph_count;
            }
        }
        StreamWriter::WriteObj(writer, graph_count);
        for (BucketIdType b = 0; b < static_cast<BucketIdType>(bucket_graphs_.size()); ++b) {
            if (bucket_graphs_[b] != nullptr &&
                bucket_graphs_[b]->TotalCount() == bucket_->GetBucketSize(b)) {
                StreamWriter::WriteObj(writer, b);
                bucket_graphs_[b]->Serialize(writer);
            }
        }
        auto bucket_graphs_size = writer.GetCursor() - bucket_graphs_start;
        datacell_sizes["bucket_graphs"].SetInt(bucket_graphs_size);
        offset += bucket_graphs_size;
    }

    // serialize footer (introduced since v0.15)
    JsonType basic_info;
    basic_info["total_elements"].SetInt(this->total_elements_);
    basic_info["use_reorder"].SetBool(this->use_reorder_);
    basic_info["is_trained"].SetBool(this->is_trained_);
    basic_info[DIM].SetInt(this->dim_);
    basic_info[EXTRA_INFO_SIZE].SetInt(0);
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    basic_info["data_type"].SetInt(static_cast<int64_t>(this->data_type_));
    basic_info["metric"].SetInt(static_cast<int64_t>(this->metric_));

    auto metadata = std::make_shared<Metadata>();
    metadata->Set(BASIC_INFO, basic_info);
    metadata->Set("datacell_offsets", datacell_offsets);
    metadata->Set("datacell_sizes", datacell_sizes);

    auto footer = std::make_shared<Footer>(metadata);
    footer->Write(writer);
}

MetadataPtr
IVF::collect_streaming_header() const {
    auto metadata = std::make_shared<Metadata>();
    metadata->Set("format", "vsag_stream_v1");
    metadata->Set("index_name", this->GetName());

    JsonType basic_info;
    basic_info["total_elements"].SetInt(this->total_elements_);
    basic_info["use_reorder"].SetBool(this->use_reorder_);
    basic_info["is_trained"].SetBool(this->is_trained_);
    basic_info[DIM].SetInt(this->dim_);
    basic_info[EXTRA_INFO_SIZE].SetInt(0);
    basic_info[INDEX_PARAM].SetString(this->create_param_ptr_->ToString());
    basic_info["data_type"].SetInt(static_cast<int64_t>(this->data_type_));
    basic_info["metric"].SetInt(static_cast<int64_t>(this->metric_));
    metadata->Set(BASIC_INFO, basic_info);

    JsonType manifest;
    auto bucket_tag = static_cast<uint32_t>(StreamSerializationTag::IVF_BUCKET);
    auto partition_tag = static_cast<uint32_t>(StreamSerializationTag::IVF_PARTITION_STRATEGY);
    auto label_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);
    AppendStreamingManifestBlock(manifest,
                                 bucket_tag,
                                 StreamSerializationBlockCurrentVersion(bucket_tag),
                                 StreamSerializationTagCritical(bucket_tag));
    AppendStreamingManifestBlock(manifest,
                                 partition_tag,
                                 StreamSerializationBlockCurrentVersion(partition_tag),
                                 StreamSerializationTagCritical(partition_tag));
    AppendStreamingManifestBlock(manifest,
                                 label_tag,
                                 StreamSerializationBlockCurrentVersion(label_tag),
                                 StreamSerializationTagCritical(label_tag));
    if (this->use_reorder_) {
        auto tag = static_cast<uint32_t>(precise_bucket_ != nullptr
                                             ? StreamSerializationTag::IVF_PRECISE_BUCKET
                                             : StreamSerializationTag::HIGH_PRECISION_CODES);
        AppendStreamingManifestBlock(manifest,
                                     tag,
                                     StreamSerializationBlockCurrentVersion(tag),
                                     StreamSerializationTagCritical(tag));
    }
    if (this->use_attribute_filter_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::ATTRIBUTE_FILTER);
        AppendStreamingManifestBlock(manifest,
                                     tag,
                                     StreamSerializationBlockCurrentVersion(tag),
                                     StreamSerializationTagCritical(tag));
    }
    if (graph_build_threshold_ > 0 && graph_param_ != nullptr &&
        has_any_fresh_bucket_graph(bucket_, bucket_graphs_)) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::IVF_BUCKET_GRAPH);
        AppendStreamingManifestBlock(manifest,
                                     tag,
                                     StreamSerializationBlockCurrentVersion(tag),
                                     StreamSerializationTagCritical(tag));
    }
    metadata->Set("block_manifest", manifest);
    metadata->SetEmptyIndex(this->GetNumElements() == 0);
    return metadata;
}

void
IVF::serialize_streaming_body(StreamWriter& writer) const {
    auto bucket_tag = static_cast<uint32_t>(StreamSerializationTag::IVF_BUCKET);
    auto partition_tag = static_cast<uint32_t>(StreamSerializationTag::IVF_PARTITION_STRATEGY);
    auto label_tag = static_cast<uint32_t>(StreamSerializationTag::LABEL_TABLE);

    WriteStreamingBlock(
        writer, bucket_tag, StreamSerializationTagCritical(bucket_tag), [this](StreamWriter& w) {
            this->bucket_->Serialize(w);
        });
    WriteStreamingBlock(writer,
                        partition_tag,
                        StreamSerializationTagCritical(partition_tag),
                        [this](StreamWriter& w) { this->partition_strategy_->Serialize(w); });
    WriteStreamingBlock(
        writer, label_tag, StreamSerializationTagCritical(label_tag), [this](StreamWriter& w) {
            this->label_table_->Serialize(w);
        });
    if (this->use_reorder_) {
        auto tag = static_cast<uint32_t>(precise_bucket_ != nullptr
                                             ? StreamSerializationTag::IVF_PRECISE_BUCKET
                                             : StreamSerializationTag::HIGH_PRECISION_CODES);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& w) {
                if (this->precise_bucket_ != nullptr) {
                    this->precise_bucket_->Serialize(w);
                } else {
                    this->reorder_codes_->Serialize(w);
                }
            });
    }
    if (this->use_attribute_filter_) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::ATTRIBUTE_FILTER);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& w) {
                this->attr_filter_index_->Serialize(w);
            });
    }
    if (graph_build_threshold_ > 0 && graph_param_ != nullptr &&
        has_any_fresh_bucket_graph(bucket_, bucket_graphs_)) {
        auto tag = static_cast<uint32_t>(StreamSerializationTag::IVF_BUCKET_GRAPH);
        WriteStreamingBlock(
            writer, tag, StreamSerializationTagCritical(tag), [this](StreamWriter& w) {
                StreamWriter::WriteObj(w, graph_build_threshold_);
                StreamWriter::WriteString(w, serialize_ivf_graph_param(graph_param_).Dump());
                auto graph_count = static_cast<uint64_t>(bucket_graphs_.size());
                StreamWriter::WriteObj(w, graph_count);
                for (uint64_t i = 0; i < graph_count; ++i) {
                    bool has_graph = (bucket_graphs_[i] != nullptr &&
                                      bucket_graphs_[i]->TotalCount() ==
                                          bucket_->GetBucketSize(static_cast<BucketIdType>(i)));
                    StreamWriter::WriteObj(w, has_graph);
                    if (has_graph) {
                        StreamWriter::WriteObj(w, static_cast<BucketIdType>(i));
                        bucket_graphs_[i]->Serialize(w);
                    }
                }
            });
    }
}

void
IVF::deserialize_streaming_body(StreamReader& reader, const MetadataPtr& metadata) {
    this->read_streaming_body(reader, metadata);
}

void
IVF::load_streaming_body(StreamReader& reader,
                         const MetadataPtr& metadata,
                         const LoadParameters& parameters) {
    this->read_streaming_body(reader, metadata, &parameters);
}

void
IVF::read_streaming_body(StreamReader& reader,
                         const MetadataPtr& metadata,
                         const LoadParameters* load_parameters) {
    auto basic_info = metadata->Get(BASIC_INFO);
    this->total_elements_ = basic_info["total_elements"].GetInt();
    this->use_reorder_ = basic_info["use_reorder"].GetBool();
    this->is_trained_ = basic_info["is_trained"].GetBool();
    if (precise_bucket_ != nullptr and not basic_info.Contains(INDEX_PARAM)) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "IVF precise bucket requires persisted index parameters");
    }
    if (basic_info.Contains(INDEX_PARAM)) {
        auto index_param = std::make_shared<IVFParameter>();
        index_param->FromString(basic_info[INDEX_PARAM].GetString());
        if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
            auto message = fmt::format("IVF index parameter not match, current: {}, new: {}",
                                       this->create_param_ptr_->ToString(),
                                       index_param->ToString());
            logger::error(message);
            throw VsagException(ErrorType::INVALID_ARGUMENT, message);
        }
    }

    bool loaded_bucket = false;
    bool loaded_partition = false;
    bool loaded_label_table = false;
    bool loaded_precise_codes = false;
    bool loaded_attribute_filter = false;

    ReaderIOParamPtr precise_reader_param = nullptr;
    auto ivf_param = std::dynamic_pointer_cast<IVFParameter>(create_param_ptr_);
    if (ivf_param != nullptr && ivf_param->precise_codes_param != nullptr &&
        ivf_param->precise_codes_param->io_parameter != nullptr &&
        ivf_param->precise_codes_param->io_parameter->GetTypeName() == IO_TYPE_VALUE_READER_IO) {
        constexpr const char* precise_reader_key = "precise_reader";
        if (load_parameters == nullptr or not load_parameters->HasReader(precise_reader_key)) {
            throw VsagException(ErrorType::INVALID_ARGUMENT,
                                "reader-backed IVF precise codes require precise_reader");
        }
        auto precise_reader = load_parameters->GetReader(precise_reader_key);
        if (precise_reader == nullptr) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, "precise_reader is null");
        }
        precise_reader_param = std::dynamic_pointer_cast<ReaderIOParameter>(
            ivf_param->precise_codes_param->io_parameter);
        if (precise_reader_param == nullptr) {
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "IVF precise reader IO parameter is invalid");
        }
        precise_reader_param->reader = std::move(precise_reader);
    }

    while (true) {
        auto block_header = StreamBlockHeader::Read(reader);
        if (block_header.IsSectionEnd()) {
            break;
        }
        BoundedForwardReader block_reader(&reader, block_header.value_len);
        if (!StreamSerializationBlockVersionSupported(block_header.tag,
                                                      block_header.block_version)) {
            if (block_header.IsCritical()) {
                throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                    fmt::format("unsupported IVF streaming block version: tag={}, "
                                                "name={}, version={}, flags={}, value_len={}",
                                                block_header.tag,
                                                StreamSerializationTagName(block_header.tag),
                                                block_header.block_version,
                                                block_header.flags,
                                                block_header.value_len));
            }
            block_reader.SkipRemaining();
            continue;
        }

        auto read_precise_block = [&](const auto& deserialize, const auto& init_io) {
            if (precise_reader_param == nullptr) {
                ReadSeekableBlockPayload(block_reader, block_header, deserialize);
                return;
            }
            block_reader.SkipRemaining();
            ReadExternalBlockPayload(precise_reader_param->reader, block_header, deserialize);
            init_io(precise_reader_param);
        };

        switch (static_cast<StreamSerializationTag>(block_header.tag)) {
            case StreamSerializationTag::IVF_BUCKET:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->bucket_->Deserialize(block);
                });
                loaded_bucket = true;
                break;
            case StreamSerializationTag::IVF_PARTITION_STRATEGY:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->partition_strategy_->Deserialize(block);
                });
                loaded_partition = true;
                break;
            case StreamSerializationTag::LABEL_TABLE:
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    this->label_table_->Deserialize(block);
                });
                loaded_label_table = true;
                break;
            case StreamSerializationTag::HIGH_PRECISION_CODES:
                if (this->use_reorder_ and this->reorder_codes_ != nullptr) {
                    read_precise_block(
                        [this](StreamReader& block) { this->reorder_codes_->Deserialize(block); },
                        [this](const IOParamPtr& io_param) {
                            this->reorder_codes_->InitIO(io_param);
                        });
                    loaded_precise_codes = true;
                }
                break;
            case StreamSerializationTag::IVF_PRECISE_BUCKET:
                if (this->use_reorder_ and this->precise_bucket_ != nullptr) {
                    read_precise_block(
                        [this](StreamReader& block) { this->precise_bucket_->Deserialize(block); },
                        [this](const IOParamPtr& io_param) {
                            this->precise_bucket_->InitIO(io_param);
                        });
                    loaded_precise_codes = true;
                }
                break;
            case StreamSerializationTag::ATTRIBUTE_FILTER:
                if (this->use_attribute_filter_) {
                    ReadSeekableBlockPayload(
                        block_reader, block_header, [this](StreamReader& block) {
                            this->attr_filter_index_->Deserialize(block);
                        });
                    loaded_attribute_filter = true;
                    this->has_attribute_ = true;
                }
                break;
            case StreamSerializationTag::IVF_BUCKET_GRAPH: {
                ReadSeekableBlockPayload(block_reader, block_header, [this](StreamReader& block) {
                    StreamReader::ReadObj(block, this->graph_build_threshold_);
                    this->graph_param_ = get_ivf_graph_param(StreamReader::ReadString(block));
                    uint64_t graph_count = 0;
                    StreamReader::ReadObj(block, graph_count);
                    auto bucket_count = this->bucket_->bucket_count_;
                    if (graph_count > static_cast<uint64_t>(bucket_count)) {
                        throw VsagException(
                            ErrorType::INVALID_BINARY,
                            fmt::format("bucket graph_count {} exceeds bucket_count {}",
                                        graph_count,
                                        bucket_count));
                    }
                    this->bucket_graphs_.resize(bucket_count);
                    for (uint64_t i = 0; i < graph_count; ++i) {
                        bool has_graph = false;
                        StreamReader::ReadObj(block, has_graph);
                        if (has_graph) {
                            BucketIdType bid = 0;
                            StreamReader::ReadObj(block, bid);
                            auto graph = GraphInterface::MakeInstance(this->graph_param_,
                                                                      this->common_param_);
                            graph->Deserialize(block);
                            if (bid >= 0 && bid < static_cast<BucketIdType>(bucket_count)) {
                                const auto total = graph->TotalCount();
                                const auto expected_total = bucket_->GetBucketSize(bid);
                                if (total != expected_total || total > graph->MaxCapacity()) {
                                    throw VsagException(
                                        ErrorType::INVALID_BINARY,
                                        "corrupt bucket graph: invalid total count");
                                }
                                for (InnerIdType nid = 0; nid < total; ++nid) {
                                    if (graph->GetNeighborSize(nid) > graph->MaximumDegree()) {
                                        throw VsagException(ErrorType::INVALID_BINARY,
                                                            "corrupt bucket graph: neighbor count "
                                                            "exceeds maximum degree");
                                    }
                                    Vector<InnerIdType> nbrs(this->allocator_);
                                    graph->GetNeighbors(nid, nbrs);
                                    for (auto nb : nbrs) {
                                        if (nb >= total) {
                                            throw VsagException(
                                                ErrorType::INVALID_BINARY,
                                                "corrupt bucket graph: neighbor ID out of range");
                                        }
                                    }
                                }
                                this->bucket_graphs_[bid] = std::move(graph);
                            }
                        }
                    }
                });
                if (graph_build_threshold_ > 0 && this->bucket_searcher_ != nullptr) {
                    // Replace flat searcher with graph searcher now that graphs are loaded.
                    this->bucket_searcher_ = std::make_shared<GraphBucketSearcher>(
                        graph_build_threshold_, bucket_graphs_, allocator_);
                }
                break;
            }
            default:
                if (block_header.IsCritical()) {
                    throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                        fmt::format("unknown IVF streaming serialization block: "
                                                    "tag={}, name={}, version={}, flags={}, "
                                                    "value_len={}",
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

    if (!loaded_bucket || !loaded_partition || !loaded_label_table) {
        throw VsagException(ErrorType::READ_ERROR,
                            "IVF streaming serialization required block is missing");
    }
    if (this->use_reorder_ && !loaded_precise_codes) {
        throw VsagException(ErrorType::READ_ERROR,
                            "IVF streaming serialization reorder block is missing");
    }
    if (this->use_attribute_filter_ && !loaded_attribute_filter) {
        throw VsagException(ErrorType::READ_ERROR,
                            "IVF streaming serialization attribute filter block is missing");
    }
    if (this->bucket_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
        this->has_raw_vector_ = true;
    }
    this->fill_location_map();
    this->cal_memory_usage();
}

#define READ_DATACELL_WITH_NAME(reader, name, datacell)                       \
    reader.PushSeek(datacell_offsets[(name)].GetInt());                       \
    (datacell)->Deserialize((reader).Slice(datacell_sizes[(name)].GetInt())); \
    (reader).PopSeek();

void
IVF::Deserialize(StreamReader& reader) {
    // try to deserialize footer (only in new version)
    auto footer = Footer::Parse(reader);

    BufferStreamReader buffer_reader(
        &reader, std::numeric_limits<uint64_t>::max(), this->allocator_);

    if (footer == nullptr) {  // old format, DON'T EDIT, remove in the future
        if (precise_bucket_ != nullptr) {
            throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                "legacy IVF serialization does not support precise bucket");
        }
        logger::debug("parse with v0.14 version format");

        StreamReader::ReadObj(buffer_reader, this->total_elements_);
        StreamReader::ReadObj(buffer_reader, this->use_reorder_);
        StreamReader::ReadObj(buffer_reader, this->is_trained_);

        this->bucket_->Deserialize(buffer_reader);
        this->partition_strategy_->Deserialize(buffer_reader);
        this->label_table_->Deserialize(buffer_reader);
        if (use_reorder_) {
            this->reorder_codes_->Deserialize(buffer_reader);
        }

        if (use_attribute_filter_) {
            this->attr_filter_index_->Deserialize(buffer_reader);
            this->has_attribute_ = true;
        }
    } else {  // create like `else if ( ver in [v0.15, v0.17] )` here if need in the future
        logger::debug("parse with new version format");

        auto metadata = footer->GetMetadata();
        if (metadata->EmptyIndex()) {
            return;
        }

        auto basic_info = metadata->Get(BASIC_INFO);
        this->total_elements_ = basic_info["total_elements"].GetInt();
        this->use_reorder_ = basic_info["use_reorder"].GetBool();
        this->is_trained_ = basic_info["is_trained"].GetBool();
        if (precise_bucket_ != nullptr and not basic_info.Contains(INDEX_PARAM)) {
            throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                                "IVF precise bucket requires persisted index parameters");
        }
        if (basic_info.Contains(INDEX_PARAM)) {
            auto param_str = basic_info[INDEX_PARAM].GetString();
            auto index_param = std::make_shared<IVFParameter>();
            index_param->FromString(param_str);
            if (not this->create_param_ptr_->CheckCompatibility(index_param)) {
                auto message = fmt::format("IVF index parameter not match, current: {}, new: {}",
                                           this->create_param_ptr_->ToString(),
                                           index_param->ToString());
                logger::error(message);
                throw VsagException(ErrorType::INVALID_ARGUMENT, message);
            }
        }

        JsonType datacell_offsets = metadata->Get(DATACELL_OFFSETS);
        logger::debug("datacell_offsets: {}", datacell_offsets.Dump());
        JsonType datacell_sizes = metadata->Get(DATACELL_SIZES);
        logger::debug("datacell_sizes: {}", datacell_sizes.Dump());

        READ_DATACELL_WITH_NAME(buffer_reader, "bucket", this->bucket_);
        READ_DATACELL_WITH_NAME(buffer_reader, "partition_strategy", this->partition_strategy_);
        READ_DATACELL_WITH_NAME(buffer_reader, "label_table", this->label_table_);
        if (use_reorder_) {
            if (precise_bucket_ != nullptr) {
                READ_DATACELL_WITH_NAME(buffer_reader, "precise_bucket", this->precise_bucket_);
            } else {
                READ_DATACELL_WITH_NAME(buffer_reader, "reorder_codes", this->reorder_codes_);
            }
        }
        if (use_attribute_filter_) {
            READ_DATACELL_WITH_NAME(buffer_reader, "attr_filter_index", this->attr_filter_index_);
            this->has_attribute_ = true;
        }
        if (this->bucket_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_FP32) {
            this->has_raw_vector_ = true;
        }

        if (datacell_offsets.Contains("bucket_graphs")) {
            auto bucket_count = bucket_->bucket_count_;
            bucket_graphs_.resize(bucket_count);
            buffer_reader.PushSeek(datacell_offsets["bucket_graphs"].GetInt());
            auto graph_reader = buffer_reader.Slice(datacell_sizes["bucket_graphs"].GetInt());
            int64_t stored_threshold = 0;
            StreamReader::ReadObj(graph_reader, stored_threshold);
            graph_build_threshold_ = stored_threshold;
            graph_param_ = get_ivf_graph_param(StreamReader::ReadString(graph_reader));
            int64_t graph_count = 0;
            StreamReader::ReadObj(graph_reader, graph_count);
            CHECK_ARGUMENT(graph_count >= 0, "bucket graph_count is negative");
            CHECK_ARGUMENT(graph_count <= bucket_count, "bucket graph_count exceeds bucket_count");
            for (int64_t g = 0; g < graph_count; ++g) {
                BucketIdType bid = 0;
                StreamReader::ReadObj(graph_reader, bid);
                auto graph = GraphInterface::MakeInstance(graph_param_, this->common_param_);
                graph->Deserialize(graph_reader);
                if (bid >= 0 && bid < bucket_count) {
                    const auto total = graph->TotalCount();
                    const auto expected_total = bucket_->GetBucketSize(bid);
                    if (total != expected_total || total > graph->MaxCapacity()) {
                        throw VsagException(ErrorType::INVALID_BINARY,
                                            "corrupt bucket graph: invalid total count");
                    }
                    for (InnerIdType nid = 0; nid < total; ++nid) {
                        if (graph->GetNeighborSize(nid) > graph->MaximumDegree()) {
                            throw VsagException(
                                ErrorType::INVALID_BINARY,
                                "corrupt bucket graph: neighbor count exceeds maximum degree");
                        }
                        Vector<InnerIdType> nbrs(allocator_);
                        graph->GetNeighbors(nid, nbrs);
                        for (auto nb : nbrs) {
                            if (nb >= total) {
                                throw VsagException(
                                    ErrorType::INVALID_BINARY,
                                    "corrupt bucket graph: neighbor ID out of range");
                            }
                        }
                    }
                    bucket_graphs_[bid] = std::move(graph);
                }
            }
            buffer_reader.PopSeek();
            if (graph_build_threshold_ > 0) {
                this->bucket_searcher_ = std::make_shared<GraphBucketSearcher>(
                    graph_build_threshold_, bucket_graphs_, allocator_);
            }
        }
    }
    this->fill_location_map();
    this->cal_memory_usage();
}
}  // namespace vsag
