
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

#include "vsag/factory.h"

#include <fmt/format.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "algorithm/bruteforce/bruteforce.h"
#include "algorithm/bruteforce/bruteforce_parameter.h"
#include "algorithm/hgraph/hgraph.h"
#include "algorithm/hgraph/hgraph_parameter.h"
#include "algorithm/ivf/ivf.h"
#include "algorithm/ivf/ivf_parameter.h"
#include "algorithm/pyramid/pyramid.h"
#include "algorithm/pyramid/pyramid_zparameters.h"
#include "algorithm/sindi/sindi.h"
#include "algorithm/sindi/sindi_parameter.h"
#include "common.h"
#include "data_type.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "json_wrapper.h"
#include "metric_type.h"
#include "storage/serialization.h"
#include "storage/serialization_tags.h"
#include "storage/stream_reader.h"
#include "storage/tlv_section.h"
#include "vsag/constants.h"
#include "vsag/engine.h"
#include "vsag/index.h"
#include "vsag/options.h"
#include "vsag/resource.h"

namespace vsag {
namespace {

IndexCommonParam
make_streaming_common_param(const MetadataPtr& metadata, Allocator* allocator) {
    auto resource = allocator == nullptr ? Resource(Engine::CreateDefaultAllocator(), nullptr)
                                         : Resource(allocator, nullptr);

    IndexCommonParam common_param;
    common_param.allocator_ = resource.GetAllocator();
    common_param.thread_pool_ = std::dynamic_pointer_cast<SafeThreadPool>(resource.GetThreadPool());

    auto basic_info = metadata->Get("basic_info");
    if (basic_info.Contains("dim")) {
        common_param.dim_ = basic_info["dim"].GetInt();
    }
    if (basic_info.Contains("metric")) {
        common_param.metric_ = static_cast<MetricType>(basic_info["metric"].GetInt());
    }
    if (basic_info.Contains("data_type")) {
        common_param.data_type_ = static_cast<DataTypes>(basic_info["data_type"].GetInt());
    }
    if (basic_info.Contains("extra_info_size")) {
        common_param.extra_info_size_ = basic_info["extra_info_size"].GetInt();
    }
    return common_param;
}

void
require_string_load_parameter(const JsonType& json, const std::string& key) {
    CHECK_ARGUMENT(json[key].IsString(),
                   fmt::format("streaming load parameter '{}' must be a string", key));
}

bool
is_valid_streaming_io_type(const char* io_type) {
    const std::string_view type(io_type);
    return type == IO_TYPE_VALUE_MEMORY_IO || type == IO_TYPE_VALUE_BLOCK_MEMORY_IO ||
           type == IO_TYPE_VALUE_BUFFER_IO || type == IO_TYPE_VALUE_ASYNC_IO ||
           type == IO_TYPE_VALUE_MMAP_IO || type == IO_TYPE_VALUE_READER_IO ||
           type == IO_TYPE_VALUE_URING_IO;
}

void
set_streaming_io_override(JsonType& index_param,
                          const char* block_key,
                          const char* io_type,
                          const char* file_path) {
    if (!index_param.Contains(block_key)) {
        return;
    }
    if (io_type != nullptr) {
        CHECK_ARGUMENT(is_valid_streaming_io_type(io_type),
                       std::string("unsupported streaming load io_type: ") + io_type);
        index_param[block_key][IO_PARAMS_KEY][TYPE_KEY].SetString(io_type);
    }
    if (file_path != nullptr) {
        index_param[block_key][IO_PARAMS_KEY][IO_FILE_PATH_KEY].SetString(file_path);
    }
}

void
apply_hgraph_streaming_load_parameters(JsonType& index_param, const std::string& parameters) {
    auto load_json = JsonType::Parse(parameters.empty() ? "{}" : parameters);
    if (load_json.Contains(HGRAPH_BASE_IO_TYPE)) {
        require_string_load_parameter(load_json, HGRAPH_BASE_IO_TYPE);
        set_streaming_io_override(index_param,
                                  BASE_CODES_KEY,
                                  load_json[HGRAPH_BASE_IO_TYPE].GetString().c_str(),
                                  nullptr);
    }
    if (load_json.Contains(HGRAPH_BASE_FILE_PATH)) {
        require_string_load_parameter(load_json, HGRAPH_BASE_FILE_PATH);
        set_streaming_io_override(index_param,
                                  BASE_CODES_KEY,
                                  nullptr,
                                  load_json[HGRAPH_BASE_FILE_PATH].GetString().c_str());
    }
    if (load_json.Contains(HGRAPH_PRECISE_IO_TYPE)) {
        require_string_load_parameter(load_json, HGRAPH_PRECISE_IO_TYPE);
        set_streaming_io_override(index_param,
                                  PRECISE_CODES_KEY,
                                  load_json[HGRAPH_PRECISE_IO_TYPE].GetString().c_str(),
                                  nullptr);
    }
    if (load_json.Contains(HGRAPH_PRECISE_FILE_PATH)) {
        require_string_load_parameter(load_json, HGRAPH_PRECISE_FILE_PATH);
        set_streaming_io_override(index_param,
                                  PRECISE_CODES_KEY,
                                  nullptr,
                                  load_json[HGRAPH_PRECISE_FILE_PATH].GetString().c_str());
    }
    if (load_json.Contains(HGRAPH_BASE_DIRECT_READ)) {
        CHECK_ARGUMENT(load_json[HGRAPH_BASE_DIRECT_READ].IsBool(),
                       "base_direct_read must be a boolean");
        if (index_param.Contains(BASE_CODES_KEY)) {
            index_param[BASE_CODES_KEY][IO_PARAMS_KEY][IO_DIRECT_READ_KEY].SetBool(
                load_json[HGRAPH_BASE_DIRECT_READ].GetBool());
        }
    }
    if (load_json.Contains(HGRAPH_PRECISE_DIRECT_READ)) {
        CHECK_ARGUMENT(load_json[HGRAPH_PRECISE_DIRECT_READ].IsBool(),
                       "precise_direct_read must be a boolean");
        if (index_param.Contains(PRECISE_CODES_KEY)) {
            index_param[PRECISE_CODES_KEY][IO_PARAMS_KEY][IO_DIRECT_READ_KEY].SetBool(
                load_json[HGRAPH_PRECISE_DIRECT_READ].GetBool());
        }
    }
    if (load_json.Contains(RAW_VECTOR_IO_TYPE)) {
        require_string_load_parameter(load_json, RAW_VECTOR_IO_TYPE);
        set_streaming_io_override(index_param,
                                  RAW_VECTOR_KEY,
                                  load_json[RAW_VECTOR_IO_TYPE].GetString().c_str(),
                                  nullptr);
    }
    if (load_json.Contains(RAW_VECTOR_FILE_PATH)) {
        require_string_load_parameter(load_json, RAW_VECTOR_FILE_PATH);
        set_streaming_io_override(index_param,
                                  RAW_VECTOR_KEY,
                                  nullptr,
                                  load_json[RAW_VECTOR_FILE_PATH].GetString().c_str());
    }
}

struct streaming_index_load_target {
    IndexPtr index;
    InnerIndexPtr inner_index;
};

template <typename IndexT, typename ParamT>
streaming_index_load_target
create_streaming_index(const JsonType& index_param, const IndexCommonParam& common_param) {
    auto param = std::make_shared<ParamT>();
    param->FromJson(index_param);
    auto inner_index = std::make_shared<IndexT>(param, common_param);
    return {std::make_shared<IndexImpl<IndexT>>(inner_index, common_param), inner_index};
}

tl::expected<streaming_index_load_target, Error>
create_streaming_index_from_metadata(const MetadataPtr& metadata,
                                     const std::string& parameters,
                                     Allocator* allocator) {
    auto index_name_json = metadata->Get("index_name");
    if (!index_name_json.IsString()) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "streaming metadata missing string field: index_name");
    }
    const auto index_name = index_name_json.GetString();

    auto basic_info = metadata->Get(BASIC_INFO);
    if (!basic_info.IsObject()) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            "streaming metadata missing object field: basic_info");
    }

    std::string build_parameters;
    std::string build_parameters_source = "basic_info.index_param";
    if (basic_info.Contains(INDEX_PARAM) && basic_info[INDEX_PARAM].IsString()) {
        build_parameters = basic_info[INDEX_PARAM].GetString();
    } else {
        auto build_parameters_json = metadata->Get("build_param_snapshot");
        if (!build_parameters_json.IsString()) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                "streaming metadata missing string field: build_param_snapshot");
        }
        build_parameters_source = "build_param_snapshot";
        build_parameters = build_parameters_json.GetString();
    }

    auto index_param = JsonType::Parse(build_parameters, false);
    if (index_param.IsDiscarded() || !index_param.IsObject()) {
        throw VsagException(
            ErrorType::INVALID_BINARY,
            fmt::format("streaming metadata {} must be a JSON object", build_parameters_source));
    }
    auto common_param = make_streaming_common_param(metadata, allocator);

    if (index_name == INDEX_BRUTE_FORCE || index_name == INDEX_WARP) {
        return create_streaming_index<BruteForce, BruteForceParameter>(index_param, common_param);
    }
    if (index_name == INDEX_HGRAPH) {
        apply_hgraph_streaming_load_parameters(index_param, parameters);
        return create_streaming_index<HGraph, HGraphParameter>(index_param, common_param);
    }
    if (index_name == INDEX_IVF) {
        return create_streaming_index<IVF, IVFParameter>(index_param, common_param);
    }
    if (index_name == INDEX_PYRAMID) {
        return create_streaming_index<Pyramid, PyramidParameters>(index_param, common_param);
    }
    if (index_name == INDEX_SINDI) {
        return create_streaming_index<SINDI, SINDIParameter>(index_param, common_param);
    }

    LOG_ERROR_AND_RETURNS(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                          "streaming load does not support index type: ",
                          index_name);
}

}  // namespace

tl::expected<std::shared_ptr<Index>, Error>
Factory::CreateIndex(const std::string& origin_name,
                     const std::string& parameters,
                     Allocator* allocator) {
    std::shared_ptr<Resource> resource{nullptr};
    if (allocator == nullptr) {
        resource = std::make_shared<Resource>(Engine::CreateDefaultAllocator(), nullptr);
    } else {
        resource = std::make_shared<Resource>(allocator, nullptr);
    }
    Engine e(resource.get());
    return e.CreateIndex(origin_name, parameters);
}

tl::expected<StreamingIndexMetadata, Error>
Index::GetStreamingMetadata(std::istream& in_stream) {
    try {
        auto read_metadata = [](StreamReader& reader,
                                uint64_t stream_begin) -> StreamingIndexMetadata {
            auto stream_header = StreamHeader::ReadRaw(reader);

            StreamingIndexMetadata result;
            result.metadata_json = std::move(stream_header.metadata_string);

            struct manifest_block {
                std::string name;
                uint32_t tag{0};
                uint32_t version{0};
                bool critical{false};
                uint64_t payload_size{0};
                bool has_payload_size{false};
            };
            std::vector<manifest_block> manifest_blocks;
            auto manifest = stream_header.metadata->Get("block_manifest");
            if (manifest.IsArray()) {
                const auto* manifest_json = manifest.GetInnerJson();
                manifest_blocks.reserve(manifest_json->size());
                for (const auto& block_json : *manifest_json) {
                    manifest_block block;
                    block.name = block_json.value("name", std::string{});
                    block.tag = block_json.value("tag", 0U);
                    block.version = block_json.value("version", 0U);
                    block.critical = block_json.value("critical", false);
                    block.has_payload_size = block_json.contains("payload_size");
                    block.payload_size = block_json.value("payload_size", uint64_t{0});
                    manifest_blocks.emplace_back(std::move(block));
                }
            }

            if (stream_header.metadata->EmptyIndex()) {
                auto block_header = StreamBlockHeader::Read(reader);
                if (!block_header.IsSectionEnd()) {
                    throw VsagException(ErrorType::INVALID_BINARY,
                                        "empty streaming body must end without blocks");
                }
                return result;
            }

            uint64_t manifest_index = 0;
            while (true) {
                const uint64_t header_offset = reader.GetCursor() - stream_begin;
                auto block_header = StreamBlockHeader::Read(reader);
                if (block_header.IsSectionEnd()) {
                    break;
                }

                StreamingBlockLayout block;
                block.name = StreamSerializationTagName(block_header.tag);
                block.tag = block_header.tag;
                block.version = block_header.block_version;
                block.critical = block_header.IsCritical();
                block.header_offset = header_offset;
                block.payload_offset = reader.GetCursor() - stream_begin;
                block.payload_size = block_header.value_len;

                if (!manifest_blocks.empty()) {
                    if (manifest_index >= static_cast<uint64_t>(manifest_blocks.size())) {
                        throw VsagException(ErrorType::INVALID_BINARY,
                                            "streaming body has more blocks than manifest");
                    }
                    const auto& manifest_block =
                        manifest_blocks[static_cast<size_t>(manifest_index)];
                    if (manifest_block.tag != block.tag ||
                        manifest_block.version != block.version ||
                        manifest_block.critical != block.critical) {
                        throw VsagException(ErrorType::INVALID_BINARY,
                                            "streaming body block does not match manifest");
                    }
                    if (manifest_block.has_payload_size && manifest_block.payload_size != 0 &&
                        manifest_block.payload_size != block.payload_size) {
                        throw VsagException(
                            ErrorType::INVALID_BINARY,
                            "streaming body block payload size does not match manifest");
                    }
                    if (!manifest_block.name.empty()) {
                        block.name = manifest_block.name;
                    }
                }

                result.blocks.emplace_back(std::move(block));
                SkipBlockPayload(reader, block_header);
                ++manifest_index;
            }
            if (!manifest_blocks.empty() && manifest_index != manifest_blocks.size()) {
                throw VsagException(ErrorType::INVALID_BINARY,
                                    "streaming body has fewer blocks than manifest");
            }

            return result;
        };

        const auto stream_pos = in_stream.tellg();
        if (stream_pos == std::istream::pos_type(-1)) {
            in_stream.clear();
            ForwardStreamReader reader(in_stream);
            return read_metadata(reader, 0);
        }

        IOStreamReader reader(in_stream);
        return read_metadata(reader, reader.GetCursor());
    } catch (const vsag::VsagException& e) {
        LOG_ERROR_AND_RETURNS(e.error_.type, e.error_.message);
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY, "not enough memory: ", e.what());
    } catch (const std::exception& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknownError: ", e.what());
    } catch (...) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknown error");
    }
}

tl::expected<IndexPtr, Error>
Index::Load(std::istream& in_stream, const LoadParameters& parameters, Allocator* allocator) {
    try {
        const auto parameter_string = parameters.Dump();
        auto load_parameters =
            JsonType::Parse(parameter_string.empty() ? "{}" : parameter_string, false);
        CHECK_ARGUMENT(not load_parameters.IsDiscarded(),
                       "streaming load parameters must be valid JSON");
        CHECK_ARGUMENT(load_parameters.IsObject(),
                       "streaming load parameters must be a JSON object");

        ForwardStreamReader reader(in_stream);
        auto metadata = StreamHeader::Read(reader);

        auto index = create_streaming_index_from_metadata(metadata, parameter_string, allocator);
        if (not index.has_value()) {
            return tl::unexpected(index.error());
        }

        index.value().inner_index->LoadStreamingBody(reader, metadata, parameters);
        return index.value().index;
    } catch (const vsag::VsagException& e) {
        LOG_ERROR_AND_RETURNS(e.error_.type, e.error_.message);
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::NO_ENOUGH_MEMORY, "not enough memory: ", e.what());
    } catch (const std::exception& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknownError: ", e.what());
    } catch (...) {
        LOG_ERROR_AND_RETURNS(ErrorType::UNKNOWN_ERROR, "unknown error");
    }
}

class LocalFileReader : public Reader {
public:
    explicit LocalFileReader(const std::string& filename,
                             int64_t base_offset = 0,
                             int64_t size = 0,
                             std::shared_ptr<SafeThreadPool> pool = nullptr)
        : filename_(filename),
          file_(std::ifstream(filename, std::ios::binary)),
          base_offset_(base_offset),
          size_(size),
          pool_(std::move(pool)),
          valid_(file_.is_open()) {
    }

    ~LocalFileReader() override {
        file_.close();
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        if (!valid_) {
            throw std::runtime_error("LocalFileReader: failed to open file: " + filename_);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        file_.seekg(static_cast<int64_t>(base_offset_ + offset), std::ios::beg);
        file_.read(static_cast<char*>(dest), static_cast<int64_t>(len));
        if (file_.fail()) {
            throw std::runtime_error("LocalFileReader: read failed on file: " + filename_);
        }
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        {
            std::scoped_lock lock(mutex_);
            if (not pool_) {
                pool_ = SafeThreadPool::FactoryDefaultThreadPool();
            }
        }
        pool_->GeneralEnqueue([this,  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)
                               offset,
                               len,
                               dest,
                               callback]() {
            try {
                this->Read(offset, len, dest);
                callback(IOErrorCode::IO_SUCCESS, "success");
            } catch (const std::exception& e) {
                callback(IOErrorCode::IO_ERROR, e.what());
            }
        });
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return size_;
    }

private:
    const std::string filename_;
    std::ifstream file_;
    int64_t base_offset_;
    uint64_t size_;
    std::mutex mutex_;
    std::shared_ptr<SafeThreadPool> pool_;
    bool valid_;
};

std::shared_ptr<Reader>
Factory::CreateLocalFileReader(const std::string& filename, int64_t base_offset, int64_t size) {
    return std::make_shared<LocalFileReader>(filename, base_offset, size);
}

class ReadFuncReader : public Reader {
public:
    ReadFuncReader(ReadFunc read_func, uint64_t base_offset, uint64_t size)
        : read_func_(std::move(read_func)), base_offset_(base_offset), size_(size) {
        if (!read_func_) {
            throw std::runtime_error("ReadFuncReader: read_func is empty");
        }
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        if (offset > size_ || len > size_ - offset) {
            throw std::runtime_error("ReadFuncReader: read range is out of bounds");
        }
        std::lock_guard<std::mutex> lock(read_mutex_);
        read_func_(base_offset_ + offset, len, dest);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        try {
            this->Read(offset, len, dest);
            callback(IOErrorCode::IO_SUCCESS, "success");
        } catch (const std::exception& e) {
            callback(IOErrorCode::IO_ERROR, e.what());
        }
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return size_;
    }

private:
    ReadFunc read_func_;
    uint64_t base_offset_{0};
    uint64_t size_{0};
    std::mutex read_mutex_;
};

std::shared_ptr<Reader>
Factory::CreateReadFuncReader(ReadFunc read_func, uint64_t size) {
    return std::make_shared<ReadFuncReader>(std::move(read_func), 0, size);
}

std::shared_ptr<Reader>
Factory::CreateReadFuncReader(ReadFunc read_func, uint64_t base_offset, uint64_t size) {
    return std::make_shared<ReadFuncReader>(std::move(read_func), base_offset, size);
}

}  // namespace vsag
