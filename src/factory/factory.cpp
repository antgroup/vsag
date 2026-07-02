
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

#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <string>

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
#include "metric_type.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "vsag/constants.h"
#include "vsag/engine.h"
#include "vsag/index.h"
#include "vsag/options.h"
#include "vsag/resource.h"

namespace vsag {
namespace {

IndexCommonParam
make_streaming_common_param(const MetadataPtr& metadata, Allocator* allocator) {
    auto resource = allocator == nullptr
                        ? std::make_shared<Resource>(Engine::CreateDefaultAllocator(), nullptr)
                        : std::make_shared<Resource>(allocator, nullptr);

    IndexCommonParam common_param;
    common_param.allocator_ = resource->GetAllocator();
    common_param.thread_pool_ =
        std::dynamic_pointer_cast<SafeThreadPool>(resource->GetThreadPool());

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
set_streaming_io_override(JsonType& index_param,
                          const char* block_key,
                          const char* io_type,
                          const char* file_path) {
    if (!index_param.Contains(block_key)) {
        return;
    }
    if (io_type != nullptr) {
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
        set_streaming_io_override(index_param,
                                  BASE_CODES_KEY,
                                  load_json[HGRAPH_BASE_IO_TYPE].GetString().c_str(),
                                  nullptr);
    }
    if (load_json.Contains(HGRAPH_BASE_FILE_PATH)) {
        set_streaming_io_override(index_param,
                                  BASE_CODES_KEY,
                                  nullptr,
                                  load_json[HGRAPH_BASE_FILE_PATH].GetString().c_str());
    }
    if (load_json.Contains(HGRAPH_PRECISE_IO_TYPE)) {
        set_streaming_io_override(index_param,
                                  PRECISE_CODES_KEY,
                                  load_json[HGRAPH_PRECISE_IO_TYPE].GetString().c_str(),
                                  nullptr);
    }
    if (load_json.Contains(HGRAPH_PRECISE_FILE_PATH)) {
        set_streaming_io_override(index_param,
                                  PRECISE_CODES_KEY,
                                  nullptr,
                                  load_json[HGRAPH_PRECISE_FILE_PATH].GetString().c_str());
    }
    if (load_json.Contains(RAW_VECTOR_IO_TYPE)) {
        set_streaming_io_override(index_param,
                                  RAW_VECTOR_KEY,
                                  load_json[RAW_VECTOR_IO_TYPE].GetString().c_str(),
                                  nullptr);
    }
    if (load_json.Contains(RAW_VECTOR_FILE_PATH)) {
        set_streaming_io_override(index_param,
                                  RAW_VECTOR_KEY,
                                  nullptr,
                                  load_json[RAW_VECTOR_FILE_PATH].GetString().c_str());
    }
}

struct StreamingIndexLoadTarget {
    IndexPtr index;
    InnerIndexPtr inner_index;
};

template <typename IndexT, typename ParamT>
StreamingIndexLoadTarget
create_streaming_index(const JsonType& index_param, const IndexCommonParam& common_param) {
    auto param = std::make_shared<ParamT>();
    param->FromJson(index_param);
    auto inner_index = std::make_shared<IndexT>(param, common_param);
    return {std::make_shared<IndexImpl<IndexT>>(inner_index, common_param), inner_index};
}

tl::expected<StreamingIndexLoadTarget, Error>
create_streaming_index_from_metadata(const MetadataPtr& metadata,
                                     const std::string& parameters,
                                     Allocator* allocator) {
    const auto index_name = metadata->Get("index_name").GetString();
    const auto build_parameters = metadata->Get("build_param_snapshot").GetString();
    auto index_param = JsonType::Parse(build_parameters);
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

tl::expected<IndexPtr, Error>
Index::Load(std::istream& in_stream, const std::string& parameters, Allocator* allocator) {
    try {
        ForwardStreamReader reader(in_stream);
        auto metadata = StreamHeader::Read(reader);

        auto index = create_streaming_index_from_metadata(metadata, parameters, allocator);
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

    uint64_t
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

}  // namespace vsag
