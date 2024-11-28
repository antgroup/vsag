
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

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <exception>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "index/diskann.h"
#include "index/diskann_zparameters.h"
#include "index/hgraph_index.h"
#include "index/hgraph_zparameters.h"
#include "index/hnsw.h"
#include "index/hnsw_zparameters.h"
#include "index/index_common_param.h"
#include "vsag/vsag.h"

namespace vsag {

tl::expected<std::shared_ptr<Index>, Error>
Factory::CreateIndex(const std::string& origin_name,
                     const std::string& parameters,
                     Allocator* allocator) {
    try {
        std::string name = origin_name;
        transform(name.begin(), name.end(), name.begin(), ::tolower);
        JsonType parsed_params = JsonType::parse(parameters);
        auto index_common_params = IndexCommonParam::CheckAndCreate(parsed_params, allocator);
        if (name == INDEX_HNSW) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_HNSW),
                           fmt::format("parameters must contains {}", INDEX_HNSW));
            auto& hnsw_param_obj = parsed_params[INDEX_HNSW];
            auto hnsw_params = HnswParameters::FromJson(hnsw_param_obj, index_common_params);
            logger::debug("created a hnsw index");
            return std::make_shared<HNSW>(hnsw_params, index_common_params);
        } else if (name == INDEX_FRESH_HNSW) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_HNSW),
                           fmt::format("parameters must contains {}", INDEX_HNSW));
            auto& hnsw_param_obj = parsed_params[INDEX_HNSW];
            auto hnsw_params = FreshHnswParameters::FromJson(hnsw_param_obj, index_common_params);
            logger::debug("created a fresh-hnsw index");
            return std::make_shared<HNSW>(hnsw_params, index_common_params);
        } else if (name == INDEX_DISKANN) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_DISKANN),
                           fmt::format("parameters must contains {}", INDEX_DISKANN));
            auto& diskann_param_obj = parsed_params[INDEX_DISKANN];
            auto diskann_params =
                DiskannParameters::FromJson(diskann_param_obj, index_common_params);
            logger::debug("created a diskann index");
            return std::make_shared<DiskANN>(diskann_params, index_common_params);
        } else if (name == INDEX_HGRAPH) {
            if (allocator == nullptr) {
                index_common_params.allocator_ = DefaultAllocator::Instance().get();
            }
            logger::debug("created a hgraph index");
            JsonType hgraph_params;
            if (parsed_params.contains(INDEX_PARAM)) {
                hgraph_params = std::move(parsed_params[INDEX_PARAM]);
            }
            HGraphParameters hgraph_param(hgraph_params, index_common_params);
            auto hgraph_index =
                std::make_shared<HGraphIndex>(hgraph_param.GetJson(), index_common_params);
            hgraph_index->Init();
            return hgraph_index;
        } else {
            LOG_ERROR_AND_RETURNS(
                ErrorType::UNSUPPORTED_INDEX, "failed to create index(unsupported): ", name);
        }
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to create index(invalid argument): ", e.what());
    } catch (const std::exception& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::UNSUPPORTED_INDEX, "failed to create index(unknown error): ", e.what());
    }
}

class LocalFileReader : public Reader {
public:
    LocalFileReader(const std::string& filename, int64_t base_offset = 0, int64_t size = 0)
        : filename_(filename),
          file_(std::ifstream(filename, std::ios::binary)),
          base_offset_(base_offset),
          size_(size) {
        pool_ = std::make_unique<progschj::ThreadPool>(Option::Instance().num_threads_io());
    }

    ~LocalFileReader() {
        file_.close();
    }

    virtual void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::lock_guard<std::mutex> lock(mutex_);
        file_.seekg(base_offset_ + offset, std::ios::beg);
        file_.read((char*)dest, len);
    }

    virtual void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
        pool_->enqueue([this, offset, len, dest, callback]() {
            this->Read(offset, len, dest);
            callback(IOErrorCode::IO_SUCCESS, "success");
        });
    }

    virtual uint64_t
    Size() const override {
        return size_;
    }

private:
    const std::string filename_;
    std::ifstream file_;
    int64_t base_offset_;
    uint64_t size_;
    std::mutex mutex_;
    std::unique_ptr<progschj::ThreadPool> pool_;
};

std::shared_ptr<Reader>
Factory::CreateLocalFileReader(const std::string& filename, int64_t base_offset, int64_t size) {
    return std::make_shared<LocalFileReader>(filename, base_offset, size);
}

}  // namespace vsag
