
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

#include "ThreadPool.h"
#include "vsag/engine.h"
#include "vsag/options.h"

namespace vsag {

tl::expected<std::shared_ptr<Index>, Error>
Factory::CreateIndex(const std::string& origin_name,
                     const std::string& parameters,
                     Allocator* allocator) {
    Resource resource(allocator, nullptr);
    Engine e(&resource);
    return e.CreateIndex(origin_name, parameters);
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
