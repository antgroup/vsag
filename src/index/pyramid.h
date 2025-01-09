
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

#pragma once

#include <utility>

#include "pyramid_zparameters.h"
#include "safe_allocator.h"

namespace vsag {

class SubReader : public Reader {
public:
    SubReader(std::shared_ptr<Reader> parrent_reader, uint64_t start_pos, uint64_t size)
        : parrent_reader_(std::move(parrent_reader)), size_(size), start_pos_(start_pos) {
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        if (offset + len > size_)
            throw std::out_of_range("Read out of range.");
        parrent_reader_->Read(offset + start_pos_, len, dest);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, CallBack callback) override {
    }

    uint64_t
    Size() const override {
        return size_;
    }

private:
    std::shared_ptr<Reader> parrent_reader_;
    uint64_t size_;
    uint64_t start_pos_;
};

Binary
binaryset_to_binary(const BinarySet binary_set);
BinarySet
binary_to_binaryset(const Binary binary);
ReaderSet
reader_to_readerset(std::shared_ptr<Reader> reader);

struct IndexNode {
    std::shared_ptr<Index> index{nullptr};
    UnorderedMap<std::string, std::shared_ptr<IndexNode>> children;
    std::string name;
    IndexNode(Allocator* allocator) : children(allocator) {
    }

    void
    CreateIndex(IndexBuildFunction func) {
        index = func();
    }
};

class Pyramid : public Index {
public:
    Pyramid(PyramidParameters pyramid_param, const IndexCommonParam& commom_param)
        : indexes_(commom_param.allocator_.get()),
          pyramid_param_(std::move(pyramid_param)),
          commom_param_(std::move(commom_param)) {
    }

    ~Pyramid() = default;

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override;

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override;

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override;

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override;

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override;

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override;

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size = -1) const override;

    tl::expected<BinarySet, Error>
    Serialize() const override;

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override;

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override;

    int64_t
    GetNumElements() const override;

    int64_t
    GetMemoryUsage() const override;

private:
    inline std::shared_ptr<IndexNode>
    try_get_node_with_init(UnorderedMap<std::string, std::shared_ptr<IndexNode>>& index_map,
                           const std::string& key) {
        auto iter = index_map.find(key);
        std::shared_ptr<IndexNode> node = nullptr;
        if (iter == index_map.end()) {
            node = std::make_shared<IndexNode>(commom_param_.allocator_.get());
            index_map[key] = node;
        } else {
            node = iter->second;
        }
        return node;
    }

private:
    IndexCommonParam commom_param_;
    PyramidParameters pyramid_param_;
    UnorderedMap<std::string, std::shared_ptr<IndexNode>> indexes_;
    int64_t data_num_{0};
};

}  // namespace vsag
