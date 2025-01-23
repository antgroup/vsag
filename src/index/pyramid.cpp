
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

#include "pyramid.h"

namespace vsag {

Binary
binaryset_to_binary(const BinarySet binary_set) {
    /*
     * The serialized layout of the Binary data in memory will be as follows:
     * | key_size_0 | key_0 (L_0 bytes) | binary_size_0 | binary_data_0 (S_0 bytes) |
     * | key_size_1 | key_1 (L_1 bytes) | binary_size_1 | binary_data_1 (S_1 bytes) |
     * | ...         | ...               | ...            | ...                        |
     * | key_size_(N-1) | key_(N-1) (L_(N-1) bytes) | binary_size_(N-1) | binary_data_(N-1) (S_(N-1) bytes) |
     * Where:
     * - `key_size_k`: size of the k-th key (in bytes)
     * - `key_k`: the actual k-th key data (length L_k)
     * - `binary_size_k`: size of the binary data associated with the k-th key (in bytes)
     * - `binary_data_k`: the actual binary data contents (length S_k)
     * - N: total number of keys in the BinarySet
     */
    size_t total_size = 0;
    auto keys = binary_set.GetKeys();

    for (const auto& key : keys) {
        total_size += sizeof(size_t) + key.size();
        total_size += sizeof(size_t);
        total_size += binary_set.Get(key).size;
    }

    Binary result;
    result.data = std::shared_ptr<int8_t[]>(new int8_t[total_size]);
    result.size = total_size;

    size_t offset = 0;

    for (const auto& key : keys) {
        size_t key_size = key.size();
        memcpy(result.data.get() + offset, &key_size, sizeof(size_t));
        offset += sizeof(size_t);
        memcpy(result.data.get() + offset, key.data(), key_size);
        offset += key_size;

        Binary binary = binary_set.Get(key);
        memcpy(result.data.get() + offset, &binary.size, sizeof(size_t));
        offset += sizeof(size_t);
        memcpy(result.data.get() + offset, binary.data.get(), binary.size);
        offset += binary.size;
    }

    return result;
}

BinarySet
binary_to_binaryset(const Binary binary) {
    /*
     * The Binary structure is serialized in the following layout:
     * | key_size (sizeof(size_t)) | key (of length key_size) | binary_size (sizeof(size_t)) | binary data (of length binary_size) |
     * Each key and its associated binary data are sequentially stored in the Binary object's data array,
     * and this information guides the deserialization process here.
    */
    BinarySet binary_set;
    size_t offset = 0;

    while (offset < binary.size) {
        size_t key_size;
        memcpy(&key_size, binary.data.get() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        std::string key(reinterpret_cast<const char*>(binary.data.get() + offset), key_size);
        offset += key_size;

        size_t binary_size;
        memcpy(&binary_size, binary.data.get() + offset, sizeof(size_t));
        offset += sizeof(size_t);

        Binary new_binary;
        new_binary.size = binary_size;
        new_binary.data = std::shared_ptr<int8_t[]>(new int8_t[binary_size]);
        memcpy(new_binary.data.get(), binary.data.get() + offset, binary_size);
        offset += binary_size;

        binary_set.Set(key, new_binary);
    }

    return binary_set;
}

ReaderSet
reader_to_readerset(std::shared_ptr<Reader> reader) {
    ReaderSet reader_set;
    size_t offset = 0;

    while (offset < reader->Size()) {
        size_t key_size;
        reader->Read(offset, sizeof(size_t), &key_size);
        offset += sizeof(size_t);
        std::shared_ptr<char[]> key_chars = std::shared_ptr<char[]>(new char[key_size]);
        reader->Read(offset, key_size, key_chars.get());
        std::string key(key_chars.get(), key_size);
        offset += key_size;

        size_t binary_size;
        reader->Read(offset, sizeof(size_t), &binary_size);
        offset += sizeof(size_t);

        auto new_reader = std::make_shared<SubReader>(reader, offset, binary_size);
        offset += binary_size;

        reader_set.Set(key, new_reader);
    }

    return reader_set;
}

template <typename T>
using Deque = std::deque<T, vsag::AllocatorWrapper<T>>;

constexpr static const char PART_OCTOTHORPE = '#';
std::vector<std::string>
split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    size_t start = 0;
    size_t end = str.find(delimiter);
    if (str.empty()) {
        throw std::runtime_error("fail to parse empty path");
    }

    while (end != std::string::npos) {
        std::string token = str.substr(start, end - start);
        if (token.empty()) {
            throw std::runtime_error("fail to parse path:" + str);
        }
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delimiter, start);
    }
    std::string last_token = str.substr(start);
    if (last_token.empty()) {
        throw std::runtime_error("fail to parse path:" + str);
    }
    tokens.push_back(str.substr(start, end - start));
    return tokens;
}

tl::expected<std::vector<int64_t>, Error>
Pyramid::Build(const DatasetPtr& base) {
    return this->Add(base);
}

tl::expected<std::vector<int64_t>, Error>
Pyramid::Add(const DatasetPtr& base) {
    auto path = base->GetPaths();
    int64_t data_num = base->GetNumElements();
    int64_t data_dim = base->GetDim();
    auto data_ids = base->GetIds();
    auto data_vectors = base->GetFloat32Vectors();
    for (int i = 0; i < data_num; ++i) {
        std::string current_path = path[i];
        auto path_slices = split(current_path, PART_SLASH);
        std::shared_ptr<IndexNode> node = try_get_node_with_init(indexes_, path_slices[0]);
        DatasetPtr single_data = Dataset::Make();
        single_data->Owner(false)
            ->NumElements(1)
            ->Dim(data_dim)
            ->Float32Vectors(data_vectors + data_dim * i)
            ->Ids(data_ids + i);
        for (int j = 1; j < path_slices.size(); ++j) {
            if (node->index) {
                node->index->Add(single_data);
            }
            node = try_get_node_with_init(node->children, path_slices[j]);
        }
        if (node->index == nullptr) {
            node->CreateIndex(pyramid_param_.index_builder);
        }
        node->index->Add(single_data);
    }
    data_num_ += data_num;
    return {};
}

tl::expected<DatasetPtr, Error>
Pyramid::knn_search(const DatasetPtr& query,
                    int64_t k,
                    const std::string& parameters,
                    const SearchFunc& search_func) const {
    auto path = query->GetPaths();  // TODO(inabao): provide different search modes.

    std::string current_path = path[0];
    auto path_slices = split(current_path, PART_SLASH);
    auto iter = indexes_.find(path_slices[0]);
    if (iter == indexes_.end()) {
        auto ret = Dataset::Make();
        ret->Dim(0)->NumElements(1);
        return ret;
    }
    std::shared_ptr<IndexNode> root = iter->second;
    for (int j = 1; j < path_slices.size(); ++j) {
        auto root_iter = root->children.find(path_slices[j]);
        if (root_iter == root->children.end()) {
            auto ret = Dataset::Make();
            ret->Dim(0)->NumElements(1);
            return ret;
        } else {
            root = root_iter->second;
        }
    }
    Deque<std::shared_ptr<IndexNode>> candidate_indexes(commom_param_.allocator_.get());

    std::priority_queue<std::pair<float, int64_t>> results;
    candidate_indexes.push_back(root);
    while (not candidate_indexes.empty()) {
        auto node = candidate_indexes.front();
        candidate_indexes.pop_front();
        if (node->index) {
            auto result = search_func(node->index);
            if (result.has_value()) {
                DatasetPtr r = result.value();
                for (int i = 0; i < r->GetDim(); ++i) {
                    results.emplace(r->GetDistances()[i], r->GetIds()[i]);
                }
            } else {
                auto error = result.error();
                LOG_ERROR_AND_RETURNS(error.type, error.message);
            }
        } else {
            for (const auto& item : node->children) {
                candidate_indexes.emplace_back(item.second);
            }
        }
        while (results.size() > k) {
            results.pop();
        }
    }

    // return result
    auto result = Dataset::Make();
    size_t target_size = results.size();
    if (results.size() == 0) {
        result->Dim(0)->NumElements(1);
        return result;
    }
    result->Dim(static_cast<int64_t>(target_size))
        ->NumElements(1)
        ->Owner(true, commom_param_.allocator_.get());
    int64_t* ids = (int64_t*)commom_param_.allocator_->Allocate(sizeof(int64_t) * target_size);
    result->Ids(ids);
    float* dists = (float*)commom_param_.allocator_->Allocate(sizeof(float) * target_size);
    result->Distances(dists);
    for (int64_t j = static_cast<int64_t>(results.size() - 1); j >= 0; --j) {
        if (j < target_size) {
            dists[j] = results.top().first;
            ids[j] = results.top().second;
        }
        results.pop();
    }
    return result;
}

tl::expected<BinarySet, Error>
Pyramid::Serialize() const {
    BinarySet binary_set;
    for (const auto& root_index : indexes_) {
        std::string path = root_index.first;
        std::vector<std::pair<std::string, std::shared_ptr<IndexNode>>> need_serialize_indexes;
        need_serialize_indexes.emplace_back(path, root_index.second);
        while (not need_serialize_indexes.empty()) {
            auto [current_path, index_node] = need_serialize_indexes.back();
            need_serialize_indexes.pop_back();
            if (index_node->index) {
                auto serialize_result = index_node->index->Serialize();
                if (not serialize_result.has_value()) {
                    return tl::unexpected(serialize_result.error());
                }
                binary_set.Set(current_path, binaryset_to_binary(serialize_result.value()));
            }
            for (const auto& sub_index_node : index_node->children) {
                need_serialize_indexes.emplace_back(
                    current_path + PART_OCTOTHORPE + sub_index_node.first, sub_index_node.second);
            }
        }
    }
    return binary_set;
}

tl::expected<void, Error>
Pyramid::Deserialize(const BinarySet& binary_set) {
    auto keys = binary_set.GetKeys();
    for (const auto& path : keys) {
        const auto& binary = binary_set.Get(path);
        auto path_slices = split(path, PART_OCTOTHORPE);
        std::shared_ptr<IndexNode> node = try_get_node_with_init(indexes_, path_slices[0]);
        for (int j = 1; j < path_slices.size(); ++j) {
            node = try_get_node_with_init(node->children, path_slices[j]);
        }
        node->CreateIndex(pyramid_param_.index_builder);
        node->index->Deserialize(binary_to_binaryset(binary));
    }
    return {};
}

tl::expected<void, Error>
Pyramid::Deserialize(const ReaderSet& reader_set) {
    auto keys = reader_set.GetKeys();
    for (const auto& path : keys) {
        const auto& reader = reader_set.Get(path);
        auto path_slices = split(path, PART_OCTOTHORPE);
        std::shared_ptr<IndexNode> node = try_get_node_with_init(indexes_, path_slices[0]);
        for (int j = 1; j < path_slices.size(); ++j) {
            node = try_get_node_with_init(node->children, path_slices[j]);
        }
        node->CreateIndex(pyramid_param_.index_builder);
        node->index->Deserialize(reader_to_readerset(reader));
    }
    return {};
}

int64_t
Pyramid::GetNumElements() const {
    return data_num_;
}

int64_t
Pyramid::GetMemoryUsage() const {
    return 0;
}

}  // namespace vsag
