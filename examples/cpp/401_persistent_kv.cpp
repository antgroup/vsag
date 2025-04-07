
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

#include <sys/stat.h>
#include <sys/types.h>
#include <vsag/vsag.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include "vsag/binaryset.h"

class LocalKvStore {
public:
    LocalKvStore(const std::string& path) : path_(path), meta_filename_(path + "/" + "_meta") {
        struct stat info;
        if (stat(path.c_str(), &info) != 0) {
            if (mkdir(path.c_str(), 0755) != 0) {
                std::cerr << "create example directory failed" << std::endl;
                abort();
            }
        }
    }

    void
    Put(const std::string& key, const std::string& value) {
        std::lock_guard<std::mutex> lock(mutex_);

        // write value
        std::ofstream value_file(path_ + "/" + key, std::ios::binary);
        if (not value_file.is_open()) {
            std::cerr << "open value file failed" << std::endl;
            abort();
        }
        value_file.write(value.c_str(), value.length());
        value_file.close();

        // update metadata if it's a new key
        auto keys = GetKeys();
        if (not keys.count(key)) {
            keys.insert(key);
            std::ofstream new_meta_file(meta_filename_);
            while (not keys.empty()) {
                auto key = *keys.begin();
                new_meta_file << key << std::endl;
                keys.erase(key);
            }
            new_meta_file.close();
        }
    }

    std::string
    Get(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto keys = GetKeys();
        if (not keys.count(key)) {
            std::cerr << "[" << key << "] not found" << std::endl;
            abort();
        }

        std::ifstream value_file(path_ + "/" + key, std::ios::binary | std::ios::ate);
        auto length = value_file.tellg();
        value_file.seekg(0, std::ios::beg);

        std::string content;
        content.resize(length);
        value_file.read(&content[0], length);
        value_file.close();

        return content;
    }

    std::unordered_set<std::string>
    GetKeys() {
        std::ifstream meta_file(meta_filename_);
        if (not meta_file.is_open()) {
            return {};
        }
        std::unordered_set<std::string> keys;
        std::string line;
        while (std::getline(meta_file, line)) {
            keys.insert(line);
        }
        meta_file.close();
        return keys;
    }

private:
    const std::string meta_filename_;
    const std::string path_;
    std::mutex mutex_;
};

int
main(int32_t argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    uint32_t num_vectors = 1000;
    uint32_t dim = 128;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (uint32_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (uint64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    /******************* Create an Index *****************/
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    vsag::IndexPtr index = nullptr;
    if (auto create_index = engine.CreateIndex("hnsw", index_paramesters);
        not create_index.has_value()) {
        std::cout << "create index failed: " << create_index.error().message << std::endl;
        abort();
    } else {
        index = *create_index;
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors)->Owner(false);
    if (auto build_index = index->Build(base); not build_index.has_value()) {
        std::cerr << "build index failed: " << build_index.error().message << std::endl;
        abort();
    }
    std::cout << "index contains vectors: " << index->GetNumElements() << std::endl;

    /******************* Save Index to KVStore *****************/
    auto serialize_result = index->Serialize();
    if (not serialize_result.has_value()) {
        std::cerr << serialize_result.error().message << std::endl;
        abort();
    }

    {
        LocalKvStore store("/tmp/vsag-persistent-kv-example");
        for (const auto& key : serialize_result.value().GetKeys()) {
            auto binary = serialize_result.value().Get(key);
            std::string value((const char*)binary.data.get(), binary.size);
            store.Put(key, value);
        }
    }

    /******************* Load Index from KVStore *****************/
    index = nullptr;
    if (auto create_index = engine.CreateIndex("hnsw", index_paramesters);
        not create_index.has_value()) {
        std::cout << "create index failed: " << create_index.error().message << std::endl;
        abort();
    } else {
        index = *create_index;
    }

    vsag::BinarySet bs;
    {
        LocalKvStore store("/tmp/vsag-persistent-kv-example");
        auto keys = store.GetKeys();
        for (const auto& key : keys) {
            auto value = store.Get(key);
            vsag::Binary binary;
            binary.data = std::shared_ptr<int8_t[]>(new int8_t[value.size()]);
            memcpy(binary.data.get(), value.data(), value.size());
            binary.size = value.size();
            bs.Set(key, binary);
        }
    }
    if (auto deserialize = index->Deserialize(bs); not deserialize.has_value()) {
        std::cerr << "load index failed: " << deserialize.error().message << std::endl;
        abort();
    }

    /******************* Search and Print Results *****************/
    auto topk = 10;
    auto query_vector = new float[dim];
    for (uint64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector)->Owner(false);
    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    if (auto knn_search = index->KnnSearch(query, topk, search_parameters);
        not knn_search.has_value()) {
        std::cerr << "search knn failed: " << knn_search.error().message << std::endl;
        abort();
    } else {
        auto result = *knn_search;
        for (int64_t i = 0; i < result->GetDim(); ++i) {
            std::cout << result->GetIds()[i] << " " << result->GetDistances()[i] << std::endl;
        }
    }

    return 0;
}
