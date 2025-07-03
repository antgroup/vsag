
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

#include "graph_interface_test.h"

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <random>

#include "fixtures.h"
#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

void
GraphInterfaceTest::BasicTest(uint64_t max_id,
                              uint64_t count,
                              const GraphInterfacePtr& other,
                              bool test_delete) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto max_degree = this->graph_->MaximumDegree();
    this->graph_->Resize(max_id);
    UnorderedMap<InnerIdType, std::shared_ptr<Vector<InnerIdType>>> maps(allocator.get());
    std::unordered_set<InnerIdType> unique_keys;
    while (unique_keys.size() < count) {
        InnerIdType new_key = random() % max_id;
        unique_keys.insert(new_key);
    }

    std::vector<InnerIdType> keys(unique_keys.begin(), unique_keys.end());
    for (auto key : keys) {
        maps[key] = std::make_shared<Vector<InnerIdType>>(allocator.get());
    }

    std::random_device rd;
    std::mt19937 rng(rd());

    for (auto& pair : maps) {
        auto& vec_ptr = pair.second;
        int max_possible_length = keys.size();
        int length = random() % (max_degree - 1) + 2;
        length = std::min(length, max_possible_length);
        std::vector<InnerIdType> temp_keys = keys;
        std::shuffle(temp_keys.begin(), temp_keys.end(), rng);

        vec_ptr->resize(length);
        for (int i = 0; i < length; ++i) {
            (*vec_ptr)[i] = temp_keys[i];
        }
    }

    if (require_sorted_) {
        for (auto& [key, value] : maps) {
            std::sort(value->begin(), value->end());
        }
    }

    for (auto& [key, value] : maps) {
        this->graph_->InsertNeighborsById(key, *value);
    }

    // Test GetNeighborSize
    SECTION("Test GetNeighborSize") {
        for (auto& [key, value] : maps) {
            REQUIRE(this->graph_->GetNeighborSize(key) == value->size());
        }
    }

    // Test GetNeighbors
    SECTION("Test GetNeighbors") {
        for (auto& [key, value] : maps) {
            Vector<InnerIdType> neighbors(allocator.get());
            this->graph_->GetNeighbors(key, neighbors);
            REQUIRE(memcmp(neighbors.data(), value->data(), value->size() * sizeof(InnerIdType)) ==
                    0);
        }
    }

    // Test Others
    SECTION("Test Others") {
        REQUIRE(this->graph_->MaxCapacity() >= this->graph_->TotalCount());
        REQUIRE(this->graph_->MaximumDegree() == max_degree);

        this->graph_->SetTotalCount(this->graph_->TotalCount());
        this->graph_->SetMaxCapacity(this->graph_->MaxCapacity());
        this->graph_->SetMaximumDegree(this->graph_->MaximumDegree());
    }

    SECTION("Serialize & Deserialize") {
        fixtures::TempDir dir("");
        auto path = dir.GenerateRandomFile();
        std::ofstream outfile(path.c_str(), std::ios::binary);
        IOStreamWriter writer(outfile);
        this->graph_->Serialize(writer);
        outfile.close();

        std::ifstream infile(path.c_str(), std::ios::binary);
        IOStreamReader reader(infile);
        other->Deserialize(reader);

        REQUIRE(this->graph_->TotalCount() == other->TotalCount());
        REQUIRE(this->graph_->MaxCapacity() == other->MaxCapacity());
        REQUIRE(this->graph_->MaximumDegree() == other->MaximumDegree());
        for (auto& [key, value] : maps) {
            Vector<InnerIdType> neighbors(allocator.get());
            other->GetNeighbors(key, neighbors);
            REQUIRE(memcmp(neighbors.data(), value->data(), value->size() * sizeof(InnerIdType)) ==
                    0);
        }

        infile.close();
    }

    if (test_delete) {
        SECTION("Delete") {
            std::unordered_set<InnerIdType> keys_to_delete;
            for (const auto& item : maps) {
                if (keys_to_delete.size() > count / 2) {
                    Vector<InnerIdType> neighbors(allocator.get());
                    this->graph_->GetNeighbors(item.first, neighbors);
                    for (const auto& neighbor_id : neighbors) {
                        REQUIRE(keys_to_delete.count(neighbor_id) == 0);
                    }
                } else {
                    this->graph_->DeleteNeighborsById(item.first);
                    keys_to_delete.insert(item.first);
                }
            }
            for (const auto& key : keys_to_delete) {
                this->graph_->InsertNeighborsById(key, *maps[key]);
            }
            for (const auto& [key, value] : maps) {
                if (keys_to_delete.find(key) == keys_to_delete.end()) {
                    Vector<InnerIdType> neighbors(allocator.get());
                    this->graph_->GetNeighbors(key, neighbors);
                    for (const auto& neighbor_id : neighbors) {
                        REQUIRE(keys_to_delete.count(neighbor_id) == 0);
                    }
                    this->graph_->InsertNeighborsById(key, *value);
                    this->graph_->GetNeighbors(key, neighbors);
                    REQUIRE(neighbors.size() == value->size());
                    REQUIRE(memcmp(neighbors.data(),
                                   value->data(),
                                   value->size() * sizeof(InnerIdType)) == 0);
                }
            }
        }
    }

    for (auto& [key, value] : maps) {
        value->resize(value->size() / 2);
        this->graph_->InsertNeighborsById(key, *value);
    }
    SECTION("Test Update Graph") {
        for (auto& [key, value] : maps) {
            REQUIRE(this->graph_->GetNeighborSize(key) == value->size());
        }
        for (auto& [key, value] : maps) {
            Vector<InnerIdType> neighbors(allocator.get());
            this->graph_->GetNeighbors(key, neighbors);
            REQUIRE(memcmp(neighbors.data(), value->data(), value->size() * sizeof(InnerIdType)) ==
                    0);
        }
    }
}

std::unordered_map<InnerIdType, std::shared_ptr<Vector<InnerIdType>>>
generate_graph(int count, int max_degree, Allocator* allocator) {
    std::unordered_map<InnerIdType, std::shared_ptr<Vector<InnerIdType>>> maps;
    for (int i = 0; i < count; ++i) {
        std::unordered_set<InnerIdType> unique_ids;
        unique_ids.insert(i);
        maps[i] = std::make_shared<Vector<InnerIdType>>(allocator);
        while (unique_ids.size() < max_degree + 1) {
            InnerIdType new_neighbor = random() % count;
            auto iter = unique_ids.insert(new_neighbor);
            if (iter.second) {
                maps[i]->push_back(new_neighbor);
            }
        }
    }
    return maps;
}

void
GraphInterfaceTest::MergeTest(GraphInterfacePtr& other, int count) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto max_degree = this->graph_->MaximumDegree();

    std::random_device rd;
    std::unordered_map<InnerIdType, std::shared_ptr<Vector<InnerIdType>>> maps =
        generate_graph(count, max_degree, allocator.get());
    std::unordered_map<InnerIdType, std::shared_ptr<Vector<InnerIdType>>> other_maps =
        generate_graph(count, max_degree, allocator.get());
    this->graph_->Resize(count);
    other->Resize(count);

    for (auto& [key, value] : maps) {
        this->graph_->InsertNeighborsById(key, *value);
    }

    for (auto& [key, value] : other_maps) {
        other->InsertNeighborsById(key, *value);
    }

    for (const auto& item : other_maps) {
        maps[item.first + count] = std::make_shared<Vector<InnerIdType>>(allocator.get());
        for (auto& nb_id : (*item.second)) {
            nb_id += count;
        }
        maps[item.first + count]->swap(*item.second);
    }

    this->graph_->Resize(2 * count);
    this->graph_->MergeOther(other, count);

    // Test GetNeighborSize
    SECTION("Test GetNeighborSize") {
        for (auto& [key, value] : maps) {
            REQUIRE(this->graph_->GetNeighborSize(key) == value->size());
        }
    }

    // Test GetNeighbors
    SECTION("Test GetNeighbors") {
        for (auto& [key, value] : maps) {
            Vector<InnerIdType> neighbors(allocator.get());
            this->graph_->GetNeighbors(key, neighbors);
            REQUIRE(memcmp(neighbors.data(), value->data(), value->size() * sizeof(InnerIdType)) ==
                    0);
        }
    }
}
