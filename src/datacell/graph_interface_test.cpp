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

#include <fstream>
#include <random>

#include "impl/allocator/default_allocator.h"
#include "impl/allocator/safe_allocator.h"
#include "storage/serialization_template_test.h"
#include "unittest.h"

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

    // Test CheckIdExists
    SECTION("Test CheckIdExists") {
        for (auto& [key, value] : maps) {
            REQUIRE(this->graph_->CheckIdExists(key) == true);
        }
        InnerIdType non_exist_id = max_id + 1000;
        REQUIRE(this->graph_->CheckIdExists(non_exist_id) == false);
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
        test_serializion(*this->graph_, *other);

        REQUIRE(this->graph_->TotalCount() == other->TotalCount());
        REQUIRE(this->graph_->MaxCapacity() == other->MaxCapacity());
        REQUIRE(this->graph_->MaximumDegree() == other->MaximumDegree());
        for (auto& [key, value] : maps) {
            Vector<InnerIdType> neighbors(allocator.get());
            other->GetNeighbors(key, neighbors);
            REQUIRE(memcmp(neighbors.data(), value->data(), value->size() * sizeof(InnerIdType)) ==
                    0);
        }
    }

    if (test_delete) {
        SECTION("Delete") {
            std::unordered_set<InnerIdType> keys_to_delete;
            std::uniform_int_distribution<> dis(0, 1);
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

                    // test tombstone recovery
                    REQUIRE_THROWS(this->graph_->RecoverDeleteNeighborsById(item.first + 10000000));
                    if (dis(rng)) {
                        this->graph_->RecoverDeleteNeighborsById(item.first);
                        REQUIRE_THROWS(this->graph_->RecoverDeleteNeighborsById(item.first));
                        keys_to_delete.erase(item.first);
                    }
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

void
GraphInterfaceTest::ReverseEdgeTest(int count) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto max_degree = this->graph_->MaximumDegree();
    this->graph_->Resize(count);

    std::unordered_map<InnerIdType, std::shared_ptr<Vector<InnerIdType>>> maps =
        generate_graph(count, max_degree, allocator.get());

    for (auto& [key, value] : maps) {
        this->graph_->InsertNeighborsById(key, *value);
    }

    auto test_func = [&]() {
        std::unordered_map<InnerIdType, std::vector<InnerIdType>> expected_reverse(count);
        for (int i = 0; i < count; ++i) {
            Vector<InnerIdType> neighs(allocator.get());
            this->graph_->GetNeighbors(i, neighs);
            for (auto nei : neighs) {
                expected_reverse[nei].emplace_back(i);
            }
        }

        for (int i = 0; i < count; ++i) {
            Vector<InnerIdType> incoming(allocator.get());
            this->graph_->GetIncomingNeighbors(i, incoming);
            auto it = expected_reverse.find(i);
            if (it != expected_reverse.end()) {
                std::sort(incoming.begin(), incoming.end());
                auto expected = it->second;
                std::sort(expected.begin(), expected.end());
                REQUIRE(incoming.size() == expected.size());
                REQUIRE(memcmp(incoming.data(),
                               expected.data(),
                               incoming.size() * sizeof(InnerIdType)) == 0);
            } else {
                REQUIRE(incoming.empty());
            }
        }
    };

    SECTION("Test GetIncomingNeighbors") {
        test_func();
    }

    SECTION("Test Update Reverse Edges") {
        InnerIdType test_id = 0;
        Vector<InnerIdType> old_neighbors(allocator.get());
        this->graph_->GetNeighbors(test_id, old_neighbors);

        Vector<InnerIdType> new_neighbors(allocator.get());
        for (int i = 0; i < max_degree && i < count - 1; ++i) {
            new_neighbors.push_back(i + 10);
        }
        this->graph_->InsertNeighborsById(test_id, new_neighbors);
        Vector<InnerIdType> get_neighbors(allocator.get());
        this->graph_->GetNeighbors(test_id, get_neighbors);
        REQUIRE(memcmp(get_neighbors.data(),
                       new_neighbors.data(),
                       get_neighbors.size() * sizeof(InnerIdType)) == 0);
        test_func();
    }
}

void
GraphInterfaceTest::MoveTest(int count) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto max_degree = this->graph_->MaximumDegree();
    this->graph_->Resize(count * 2);

    std::unordered_map<InnerIdType, std::shared_ptr<Vector<InnerIdType>>> maps =
        generate_graph(count, max_degree, allocator.get());

    for (auto& [key, value] : maps) {
        this->graph_->InsertNeighborsById(key, *value);
    }
    constexpr int move_count = 20;
    for (int ii = 0; ii < move_count; ++ii) {
        InnerIdType from = count - 1 - ii;
        InnerIdType to = ii;

        this->graph_->Move(from, to);
    }
    count -= move_count;

    auto test_func = [&]() {
        std::unordered_map<InnerIdType, std::vector<InnerIdType>> expected_reverse(count);
        for (int i = 0; i < count; ++i) {
            Vector<InnerIdType> neighs(allocator.get());
            this->graph_->GetNeighbors(i, neighs);
            for (auto nei : neighs) {
                REQUIRE(nei < count);
                expected_reverse[nei].emplace_back(i);
            }
        }

        for (int i = 0; i < count; ++i) {
            Vector<InnerIdType> incoming(allocator.get());
            this->graph_->GetIncomingNeighbors(i, incoming);
            auto it = expected_reverse.find(i);
            if (it != expected_reverse.end()) {
                std::sort(incoming.begin(), incoming.end());
                auto expected = it->second;
                std::sort(expected.begin(), expected.end());
                REQUIRE(incoming.size() == expected.size());
                REQUIRE(memcmp(incoming.data(),
                               expected.data(),
                               incoming.size() * sizeof(InnerIdType)) == 0);
            } else {
                REQUIRE(incoming.empty());
            }
        }
    };

    test_func();
}
