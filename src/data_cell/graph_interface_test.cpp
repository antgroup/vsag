
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

#include "default_allocator.h"
#include "fixtures.h"
#include "safe_allocator.h"

using namespace vsag;

void
GraphInterfaceTest::BasicTest(uint64_t max_id, uint64_t count, const GraphInterfacePtr& other) {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    auto max_degree = this->graph_->MaximumDegree();
    this->graph_->Resize(max_id);

    auto generate_graph = [&]() {
        UnorderedMap<InnerIdType, std::shared_ptr<Vector<InnerIdType>>> cur_map(allocator.get());
        for (auto i = 0; i < count; ++i) {
            auto length = random() % max_degree + 1;
            auto ids = std::make_shared<Vector<InnerIdType>>(length, allocator.get());
            for (auto& id : *ids) {
                id = random() % max_id;
            }
            auto cur_id = random() % max_id;
            cur_map[cur_id] = ids;
        }
        if (require_sorted_) {
            for (auto& [key, value] : cur_map) {
                std::sort(value->begin(), value->end());
            }
        }
        return cur_map;
    };

    UnorderedMap<InnerIdType, std::shared_ptr<Vector<InnerIdType>>> maps = generate_graph();
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

    maps = generate_graph();
    for (auto& [key, value] : maps) {
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
