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

#include "reverse_edge.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"

namespace vsag {

TEST_CASE("ReverseEdge Basic Operations", "[ut][ReverseEdge]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    ReverseEdge reverse_edge(allocator.get());

    SECTION("Add and Get Reverse Edge") {
        reverse_edge.AddReverseEdge(1, 2);
        reverse_edge.AddReverseEdge(3, 2);

        Vector<InnerIdType> neighbors(allocator.get());
        reverse_edge.GetIncomingNeighbors(2, neighbors);

        REQUIRE(neighbors.size() == 2);
        REQUIRE(std::find(neighbors.begin(), neighbors.end(), 1) != neighbors.end());
        REQUIRE(std::find(neighbors.begin(), neighbors.end(), 3) != neighbors.end());
    }

    SECTION("Duplicate Add Should Be Idempotent") {
        reverse_edge.AddReverseEdge(1, 2);
        reverse_edge.AddReverseEdge(1, 2);
        reverse_edge.AddReverseEdge(1, 2);

        Vector<InnerIdType> neighbors(allocator.get());
        reverse_edge.GetIncomingNeighbors(2, neighbors);

        REQUIRE(neighbors.size() == 1);
        REQUIRE(neighbors[0] == 1);
    }

    SECTION("Remove Reverse Edge") {
        reverse_edge.AddReverseEdge(1, 2);
        reverse_edge.AddReverseEdge(3, 2);

        reverse_edge.RemoveReverseEdge(1, 2);

        Vector<InnerIdType> neighbors(allocator.get());
        reverse_edge.GetIncomingNeighbors(2, neighbors);

        REQUIRE(neighbors.size() == 1);
        REQUIRE(neighbors[0] == 3);
    }

    SECTION("Remove Non-Existent Edge Should Not Crash") {
        reverse_edge.AddReverseEdge(1, 2);
        reverse_edge.RemoveReverseEdge(3, 2);
        reverse_edge.RemoveReverseEdge(1, 3);

        Vector<InnerIdType> neighbors(allocator.get());
        reverse_edge.GetIncomingNeighbors(2, neighbors);

        REQUIRE(neighbors.size() == 1);
        REQUIRE(neighbors[0] == 1);
    }

    SECTION("Get Incoming Neighbors for Non-Existent Node") {
        Vector<InnerIdType> neighbors(allocator.get());
        reverse_edge.GetIncomingNeighbors(999, neighbors);

        REQUIRE(neighbors.empty());
    }

    SECTION("Clear Incoming Neighbors") {
        reverse_edge.AddReverseEdge(1, 2);
        reverse_edge.AddReverseEdge(3, 2);
        reverse_edge.AddReverseEdge(4, 5);

        reverse_edge.ClearIncomingNeighbors(2);

        Vector<InnerIdType> neighbors(allocator.get());
        reverse_edge.GetIncomingNeighbors(2, neighbors);
        REQUIRE(neighbors.empty());

        reverse_edge.GetIncomingNeighbors(5, neighbors);
        REQUIRE(neighbors.size() == 1);
        REQUIRE(neighbors[0] == 4);
    }

    SECTION("Clear All") {
        reverse_edge.AddReverseEdge(1, 2);
        reverse_edge.AddReverseEdge(3, 2);
        reverse_edge.AddReverseEdge(4, 5);

        reverse_edge.Clear();

        Vector<InnerIdType> neighbors(allocator.get());
        reverse_edge.GetIncomingNeighbors(2, neighbors);
        REQUIRE(neighbors.empty());

        reverse_edge.GetIncomingNeighbors(5, neighbors);
        REQUIRE(neighbors.empty());
    }

    SECTION("Multiple Edges") {
        reverse_edge.AddReverseEdge(1, 2);
        reverse_edge.AddReverseEdge(3, 2);
        reverse_edge.AddReverseEdge(5, 2);

        reverse_edge.AddReverseEdge(1, 4);
        reverse_edge.AddReverseEdge(3, 4);

        Vector<InnerIdType> neighbors_2(allocator.get());
        reverse_edge.GetIncomingNeighbors(2, neighbors_2);
        REQUIRE(neighbors_2.size() == 3);

        Vector<InnerIdType> neighbors_4(allocator.get());
        reverse_edge.GetIncomingNeighbors(4, neighbors_4);
        REQUIRE(neighbors_4.size() == 2);

        reverse_edge.RemoveReverseEdge(1, 2);
        reverse_edge.GetIncomingNeighbors(2, neighbors_2);
        REQUIRE(neighbors_2.size() == 2);
    }

    SECTION("Get Memory Usage") {
        reverse_edge.AddReverseEdge(1, 2);
        reverse_edge.AddReverseEdge(3, 2);

        int64_t usage = reverse_edge.GetMemoryUsage();
        REQUIRE(usage > 0);
        REQUIRE(usage > sizeof(ReverseEdge));
    }
}

TEST_CASE("ReverseEdge Thread Safety", "[ut][ReverseEdge]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    ReverseEdge reverse_edge(allocator.get());

    const int num_threads = 4;
    const int num_iterations = 100;

    SECTION("Concurrent Add and Get") {
        std::vector<std::thread> threads;

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&reverse_edge, t, num_iterations]() {
                for (int i = 0; i < num_iterations; ++i) {
                    reverse_edge.AddReverseEdge(t * num_iterations + i, 0);
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        Vector<InnerIdType> neighbors(allocator.get());
        reverse_edge.GetIncomingNeighbors(0, neighbors);
        REQUIRE(neighbors.size() == static_cast<size_t>(num_threads * num_iterations));
    }

    SECTION("Concurrent Add and Remove") {
        std::vector<std::thread> threads;

        for (int t = 0; t < num_threads; ++t) {
            threads.emplace_back([&reverse_edge, t, num_iterations]() {
                for (int i = 0; i < num_iterations; ++i) {
                    int id = t * num_iterations + i;
                    reverse_edge.AddReverseEdge(id, 0);
                    if (i % 2 == 0) {
                        reverse_edge.RemoveReverseEdge(id, 0);
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        Vector<InnerIdType> neighbors(allocator.get());
        reverse_edge.GetIncomingNeighbors(0, neighbors);
        REQUIRE(neighbors.size() <= static_cast<size_t>(num_threads * num_iterations));
    }
}

}  // namespace vsag
