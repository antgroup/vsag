
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

#include "sparse_term_computer.h"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "algorithm/sindi/sindi_parameter.h"
#include "impl/allocator/safe_allocator.h"

using namespace vsag;

TEST_CASE("SparseTermComputer Basic Test", "[ut][SparseTermComputer]") {
    // prepare data
    auto query_prune_ratio = 0.2;
    auto query_len = 10;
    SparseVector query_sv;
    query_sv.len_ = query_len;
    query_sv.ids_ = new uint32_t[query_sv.len_];
    query_sv.vals_ = new float[query_sv.len_];
    for (auto i = 0; i < query_len; i++) {
        query_sv.ids_[i] = i;
        query_sv.vals_[i] = i;
    }

    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    SINDISearchParameter search_params;
    search_params.query_prune_ratio = query_prune_ratio;
    auto computer = std::make_shared<SparseTermComputer>(query_sv, search_params, allocator.get());

    REQUIRE(computer->sorted_query_.size() == query_len);
    REQUIRE(computer->pruned_len_ == query_len * (1.0F - query_prune_ratio));
    for (auto i = 0; i < computer->sorted_query_.size(); i++) {
        auto id = query_len - i - 1;
        REQUIRE(computer->sorted_query_[i].first == id);
        REQUIRE(std::abs(computer->sorted_query_[i].second - (-1.0 * id)) < 1e-3);
    }

    // test term iterator
    for (auto i = 0; i < computer->pruned_len_; i++) {
        REQUIRE(computer->term_iterator_ == i);
        REQUIRE(computer->HasNextTerm() == true);
        REQUIRE(computer->NextTermIter() == i);
        REQUIRE(computer->GetTerm(i) == query_len - i - 1);
        REQUIRE(computer->term_iterator_ == i + 1);
    }
    REQUIRE(computer->HasNextTerm() == false);
    computer->ResetTerm();
    REQUIRE(computer->term_iterator_ == 0);

    // test scan
    auto test_term_it = 5;
    auto query_id = computer->sorted_query_[test_term_it].first;
    REQUIRE(query_id == query_len - test_term_it - 1);
    auto query_val = computer->sorted_query_[test_term_it].second;
    REQUIRE(std::abs(query_val - (-1.0 * query_id)) < 1e-3);
    std::vector<float> dists(10, 0);
    std::vector<uint32_t> term_ids = {0, 2, 4, 6, 8};
    std::vector<float> term_vals = {0, 2, 4, 6, 8};
    computer->ScanForAccumulate(
        test_term_it, term_ids.data(), term_vals.data(), term_ids.size(), dists.data());
    for (auto i = 0; i < term_ids.size(); i++) {
        auto id = term_ids[i];
        REQUIRE(dists[id] == term_vals[i] * query_val);
    }

    // clean
    delete[] query_sv.vals_;
    delete[] query_sv.ids_;
}
