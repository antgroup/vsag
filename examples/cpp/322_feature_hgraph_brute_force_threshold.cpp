
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

// Demonstrates the HGraph search-time parameter `brute_force_threshold`.
//
// Background
// ----------
// When a filter is supplied to HGraph's KnnSearch / RangeSearch, the graph
// beam search has to expand more and more candidates to fill `ef_search` with
// ids that actually pass the predicate. Under a very selective filter (only a
// tiny fraction of ids survive) the beam quickly runs out of useful neighbours
// and recall drops.
//
// `brute_force_threshold` lets HGraph automatically skip the graph traversal
// and run an exact scan over the surviving ids whenever the filter's
// `ValidRatio()` falls below the configured threshold. The brute-force scan
// uses the most precise flatten storage available (raw vectors > precise
// reorder codes > base quantized codes) and skips the post-search reorder
// because it already computed precise distances.
//
// This example builds a small HGraph index, defines a `Filter` whose
// `ValidRatio()` is 0.02, and runs the same query three ways:
//   1. graph search, no brute-force fallback (baseline)
//   2. brute_force_threshold = 0.01 — filter ratio (0.02) is ABOVE the
//      threshold, so the fallback is NOT triggered (behaves like baseline)
//   3. brute_force_threshold = 0.05 — filter ratio (0.02) is at or BELOW the
//      threshold, so HGraph runs the exact scan
//
// Run (1) and (3) should produce identical, exact results (mode 3 is by
// definition an exact scan over the surviving ids).

#include <vsag/vsag.h>

#include <iostream>
#include <random>
#include <vector>

namespace {

/// A toy filter that keeps only ids where `id % modulus == residue`.
/// It returns a fixed `ValidRatio()` hint (e.g. 1/modulus) so HGraph can
/// compare it against `brute_force_threshold`.
class ModuloFilter : public vsag::Filter {
public:
    ModuloFilter(int64_t modulus, int64_t residue, float valid_ratio)
        : modulus_(modulus), residue_(residue), valid_ratio_(valid_ratio) {
    }

    bool
    CheckValid(int64_t id) const override {
        return (id % modulus_) == residue_;
    }

    float
    ValidRatio() const override {
        return valid_ratio_;
    }

private:
    int64_t modulus_;
    int64_t residue_;
    float valid_ratio_;
};

void
print_results(const std::string& label, const vsag::DatasetPtr& result) {
    std::cout << label << " (top " << result->GetDim() << "):\n";
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << "  id=" << result->GetIds()[i] << "  dist=" << result->GetDistances()[i]
                  << "\n";
    }
}

}  // namespace

int
main(int /*argc*/, char** /*argv*/) {
    vsag::init();

    /******************* Prepare base dataset *****************/
    constexpr int64_t num_vectors = 10000;
    constexpr int64_t dim = 64;
    constexpr int64_t modulus = 50;  // filter keeps 1 in every 50 ids -> ratio = 0.02
    constexpr int64_t residue = 7;
    constexpr int64_t topk = 5;
    constexpr float filter_valid_ratio = 1.0F / static_cast<float>(modulus);

    std::vector<int64_t> ids(num_vectors);
    std::vector<float> data(num_vectors * dim);
    std::mt19937 rng(20260528);
    std::uniform_real_distribution<float> distrib(-1.0F, 1.0F);
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < num_vectors * dim; ++i) {
        data[i] = distrib(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(data.data())
        ->Owner(false);

    /******************* Build HGraph index *****************/
    const std::string build_params = R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 32,
            "ef_construction": 200
        }
    })";
    auto index = vsag::Factory::CreateIndex("hgraph", build_params).value();
    if (auto build_res = index->Build(base); not build_res.has_value()) {
        std::cerr << "Build failed: " << build_res.error().message << std::endl;
        return -1;
    }

    /******************* Prepare query and filter *****************/
    std::vector<float> query_vec(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vec[i] = distrib(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vec.data())->Owner(false);

    // ValidRatio is the user's selectivity hint to HGraph; it should reflect
    // the fraction of ids that actually pass `CheckValid`.
    auto filter = std::make_shared<ModuloFilter>(modulus, residue, filter_valid_ratio);
    std::cout << "Filter keeps approximately " << (filter_valid_ratio * 100.0F) << "% of ids\n\n";

    /******************* 1. Baseline: graph search, no fallback *****************/
    {
        const std::string params = R"({"hgraph": {"ef_search": 200}})";
        auto res = index->KnnSearch(query, topk, params, filter);
        if (not res.has_value()) {
            std::cerr << "Search 1 failed: " << res.error().message << std::endl;
            return -1;
        }
        print_results("[1] graph search (brute_force_threshold unset)", res.value());
    }

    /******************* 2. Threshold below the filter's valid ratio:
                              fallback does NOT trigger             *****************/
    {
        const std::string params =
            R"({"hgraph": {"ef_search": 200, "brute_force_threshold": 0.01}})";
        auto res = index->KnnSearch(query, topk, params, filter);
        if (not res.has_value()) {
            std::cerr << "Search 2 failed: " << res.error().message << std::endl;
            return -1;
        }
        print_results("\n[2] graph search (threshold 0.01 < valid_ratio 0.02 -> graph path)",
                      res.value());
    }

    /******************* 3. Threshold at or above the filter's valid ratio:
                              fallback triggers, exact scan         *****************/
    {
        const std::string params =
            R"({"hgraph": {"ef_search": 200, "brute_force_threshold": 0.05}})";
        auto res = index->KnnSearch(query, topk, params, filter);
        if (not res.has_value()) {
            std::cerr << "Search 3 failed: " << res.error().message << std::endl;
            return -1;
        }
        print_results(
            "\n[3] brute-force fallback (threshold 0.05 >= valid_ratio 0.02 -> exact scan)",
            res.value());
    }

    /******************* Reference: hand-rolled exact scan *****************/
    std::vector<std::pair<float, int64_t>> reference;
    for (int64_t i = 0; i < num_vectors; ++i) {
        if ((ids[i] % modulus) != residue) {
            continue;
        }
        float d = 0.0F;
        for (int64_t j = 0; j < dim; ++j) {
            float diff = data[i * dim + j] - query_vec[j];
            d += diff * diff;
        }
        reference.emplace_back(d, ids[i]);
    }
    std::sort(reference.begin(), reference.end());
    std::cout << "\n[ref] hand-rolled exact scan over filtered ids:\n";
    for (int64_t k = 0; k < topk; ++k) {
        std::cout << "  id=" << reference[k].second << "  dist=" << reference[k].first << "\n";
    }
    std::cout << "\nResult set (3) above must match this reference; results (1) and (2)\n"
                 "are approximate and may differ when ef_search is too small relative to\n"
                 "the filter's selectivity.\n";

    return 0;
}
