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

#include <vsag/vsag.h>

#include <iostream>
#include <random>
#include <vector>

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    int64_t dim = 64;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> datas(num_vectors * dim);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->Owner(false);

    /******************* Create HGraph Index *****************/
    std::string build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 64,
        "index_param": {
            "base_quantization_type": "fp32",
            "max_degree": 16,
            "ef_construction": 100,
            "use_reorder": true,
            "precise_quantization_type": "fp32"
        }
    })";

    auto created = vsag::Factory::CreateIndex("hgraph", build_parameters);
    if (!created.has_value()) {
        std::cerr << "CreateIndex failed: " << created.error().message << std::endl;
        return 1;
    }
    auto index = created.value();
    auto built = index->Build(base);
    if (!built.has_value()) {
        std::cerr << "Build failed: " << built.error().message << std::endl;
        return 1;
    }
    if (!built.value().empty()) {
        std::cerr << "Build rejected some labels" << std::endl;
        return 1;
    }

    /******************* Prepare Query *****************/
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(datas.data())->Owner(false);

    std::string search_params = R"({"hgraph": {"ef_search": 32}})";

    /******************* KNN Search (no reasoning) *****************/
    auto result = index->KnnSearch(query, 3, search_params);
    if (!result.has_value()) {
        std::cerr << "KnnSearch failed: " << result.error().message << std::endl;
        return 1;
    }
    // For search results, GetDim() is the neighbor count, not GetNumElements().
    const auto print_results = [](const vsag::DatasetPtr& results) {
        std::cout << "Neighbor count: " << results->GetDim() << std::endl;
        for (int64_t i = 0; i < results->GetDim(); ++i) {
            std::cout << "  rank " << i << ": id=" << results->GetIds()[i]
                      << ", distance=" << results->GetDistances()[i] << std::endl;
        }
    };
    std::cout << "=== KNN Search (no reasoning) ===" << std::endl;
    print_results(result.value());

    /******************* SearchWithRequest (with reasoning) *****************/
    vsag::SearchRequest req;
    req.topk_ = 3;
    req.query_ = query;
    req.params_str_ = search_params;
    req.expected_labels_ = {0, 3, 10};

    auto reasoning_result = index->SearchWithRequest(req);
    if (!reasoning_result.has_value()) {
        std::cerr << "SearchWithRequest failed: " << reasoning_result.error().message << std::endl;
        return 1;
    }
    std::cout << "\n=== SearchWithRequest (with reasoning) ===" << std::endl;
    print_results(reasoning_result.value());
    if (result.value()->GetDim() != 3 || reasoning_result.value()->GetDim() != 3) {
        std::cerr << "Expected three neighbors from both searches" << std::endl;
        return 1;
    }
    for (int64_t i = 0; i < result.value()->GetDim(); ++i) {
        if (result.value()->GetIds()[i] != reasoning_result.value()->GetIds()[i] ||
            result.value()->GetDistances()[i] != reasoning_result.value()->GetDistances()[i]) {
            std::cerr << "Reasoning changed search results" << std::endl;
            return 1;
        }
    }
    auto reasoning = reasoning_result.value()->GetReasoning();
    if (reasoning.empty()) {
        std::cerr << "Expected a nonempty reasoning report" << std::endl;
        return 1;
    }
    std::cout << reasoning << std::endl;

    return 0;
}

/*
 * Annotated report excerpt (not a JSON Schema or a fixed expected result).
 * Values and traversal order can vary between runs. The second missed target
 * is omitted here for brevity. The current HGraph reasoning distance counter
 * is not populated; zero does not mean the search computed no distances.
 * supports_range describes reasoning support, not the index's RangeSearch API.
 *
 * {
 *   "expected_analysis": {
 *     "summary": "1/3 expected labels found, 2 missed",
 *     "missed_targets": [
 *       {
 *         "label":              10,         // int64: user-provided label
 *         "inner_id":           10,         // int64: internal index id
 *         "diagnosis":          "not_reachable",   // string: why this target was missed
 *                                            //   one of: "success", "not_reachable",
 *                                            //           "filter_rejected",
 *                                            //           "quantization_error",
 *                                            //           "ef_too_small",
 *                                            //           "reorder_evicted", "unknown"
 *         "was_visited":        false,        // bool: whether the target was encountered
 *         "visited_at_hop":     -1,          // int64: hop number when first visited
 *         "was_evicted":        false,       // bool: whether evicted from the candidate pool
 *         "filter_rejected":    false,       // bool: whether the filter rejected this target
 *         "quantized_distance": 0.0,     // float64: distance computed by the approximate phase
 *         "true_distance":      11.0995,     // float64: exact distance (only available with reorder)
 *         "reorder_evicted":    false        // bool: whether evicted during reorder phase
 *       }
 *     ]
 *   },
 *   "meta": {
 *     "schema_version":          1,           // int: report schema version
 *     "status":                  "ok",        // string: "ok", "skipped_range_search",
 *                                             //         "unsupported_by_index", "empty_index"
 *     "index_type":              "HGraph",    // string: index name (e.g. "HGraph", "IVF",
 *                                             //         "SINDI_V2", "BruteForce", "Pyramid")
 *     "search_mode":             "knn",       // string: "knn" or "range"
 *     "topk":                    3,           // int64: requested top-k (knn mode),
 *                                             //         -1 for range search
 *     "use_reorder":             true,       // bool: whether the index used a reorder step
 *     "filter_active":           false,       // bool: whether a filter was applied
 *     "termination_reason":      "lower_bound_reached",
 *                                             // string: why the search stopped
 *                                             //   one of: "none", "lower_bound_reached",
 *                                             //           "hops_limit_reached", "timeout"
 *     "total_hops":              37,          // int64: total graph hops performed
 *     "total_distance_computations": 0,       // int64: total distance computations
 *     "supports_range":          false,       // bool: whether range reasoning is supported
 *     "available_diagnoses": [               // [string]: diagnosis types this index can produce
 *       "success", "not_reachable", "filter_rejected",
 *       "quantization_error", "ef_too_small", "reorder_evicted", "unknown"
 *     ],
 *     "available_events": [                  // [string]: event types this index can record
 *       "visit", "eviction", "filter_reject", "reorder", "reorder_eviction"
 *     ]
 *   }
 * }
 */