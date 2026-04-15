
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use hgraph_ file except in compliance with the License.
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

#include <memory>

#include "algorithm/hgraph.h"
#include "algorithm/inner_index_interface.h"
#include "utils/pointer_define.h"
#include "vsag/allocator.h"
namespace vsag {

DEFINE_POINTER(AnalyzerBase);

/**
 * @brief Base class for index analyzers that provide diagnostic and performance analysis.
 *
 * This abstract class defines the interface for analyzing vector indexes,
 * including statistics collection and search-based analysis.
 */
class AnalyzerBase {
public:
    /**
     * @brief Constructs an analyzer with the given allocator and total element count.
     *
     * @param allocator Pointer to the allocator for memory management.
     * @param total_count Total number of elements in the index to be analyzed.
     */
    AnalyzerBase(Allocator* allocator, uint32_t total_count)
        : allocator_(allocator), total_count_(total_count) {
    }

    /**
     * @brief Gets statistical information about the analyzed index.
     *
     * @return JsonType containing various statistics and metrics.
     */
    virtual JsonType
    GetStats() = 0;

    virtual ~AnalyzerBase() = default;

    /**
     * @brief Analyzes the index by performing searches with the given request.
     *
     * @param request The search request parameters for analysis.
     * @return JsonType containing analysis results.
     */
    virtual JsonType
    AnalyzeIndexBySearch(const SearchRequest& request) = 0;

protected:
    /// Allocator for memory management operations.
    Allocator* allocator_;
    /// Total number of elements in the index.
    uint32_t total_count_;
    /// Dimensionality of vectors in the index.
    uint32_t dim_;
};

/**
 * @brief Parameters for creating an analyzer instance.
 *
 * This structure contains configuration options for analyzer initialization,
 * including search parameters and sampling sizes.
 */
struct AnalyzerParam {
public:
    /**
     * @brief Constructs analyzer parameters with the given allocator.
     *
     * @param allocator Pointer to the allocator for memory management.
     */
    AnalyzerParam(Allocator* allocator) : allocator(allocator) {
    }

public:
    /// Allocator for memory management operations.
    Allocator* allocator;
    /// Number of top results to consider in analysis.
    int64_t topk{100};
    /// Number of base vectors to sample for analysis.
    uint64_t base_sample_size{10};
    /// Search parameters string for analysis queries.
    std::string search_params;
};

/**
 * @brief Creates an analyzer instance for the given index.
 *
 * @param index Pointer to the inner index interface to be analyzed.
 * @param param Analyzer parameters for configuration.
 * @return AnalyzerBasePtr pointing to the created analyzer instance.
 */
AnalyzerBasePtr
CreateAnalyzer(const InnerIndexInterface* index, const AnalyzerParam& param);

}  // namespace vsag
