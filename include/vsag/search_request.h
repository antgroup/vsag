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

#pragma once
#include <cstdint>
#include <string>

#include "vsag/dataset.h"
#include "vsag/filter.h"

namespace vsag {
enum class SearchMode {
    KNN_SEARCH = 1,
    RANGE_SEARCH = 2,
};

class SearchRequest {
public:
    // basic params
    /** 
     * @brief Query dataset containing the vector to search for
     * @details This DatasetPtr holds the query vector used for similarity search. 
     *          Only one query vector is allowed.
     */
    DatasetPtr query_{nullptr};

    /**
     * @brief Search mode determining the type of similarity search to perform
     * @details Specifies whether to perform K-Nearest Neighbors (KNN) search or range-based search.
     *          - KNN_SEARCH: Returns the top-k most similar vectors
     *          - RANGE_SEARCH: Returns all vectors within a specified distance/radius
     */
    SearchMode mode_{SearchMode::KNN_SEARCH};

    /**
     * @brief Number of nearest neighbors to return in KNN search mode
     * @details Only applicable when mode_ is set to KNN_SEARCH. 
     *          Defines how many of the most similar vectors should be returned.
     *          Default value is 10, must be a positive integer.
     */
    int64_t topk_{10};

    /**
     * @brief Search radius for range-based search mode
     * @details Only used when mode_ is set to RANGE_SEARCH.
     *          Defines the maximum distance threshold from the query vector.
     *          All vectors within this radius will be allowed.
     *          The result size must be smaller than the limited_size_(when limited_size_ is positive).
     *          Default value is 0.5, must be a non-negative float.
     */
    float radius_{0.5F};

    /**
     * @brief Limited size for search results
     * @details Only used when mode_ is set to RANGE_SEARCH.
     *          Defines the maximum number of results to return.
     *          Default value is -1, which means no limit.
     */
    int64_t limited_size_{-1};

    /**
     * @brief Additional search parameters as a JSON string
     * @details Contains algorithm-specific parameters in JSON format.
     *          Used to pass fine-tuned configuration options to the underlying search algorithm.
     *          Examples: ef_search, etc.
     */
    std::string params_str_{};

    // for attribute filter
    /**
     * @brief Flag to enable attribute-based filtering during search
     * @details When set to true, enables filtering of search results based on vector attributes.
     *          Requires attribute_filter_str_ to contain valid filter criteria.
     *          Default is false (no attribute filtering applied).
     */
    bool enable_attribute_filter_{false};

    /**
     * @brief Attribute filter criteria as a SQL string
     * @details Contains the filtering conditions for vector attributes in SQL format.
     *          Only used when enable_attribute_filter_ is set to true.
     *          Format depends on the attribute schema defined in the dataset.
     *          Examples: 
     *              1. "category = 'electronics' AND price != 1000",
     *              2. "multi_in(category, ['electronics', 'clothing']) AND multi_notin(color, ['red', 'blue'])",
     */
    std::string attribute_filter_str_{};

    // for callback filter
    /**
     * @brief Flag to enable custom callback-based filtering
     * @details When set to true, enables the use of a custom Filter object for result filtering.
     *          Requires filter_ to be properly initialized with a valid FilterPtr.
     *          Default is false (no callback filtering applied).
     */
    bool enable_filter_{false};

    /**
     * @brief Custom filter object for advanced result filtering
     * @details A smart pointer to a Filter implementation that provides custom filtering logic.
     *          The filter is applied after the initial similarity search to further refine results.
     *          Only used when enable_filter_ is set to true.
     *          Allows for complex filtering scenarios not covered by attribute-based filtering.
     *          All Filter Use *AND* to connect
     */
    FilterPtr filter_{nullptr};

    /**
     * @brief Flag to enable bitset filter
     * @details When set to true, enables the use of a bitset filter for result filtering.
     *          Requires bitset_filter_ to be properly initialized with a valid BitsetPtr.
     *          Default is false (no bitset filter applied).
     */
    bool enable_bitset_filter_{false};

    /**
     * @brief Bitset filter for result filtering
     * @details A smart pointer to a Bitset implementation that provides result filtering based on a bitset.
     *          Only used when enable_bitset_filter_ is set to true.
     *          If the bitset filter's Test(id) return True, the id will be filtered out.
     *          Default is nullptr (no bitset filter applied).
     */
    BitsetPtr bitset_filter_{nullptr};

    // search resource alloc
    /**
     * @brief Custom memory allocator for search operations
     * @details Pointer to a custom Allocator object used for memory management during search.
     *          If nullptr, the default index allocator is used.
     *          Useful for memory-constrained environments or when using specialized memory pools.
     *          Can help optimize memory usage patterns for large-scale searches.
     */
    Allocator* search_allocator_{nullptr};

    // for iterator search
    /**
     * @brief Flag to enable iterator-based search mode
     * @details When set to true, enables incremental search using an iterator pattern.
     *          Allows for processing search with steps.
     *          Hold the current search context in the iterator context.
     *          Requires p_iter_ctx_ to be properly initialized.
     *          Default is false (standard search mode).
     */
    bool enable_iterator_search_{false};

    /**
     * @brief Pointer to iterator context for incremental search
     * @details Double pointer to an IteratorContext object used for maintaining search state
     *          across multiple incremental search calls. Only used when enable_iterator_search_ is true.
     *          The context stores the current position in the search space and allows resuming
     *          searches from where they left off. Must be initialized before starting iterator search.
     */
    IteratorContext** p_iter_ctx_{nullptr};

    /**
     * @brief Flag indicating this is the final search in an iterator sequence
     * @details When set to true, signals that no more results are expected from this iterator.ß
     */
    bool is_last_search_{false};
};

}  // namespace vsag
