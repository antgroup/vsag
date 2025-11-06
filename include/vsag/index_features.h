
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

namespace vsag {

enum IndexFeature {
    NEED_TRAIN = 1, /**< Indicates that the class needs to be trained */

    SUPPORT_BUILD,                     /**< Supports building the index */
    SUPPORT_ADD_AFTER_BUILD,           /**< Supports adding new elements after building*/
    SUPPORT_ADD_FROM_EMPTY,            /**< Supports adding elements to an empty index */
    SUPPORT_KNN_SEARCH,                /**< Supports K-nearest neighbor search */
    SUPPORT_KNN_SEARCH_WITH_ID_FILTER, /**< Supports K-nearest neighbor search with ID filtering */
    SUPPORT_RANGE_SEARCH,              /**< Supports range search */
    SUPPORT_RANGE_SEARCH_WITH_ID_FILTER, /**< Supports range search with ID filtering */
    SUPPORT_KNN_ITERATOR_FILTER_SEARCH,  /**< Support iterator filter search **/
    SUPPORT_DELETE_BY_ID,                /**< Supports deleting elements by ID */
    SUPPORT_BATCH_SEARCH,                /**< Supports batch search */
    SUPPORT_METRIC_TYPE_L2,              /**< Supports L2 metric type */
    SUPPORT_METRIC_TYPE_INNER_PRODUCT,   /**< Supports inner product metric type */
    SUPPORT_METRIC_TYPE_COSINE,          /**< Supports cosine metric type */
    SUPPORT_SERIALIZE_FILE,              /**< Supports serialization to a file */
    SUPPORT_SERIALIZE_BINARY_SET,        /**< Supports serialization to a binary set */
    SUPPORT_SERIALIZE_WRITE_FUNC,        /**< Supports serialization to a write function */
    SUPPORT_DESERIALIZE_FILE,            /**< Supports deserialization from a file */
    SUPPORT_DESERIALIZE_BINARY_SET,      /**< Supports deserialization from a binary set */
    SUPPORT_DESERIALIZE_READER_SET,      /**< Supports deserialization from a reader set */
    SUPPORT_RESET,                       /**< Supports resetting the class */
    SUPPORT_FEEDBACK,                    /**< Supports feedback */
    SUPPORT_CAL_DISTANCE_BY_ID,          /**< Supports calculating distance by ID */

    SUPPORT_TRAIN_WITH_MULTI_THREAD,      /**< Supports training with multi-threading */
    SUPPORT_BUILD_WITH_MULTI_THREAD,      /**< Supports building with multi-threading */
    SUPPORT_BATCH_ADD_WITH_MULTI_THREAD,  /**< Supports batch adding with multi-threading */
    SUPPORT_SEARCH_ONE_WITH_MULTI_THREAD, /**< Supports searching one element with multi-threading */
    SUPPORT_BATCH_SEARCH_WITH_MULTI_THREAD, /**< Supports batch searching with multi-threading */

    SUPPORT_ADD_CONCURRENT,               /**< Supports concurrent addition of elements */
    SUPPORT_UPDATE_ID_CONCURRENT,         /**< Supports concurrent update id of elements */
    SUPPORT_UPDATE_VECTOR_CONCURRENT,     /**< Supports concurrent update vector of elements */
    SUPPORT_UPDATE_EXTRA_INFO_CONCURRENT, /**< Supports concurrent update extra_info of elements */
    SUPPORT_SEARCH_CONCURRENT,            /**< Supports concurrent searching */
    SUPPORT_DELETE_CONCURRENT,            /**< Supports concurrent deletion */
    SUPPORT_ADD_SEARCH_CONCURRENT,        /**< Supports concurrent addition and searching */
    SUPPORT_ADD_DELETE_CONCURRENT,        /**< Supports concurrent addition and deletion */
    SUPPORT_SEARCH_DELETE_CONCURRENT,     /**< Supports concurrent searching and deletion */
    SUPPORT_ADD_SEARCH_DELETE_CONCURRENT, /**< Supports concurrent addition, searching, and deletion */

    SUPPORT_ESTIMATE_MEMORY, /**< Supports estimate memory usage by data count */

    SUPPORT_CHECK_ID_EXIST, /**< Supports check whether given id exists in index */

    SUPPORT_MERGE_INDEX, /**< Supports to merge indices of the same type */

    SUPPORT_GET_EXTRA_INFO_BY_ID, /**< Supports get extra_info by id */

    SUPPORT_GET_RAW_VECTOR_BY_IDS, /**< Supports get raw vectors by ids */
    SUPPORT_GET_DATA_BY_IDS,       /**< Supports get data by ids */

    SUPPORT_EXPORT_IDS, /**< Supports export ids */

    SUPPORT_KNN_SEARCH_WITH_EX_FILTER, /**< Supports K-nearest neighbor search with extra info filtering */

    SUPPORT_CLONE, /**< Supports clone index */

    SUPPORT_EXPORT_MODEL, /**< Supports export model */

    INDEX_FEATURE_COUNT /** must be last one */
};
}  // namespace vsag
