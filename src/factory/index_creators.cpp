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

#include "index_creators.h"

#include <fmt/format.h>

#include <mutex>

#include "algorithm/brute_force.h"
#include "algorithm/hgraph.h"
#include "algorithm/ivf.h"
#include "algorithm/pyramid.h"
#include "algorithm/pyramid_zparameters.h"
#include "algorithm/sindi/sindi.h"
#include "algorithm/sparse_index.h"
#include "common.h"
#include "index/diskann.h"
#include "index/diskann_zparameters.h"
#include "index/hnsw.h"
#include "index/hnsw_zparameters.h"
#include "index/index_impl.h"
#include "index_registry.h"

namespace vsag {
namespace {

JsonType
get_index_param_or_empty(const JsonType& parsed_params) {
    JsonType json;
    if (parsed_params.Contains(INDEX_PARAM)) {
        json = parsed_params[INDEX_PARAM];
    }
    return json;
}

tl::expected<std::shared_ptr<Index>, Error>
create_hnsw_index(JsonType& parsed_params, const IndexCommonParam& index_common_params) {
    CHECK_ARGUMENT(parsed_params.Contains(INDEX_HNSW),
                   fmt::format("parameters must contain {}", INDEX_HNSW));
    auto hnsw_param_obj = parsed_params[INDEX_HNSW];
    auto hnsw_params = HnswParameters::FromJson(hnsw_param_obj, index_common_params);
    logger::debug("created a hnsw index");
    auto index = std::make_shared<HNSW>(hnsw_params, index_common_params);
    if (auto result = index->InitMemorySpace(); not result.has_value()) {
        return tl::unexpected(result.error());
    }
    return index;
}

tl::expected<std::shared_ptr<Index>, Error>
create_fresh_hnsw_index(JsonType& parsed_params, const IndexCommonParam& index_common_params) {
    CHECK_ARGUMENT(parsed_params.Contains(INDEX_FRESH_HNSW),
                   fmt::format("parameters must contain {}", INDEX_FRESH_HNSW));
    auto hnsw_param_obj = parsed_params[INDEX_FRESH_HNSW];
    auto hnsw_params = FreshHnswParameters::FromJson(hnsw_param_obj, index_common_params);
    logger::debug("created a fresh-hnsw index");
    auto index = std::make_shared<HNSW>(hnsw_params, index_common_params);
    if (auto result = index->InitMemorySpace(); not result.has_value()) {
        return tl::unexpected(result.error());
    }
    return index;
}

tl::expected<std::shared_ptr<Index>, Error>
create_brute_force_index(JsonType& parsed_params, const IndexCommonParam& index_common_params) {
    logger::debug("created a brute_force index");
    return {std::make_shared<IndexImpl<BruteForce>>(get_index_param_or_empty(parsed_params),
                                                    index_common_params)};
}

tl::expected<std::shared_ptr<Index>, Error>
create_diskann_index(JsonType& parsed_params, const IndexCommonParam& index_common_params) {
    CHECK_ARGUMENT(parsed_params.Contains(INDEX_DISKANN),
                   fmt::format("parameters must contain {}", INDEX_DISKANN));
    auto diskann_param_obj = parsed_params[INDEX_DISKANN];
    auto diskann_params = DiskannParameters::FromJson(diskann_param_obj, index_common_params);
    logger::debug("created a diskann index");
    return std::make_shared<DiskANN>(diskann_params, index_common_params);
}

template <typename T>
tl::expected<std::shared_ptr<Index>, Error>
create_index_impl_with_param_log(const char* log_message,
                                 JsonType& parsed_params,
                                 const IndexCommonParam& index_common_params) {
    logger::debug(log_message);
    return {std::make_shared<IndexImpl<T>>(get_index_param_or_empty(parsed_params),
                                           index_common_params)};
}

tl::expected<std::shared_ptr<Index>, Error>
create_pyramid_index(JsonType& parsed_params, const IndexCommonParam& index_common_params) {
    CHECK_ARGUMENT(parsed_params.Contains(INDEX_PARAM),
                   fmt::format("parameters must contain {}", INDEX_PARAM));
    logger::debug("created a pyramid index");
    return {std::make_shared<IndexImpl<Pyramid>>(parsed_params[INDEX_PARAM], index_common_params)};
}

tl::expected<std::shared_ptr<Index>, Error>
create_hgraph_index(JsonType& parsed_params, const IndexCommonParam& index_common_params) {
    return create_index_impl_with_param_log<HGraph>(
        "created a hgraph index", parsed_params, index_common_params);
}

tl::expected<std::shared_ptr<Index>, Error>
create_ivf_index(JsonType& parsed_params, const IndexCommonParam& index_common_params) {
    return create_index_impl_with_param_log<IVF>(
        "created an ivf index", parsed_params, index_common_params);
}

tl::expected<std::shared_ptr<Index>, Error>
create_sparse_index(JsonType& parsed_params, const IndexCommonParam& index_common_params) {
    return create_index_impl_with_param_log<SparseIndex>(
        "created a sparse index", parsed_params, index_common_params);
}

tl::expected<std::shared_ptr<Index>, Error>
create_sindi_index(JsonType& parsed_params, const IndexCommonParam& index_common_params) {
    return create_index_impl_with_param_log<SINDI>(
        "created a sindi index", parsed_params, index_common_params);
}

}  // namespace

void
register_all_index_creators() {
    static std::once_flag registration_once;
    std::call_once(registration_once, []() {
        register_index_creator(INDEX_HNSW, &create_hnsw_index);
        register_index_creator(INDEX_FRESH_HNSW, &create_fresh_hnsw_index);
        register_index_creator(INDEX_BRUTE_FORCE, &create_brute_force_index);
        register_index_creator(INDEX_DISKANN, &create_diskann_index);
        register_index_creator(INDEX_HGRAPH, &create_hgraph_index);
        register_index_creator(INDEX_IVF, &create_ivf_index);
        register_index_creator(INDEX_PYRAMID, &create_pyramid_index);
        register_index_creator(INDEX_SPARSE, &create_sparse_index);
        register_index_creator(INDEX_SINDI, &create_sindi_index);
    });
}

}  // namespace vsag
