
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

#include "vsag/engine.h"

#include <fmt/format.h>

#include <string>

#include "algorithm/brute_force.h"
#include "algorithm/hgraph.h"
#include "algorithm/ivf.h"
#include "algorithm/pyramid.h"
#include "algorithm/pyramid_zparameters.h"
#include "algorithm/sparse_index.h"
#include "common.h"
#include "index/diskann.h"
#include "index/diskann_zparameters.h"
#include "index/hnsw.h"
#include "index/hnsw_zparameters.h"
#include "index/index_common_param.h"
#include "index/index_impl.h"
#include "resource_owner_wrapper.h"
#include "safe_thread_pool.h"
#include "typing.h"

// NOLINTBEGIN(readability-else-after-return )

namespace vsag {

Engine::Engine(Resource* resource) {
    if (resource == nullptr) {
        this->resource_ = std::make_shared<ResourceOwnerWrapper>(new Resource(), /*owned*/ true);
    } else {
        this->resource_ = std::make_shared<ResourceOwnerWrapper>(resource, /*owned*/ false);
    }
}

void
Engine::Shutdown() {
    auto refcount = this->resource_.use_count();
    this->resource_.reset();

    // TODO(LHT): add refcount warning
}

tl::expected<std::shared_ptr<Index>, Error>
Engine::CreateIndex(const std::string& origin_name, const std::string& parameters) {
    try {
        std::string name = origin_name;
        transform(name.begin(), name.end(), name.begin(), ::tolower);
        JsonType parsed_params = JsonType::parse(parameters);
        auto index_common_params = IndexCommonParam::CheckAndCreate(parsed_params, this->resource_);
        if (name == INDEX_HNSW) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_HNSW),
                           fmt::format("parameters must contains {}", INDEX_HNSW));
            auto& hnsw_param_obj = parsed_params[INDEX_HNSW];
            auto hnsw_params = HnswParameters::FromJson(hnsw_param_obj, index_common_params);
            logger::debug("created a hnsw index");
            auto index = std::make_shared<HNSW>(hnsw_params, index_common_params);
            if (auto result = index->InitMemorySpace(); not result.has_value()) {
                return tl::unexpected(result.error());
            }
            return index;
        } else if (name == INDEX_FRESH_HNSW) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_FRESH_HNSW),
                           fmt::format("parameters must contains {}", INDEX_FRESH_HNSW));
            auto& hnsw_param_obj = parsed_params[INDEX_FRESH_HNSW];
            auto hnsw_params = FreshHnswParameters::FromJson(hnsw_param_obj, index_common_params);
            logger::debug("created a fresh-hnsw index");
            auto index = std::make_shared<HNSW>(hnsw_params, index_common_params);
            if (auto result = index->InitMemorySpace(); not result.has_value()) {
                return tl::unexpected(result.error());
            }
            return index;
        } else if (name == INDEX_BRUTE_FORCE) {
            logger::debug("created a brute_force index");
            JsonType json;
            if (parsed_params.contains(INDEX_PARAM)) {
                json = std::move(parsed_params[INDEX_PARAM]);
            }
            auto brute_force = std::make_shared<IndexImpl<BruteForce>>(json, index_common_params);
            return brute_force;
        } else if (name == INDEX_DISKANN) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_DISKANN),
                           fmt::format("parameters must contains {}", INDEX_DISKANN));
            auto& diskann_param_obj = parsed_params[INDEX_DISKANN];
            auto diskann_params =
                DiskannParameters::FromJson(diskann_param_obj, index_common_params);
            logger::debug("created a diskann index");
            return std::make_shared<DiskANN>(diskann_params, index_common_params);
        } else if (name == INDEX_HGRAPH) {
            logger::debug("created a hgraph index");
            JsonType hgraph_json;
            if (parsed_params.contains(INDEX_PARAM)) {
                hgraph_json = std::move(parsed_params[INDEX_PARAM]);
            }
            auto hgraph_index =
                std::make_shared<IndexImpl<HGraph>>(hgraph_json, index_common_params);
            return hgraph_index;
        } else if (name == INDEX_IVF) {
            logger::debug("created an ivf index");
            JsonType ivf_json;
            if (parsed_params.contains(INDEX_PARAM)) {
                ivf_json = std::move(parsed_params[INDEX_PARAM]);
            }
            auto ivf_index = std::make_shared<IndexImpl<IVF>>(ivf_json, index_common_params);
            return ivf_index;
        } else if (name == INDEX_PYRAMID) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_PARAM),
                           fmt::format("parameters must contains {}", INDEX_PARAM));
            auto& pyramid_param_obj = parsed_params[INDEX_PARAM];
            logger::debug("created a pyramid index");
            auto pyramid_index =
                std::make_shared<IndexImpl<Pyramid>>(pyramid_param_obj, index_common_params);
            return pyramid_index;
        } else if (name == INDEX_SPARSE) {
            logger::debug("created a sparse index");
            JsonType sparse_json;
            if (parsed_params.contains(INDEX_PARAM)) {
                sparse_json = std::move(parsed_params[INDEX_PARAM]);
            }
            auto sparse_index =
                std::make_shared<IndexImpl<SparseIndex>>(sparse_json, index_common_params);
            return sparse_index;
        } else {
            LOG_ERROR_AND_RETURNS(
                ErrorType::UNSUPPORTED_INDEX, "failed to create index(unsupported): ", name);
        }
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to create index(invalid argument): ", e.what());
    } catch (const std::exception& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::UNSUPPORTED_INDEX, "failed to create index(unknown error): ", e.what());
    } catch (const vsag::VsagException& e) {
        LOG_ERROR_AND_RETURNS(e.error_.type, "failed to create index: " + e.error_.message);
    }
}

std::shared_ptr<Allocator>
Engine::CreateDefaultAllocator() {
    return std::make_shared<DefaultAllocator>();
}

tl::expected<std::shared_ptr<ThreadPool>, Error>
Engine::CreateThreadPool(uint32_t num_threads) {
    if (num_threads <= 0 || num_threads > 512) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to create thread pool: invalid number of threads:",
                              std::to_string(num_threads));
    }
    return std::make_shared<DefaultThreadPool>(num_threads);
}

}  // namespace vsag

// NOLINTEND(readability-else-after-return )
