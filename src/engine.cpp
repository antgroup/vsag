
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

#include <fmt/format-inl.h>

#include <string>

#include "common.h"
#include "index/brute_force.h"
#include "index/brute_force_parameter.h"
#include "index/diskann.h"
#include "index/diskann_zparameters.h"
#include "index/hgraph_index.h"
#include "index/hgraph_index_zparameters.h"
#include "index/hnsw.h"
#include "index/hnsw_zparameters.h"
#include "index/index_common_param.h"
#include "index/pyramid.h"
#include "index/pyramid_zparameters.h"
#include "resource_owner_wrapper.h"
#include "safe_thread_pool.h"
#include "typing.h"

// NOLINTBEGIN(readability-else-after-return )

namespace vsag {

Engine::Engine() {
    this->resource_ = std::make_shared<ResourceOwnerWrapper>(new Resource(), /*owned*/ true);
}

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
            BruteForceParameter param;
            param.FromJson(json);
            auto brute_force = std::make_shared<BruteForce>(param, index_common_params);

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
            auto hgraph_param = std::make_shared<HGraphIndexParameter>(index_common_params);
            hgraph_param->FromJson(hgraph_json);
            auto hgraph_index = std::make_shared<HGraphIndex>(*hgraph_param, index_common_params);
            return hgraph_index;
        } else if (name == INDEX_PYRAMID) {
            // read parameters from json, throw exception if not exists
            CHECK_ARGUMENT(parsed_params.contains(INDEX_PARAM),
                           fmt::format("parameters must contains {}", INDEX_PARAM));
            auto& pyramid_param_obj = parsed_params[INDEX_PARAM];
            PyramidParameters pyramid_params;
            pyramid_params.FromJson(pyramid_param_obj);
            logger::debug("created a pyramid index");
            return std::make_shared<Pyramid>(pyramid_params, index_common_params);
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
    }
}
}  // namespace vsag

// NOLINTEND(readability-else-after-return )
