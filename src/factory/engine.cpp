
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

#include <string>

#include "common.h"
#include "impl/allocator/default_allocator.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "index_common_param.h"
#include "index_creators.h"
#include "index_registry.h"
#include "resource_owner_wrapper.h"
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
        register_all_index_creators();
        auto parsed_params = JsonType::Parse(parameters);
        auto index_common_params = IndexCommonParam::CheckAndCreate(parsed_params, this->resource_);
        return create_registered_index(origin_name, parsed_params, index_common_params);
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to create index(invalid argument): ", e.what());
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::NO_ENOUGH_MEMORY, "failed to create index(not enough memory): ", e.what());
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
