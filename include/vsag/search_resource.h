// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <memory>
#include <optional>

#include "vsag/resource.h"

namespace vsag {
class Engine;
class SearchExecutionContext;

struct SearchResourceOptions {
    // Omitted: visible CPU budget minus the calling thread, bounded by 512.
    // Zero: a ready execution context with no background workers.
    std::optional<uint32_t> max_background_workers;
};

/** Optional, index-independent query resources. The common Resource ABI is unchanged.
 * Construction confirms worker startup; failure publishes no partially ready context.
 * Keep this object alive while an Engine borrows it. Indices retain the shared query
 * context. Ordinary Resource users do not implicitly acquire query resources.
 */
class SearchResource : public Resource {
public:
    explicit SearchResource(const Resource& common_resources,
                            const SearchResourceOptions& options = {});
    // Reuse query resources while composing a different common allocator/pool.
    SearchResource(const Resource& common_resources, const SearchResource& shared_search);
    ~SearchResource() override;

private:
    friend class Engine;
    std::shared_ptr<SearchExecutionContext> search_context_;
};
}  // namespace vsag
