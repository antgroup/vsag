// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "vsag/search_resource.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>

#include "common.h"
#ifdef __linux__
#include <sched.h>
#endif
#include "impl/searcher/search_execution_context.h"
#include "impl/thread_pool/cpu_resource.h"
#include "vsag/engine.h"
#include "vsag_exception.h"

namespace vsag {
namespace {
uint32_t
VisibleCpuSlots() {
    uint32_t count = std::max(1U, std::thread::hardware_concurrency());
#ifdef __linux__
    cpu_set_t affinity;
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
        count = static_cast<uint32_t>(std::max(1, CPU_COUNT(&affinity)));
    const auto read = [](const std::string& path) {
        std::ifstream file(path);
        std::ostringstream value;
        value << file.rdbuf();
        return value.str();
    };
    count = CgroupCpuSlots(read("/proc/self/cgroup"), read("/proc/self/mountinfo"), read, count);
#endif
    return count;
}
}  // namespace
SearchExecutionContext::SearchExecutionContext(std::optional<uint32_t> maximum)
    : slots(VisibleCpuSlots()), limit(std::min(slots - 1, maximum.value_or(512))) {
    CHECK_ARGUMENT(!maximum || *maximum <= 512, "max_background_workers must be in [0,512]");
    if (limit != 0) {
        auto result = Engine::CreateThreadPool(limit);
        if (!result.has_value())
            throw VsagException(result.error());
        pool = std::make_shared<SafeThreadPool>(result.value());
    }
}
SearchResource::SearchResource(const Resource& common, const SearchResourceOptions& options)
    : Resource(common.GetAllocator(), common.GetThreadPool()),
      search_context_(std::make_shared<SearchExecutionContext>(options.max_background_workers)) {
}
SearchResource::SearchResource(const Resource& common, const SearchResource& shared_search)
    : Resource(common.GetAllocator(), common.GetThreadPool()),
      search_context_(shared_search.search_context_) {
}
SearchResource::~SearchResource() = default;
}  // namespace vsag
