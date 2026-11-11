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

#include <cstddef>
#include <limits>
#include <memory>
#include <set>
#include <vector>

#include "algorithm/hgraph/hgraph_parameter.h"
#include "impl/searcher/searcher_utils.h"
#include "typing.h"
#include "vsag/filter.h"
#include "vsag/search_session.h"
#include "vsag_exception.h"

namespace vsag {
class HGraph;
class InnerIndexInterface;

// FindFiniteSessionRoute is defined in the sibling internal-routing header
// (src/algorithm/hgraph/hgraph_continue_session_routing.h) so template
// instantiation stays confined to the single translation unit that needs it.

// Internal guards shared with boundary tests; no backend mutation hooks are exposed.
inline void
ValidateSessionId(InnerIdType id, uint64_t count) {
    if (static_cast<uint64_t>(id) >= count) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "session graph id is out of bounds");
    }
}

inline uint64_t
CheckedSessionBytes(uint64_t count, uint64_t width) {
    if (width != 0 and count > std::numeric_limits<uint64_t>::max() / width) {
        throw VsagException(ErrorType::NO_ENOUGH_MEMORY, "session result size overflows");
    }
    return count * width;
}

// Diagnostic-only arithmetic must not throw or wrap when reporting logical payload.
inline uint64_t
SaturatingSessionPayload(uint64_t total, uint64_t count, uint64_t width) noexcept {
    const auto limit = std::numeric_limits<uint64_t>::max();
    if (width != 0 and count > (limit - total) / width) {
        return limit;
    }
    return total + count * width;
}

class HGraphContinueSession final : public SearchSession {
private:
    friend class HGraph;
    HGraphContinueSession(const HGraph& hgraph,
                          const std::shared_ptr<const InnerIndexInterface>& owner,
                          DatasetPtr query,
                          std::string parameters,
                          FilterPtr filter,
                          Allocator* allocator);
    void
    Initialize(int64_t k, int64_t ef_search);

public:
    ~HGraphContinueSession() override;
    HGraphContinueSession(const HGraphContinueSession&) = delete;
    HGraphContinueSession&
    operator=(const HGraphContinueSession&) = delete;

    tl::expected<DatasetPtr, Error>
    Next(uint64_t max_candidates) override;
    tl::expected<DatasetPtr, Error>
    Next(const SearchSessionNextOptions& options) override;
    bool
    HasMore() const noexcept override;
    void
    Close() noexcept override;

private:
    struct Counters {
        uint64_t routing{0};
        uint64_t computers{0};
        uint64_t scored{0};
        uint64_t reordered{0};
        uint64_t expanded{0};
    };
    Counters
    GetCounters() const noexcept;
    std::string
    Statistics(const Counters& before) const;
    tl::expected<DatasetPtr, Error>
    NextImpl(const SearchSessionNextOptions& options, uint64_t minimum_effort);
    struct ValidatedParameters {
        HGraphSearchParameters search;
        std::optional<float> threshold;
    };
    ValidatedParameters
    ValidateParameters(const std::string& parameters) const;
    uint64_t default_effort_{1};
    Counters closed_counters_;
    // Strong backend ownership keeps graph state and its allocator alive.
    std::shared_ptr<const HGraph> hgraph_;
    DatasetPtr query_;
    std::string parameters_;
    FilterPtr filter_;
    Allocator* allocator_;
    struct Traversal;
    std::unique_ptr<Traversal> traversal_;
    bool closed_{false};
};
}  // namespace vsag
