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

#include <memory>
#include <string>

#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/filter.h"

namespace vsag {

/** Options replace eligibility and effort for this call only; nullptr accepts all labels. */
struct SearchSessionNextOptions {
    uint64_t max_candidates{10};
    std::string search_parameters{R"({"hgraph":{"ef_search":100}})"};
    FilterPtr filter{nullptr};
};

/**
 * @brief Continue-search session produced by creating a SearchSession from an index.
 *
 * Owns the immutable query specification, search state, and lifecycle of a
 * continuing similarity search.  After OpenSearchSession the caller advances
 * the search with Next to receive non-overlapping candidate batches until
 * exhausted or closed. HGraph initializes query computers and routes lazily
 * on the first Next, and retains its frontier, cached scores and delivery state.
 * Each Next expands at least max(ef_search, max_candidates) remaining
 * frontier vertices (or exhausts it), continuing further to fill a filtered page.
 * Results are sorted within a page, not guaranteed globally sorted across pages.
 * Does not extend IteratorContext; existing iterator
 * APIs remain unchanged. The query is deep-copied and the backend is retained.
 * Sessions are single-consumer: do not call Next/HasMore/Close concurrently,
 * or mutate the index. Filters may change between calls (even the same object),
 * but must remain stable during Next. Every call rechecks discovered undelivered
 * IDs, including earlier rejects. Delivered IDs never return again.
 * Empty filtered batches are not terminal: stop on empty for the current request,
 * rather than looping solely on HasMore under an unsatisfiable filter.
 * Exhaustion means the retained approximate search scope is exhausted, not
 * that every vector in the index has been enumerated. HGraph enumerates the
 * reachable bottom graph. HasMore may be true before filtering finds any result.
 * The optional caller allocator must outlive the session; returned result buffers
 * are independently owned and may outlive both session and index.
 * Specialized RaBitQ, parallel/timed/factored/brute-force/hop-limited searches,
 * explicit filter skipping, shared duplicate storage, force removal, active MCI and
 * conjugate search are rejected. Ordinary filtering, threshold and reorder are
 * supported, including ordinary duplicate labels; filtered vertices remain traversable bridges.
 * Invalid Next demand returns INVALID_ARGUMENT without changing session state.
 * Errors during traversal close the session; already returned results stay valid.
 * Terminal/closed Next calls return empty with cumulative work statistics preserved
 * and zero per-round work, without demand validation. Close releases traversal storage.
 */
class SearchSession {
public:
    virtual ~SearchSession() = default;

    /**
     * @brief Advance the session and return the next batch of candidates.
     *
     * @param max_candidates  upper bound on returned candidate count.
     * @return a Dataset of at most max_candidates results in output-score order;
     *         empty when no more candidates exist or session is terminal.
     */
    [[nodiscard]] virtual tl::expected<DatasetPtr, Error>
    Next(uint64_t max_candidates) = 0;

    /** Per-call conditions replace defaults; invalid options preserve valid session state. */
    [[nodiscard]] virtual tl::expected<DatasetPtr, Error>
    Next(const SearchSessionNextOptions& options) = 0;

    /**
     * @brief Whether the session has at least one pending candidate.
     *
     * true  → Next may return a non-empty batch (not guaranteed).
     * false → Next is guaranteed to return empty; the search scope is exhausted.
     */
    [[nodiscard]] virtual bool
    HasMore() const noexcept = 0;

    /**
     * @brief Idempotent close.  After close, HasMore() returns false and
     *        Next returns empty without executing search work.  Previously
     *        returned results remain valid.
     */
    virtual void
    Close() noexcept = 0;
};

}  // namespace vsag
