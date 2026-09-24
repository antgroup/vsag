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
#include <sstream>

#include "vsag_exception.h"

namespace vsag {

struct GraphBuildProgress {
    // Completed input members, NOT graph TotalCount (duplicates and sparse slots differ).
    uint64_t processed_members{0};
};

// Synchronous graph construction steps shared by candidate replay and prepared online
// insertion. The owner retains storage, workspaces, locks and publication responsibility.
class GraphBuildHelper {
public:
    // Named operations own reusable workspace. MakeState returns a directly constructed
    // per-member state (no assignment of resource-owning search parameters). Prepare returns
    // true for an initialized first member; otherwise run search/duplicate/connect exactly once.
    // Discard candidates after an exception: progress is accounting, not a rollback journal.
    template <typename Ids, typename Operations>
    static void
    AppendIds(const Ids& ids, GraphBuildProgress& progress, Operations& operations) {
        for (const auto id : ids) {
            auto state = operations.MakeState(id, progress.processed_members);
            if (not operations.Prepare(state)) {
                RunPrepared(operations, state);
            }
            ++progress.processed_members;
        }
    }

    // Also usable by an online owner after applying its own lock-release policy to the
    // prepared state. This does not acquire locks or publish/advance candidate progress.
    template <typename Operations, typename State>
    static void
    RunPrepared(Operations& operations, State& state) {
        auto candidates = operations.Search(state);
        if (not operations.AcceptDuplicate(state, candidates)) {
            operations.Connect(state, candidates);
            operations.Finish(state);
        }
    }

    // Complete-index backend: preserve the exported Dataset without copying/re-encoding
    // here. Reject partially built indexes before ownership can reach the active owner.
    // The concrete index dependency stays in the caller's factory, not this helper.
    template <typename Factory, typename Dataset>
    static auto
    BuildIndexCandidate(Factory&& factory, const Dataset& dataset, const char* failure_message) {
        auto candidate = factory();
        candidate->InitFeatures();
        auto failed_ids = candidate->Build(dataset);
        if (not failed_ids.empty()) {
            throw VsagException(ErrorType::INVALID_ARGUMENT, failure_message);
        }
        return candidate;
    }
};

}  // namespace vsag
