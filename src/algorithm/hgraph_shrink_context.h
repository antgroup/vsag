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

#include <chrono>

#include "hgraph.h"
#include "hgraph_shrink_checkpoint.h"
#include "hgraph_shrink_edges.h"
#include "hgraph_shrink_state.h"

namespace vsag {

class HGraphShrinkContext {
public:
    explicit HGraphShrinkContext(HGraph* hgraph);

    void
    Run(double timeout_ms);

private:
    bool
    prepare(double timeout_ms);

    bool
    collect_forward_edges(double timeout_ms);

    bool
    collect_reverse_edges(double timeout_ms);

    bool
    process_reverse_edges(double timeout_ms);

    bool
    process_forward_edges(double timeout_ms);

    bool
    cleanup(double timeout_ms);

    inline bool
    check_timeout(double timeout_ms) const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
        return elapsed_ms >= timeout_ms;
    }

private:
    std::chrono::steady_clock::time_point start_time_;
    HGraph* hgraph_{nullptr};
    HGraphShrinkState state_{HGraphShrinkState::IDLE};
    HgraphShrinkCheckPoint check_point_;
};

}  // namespace vsag
