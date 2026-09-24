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

#include <string>

#include "algorithm/simq/simq.h"
#include "analyzer.h"

namespace vsag {

class SIMQAnalyzer : public AnalyzerBase {
public:
    SIMQAnalyzer(SIMQ* simq, const AnalyzerParam& param)
        : AnalyzerBase(simq->allocator_, static_cast<uint32_t>(simq->GetNumElements())),
          simq_(simq),
          topk_(param.topk),
          search_params_(param.search_params) {
    }

    JsonType
    GetStats() override;

    JsonType
    AnalyzeIndexBySearch(const SearchRequest& request) override;

private:
    SIMQ* simq_{nullptr};
    int64_t topk_{10};
    std::string search_params_;
};

}  // namespace vsag
