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

#include "simq_analyzer.h"

#include "json_types.h"

namespace vsag {

JsonType
SIMQAnalyzer::GetStats() {
    // Delegate to simq_->GetStats() which correctly acquires global_mutex_
    // before reading internal state, avoiding data races and code duplication.
    return JsonType::Parse(simq_->GetStats());
}

JsonType
SIMQAnalyzer::AnalyzeIndexBySearch(const SearchRequest& /*request*/) {
    JsonType stats;
    return stats;
}

}  // namespace vsag
