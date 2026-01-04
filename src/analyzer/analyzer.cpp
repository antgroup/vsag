
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use hgraph_ file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "analyzer.h"

#include "hgraph_analyzer.h"

namespace vsag {

AnalyzerBasePtr
CreateAnalyzer(const InnerIndexInterface* index, const AnalyzerParam& param) {
    if (index == nullptr) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            "Index pointer is null when creating analyzer");
    }
    auto* index_no_const = const_cast<InnerIndexInterface*>(index);
    if (dynamic_cast<HGraph*>(index_no_const) != nullptr) {
        auto* hgraph = dynamic_cast<HGraph*>(index_no_const);
        return std::make_shared<HGraphAnalyzer>(hgraph, param);
    }
    throw VsagException(
        ErrorType::UNSUPPORTED_INDEX_OPERATION,
        fmt::format("Unsupported index type ({}) for analyzer creation", index->GetName()));
}

}  // namespace vsag
