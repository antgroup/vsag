
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

#include "hgraph_index.h"
namespace vsag {
HGraphIndex::HGraphIndex(const HGraphIndexParameter& param,
                         const vsag::IndexCommonParam& common_param) noexcept {
    this->hgraph_ = std::make_unique<HGraph>(*param.hgraph_parameter_, common_param);
    this->allocator_ = common_param.allocator_;
}

HGraphIndex::~HGraphIndex() {
    this->hgraph_.reset();
    this->allocator_.reset();
}
}  // namespace vsag
