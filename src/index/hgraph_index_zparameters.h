
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

#include "algorithm/hgraph_parameter.h"
#include "index_common_param.h"
#include "parameter.h"
#include "typing.h"

namespace vsag {
class HGraphIndexParameter : public Parameter {
public:
    explicit HGraphIndexParameter(IndexCommonParam common_param);

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() override;

public:
    std::shared_ptr<HGraphParameter> hgraph_parameter_{nullptr};

private:
    void
    check_common_param() const;

private:
    const IndexCommonParam common_param_;
};

class HGraphSearchParameters {
public:
    static HGraphSearchParameters
    FromJson(const std::string& json_string);

public:
    int64_t ef_search{30};
    bool use_reorder{false};

private:
    HGraphSearchParameters() = default;
};

}  // namespace vsag
