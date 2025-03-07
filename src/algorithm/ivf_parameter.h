
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
#include "data_cell/bucket_datacell_parameter.h"
#include "fmt/format-inl.h"
#include "inner_string_params.h"
#include "parameter.h"
#include "typing.h"

namespace vsag {
class IVFParameter : public Parameter {
public:
    explicit IVFParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() override;

public:
    BucketDataCellParamPtr bucket_param{nullptr};

    bool use_residual{false};
};

using IVFParameterPtr = std::shared_ptr<IVFParameter>;

class IVFSearchParameters {
public:
    static IVFSearchParameters
    FromJson(const std::string& json_string) {
        JsonType params = JsonType::parse(json_string);

        IVFSearchParameters obj;

        // set obj.scan_buckets_count
        CHECK_ARGUMENT(params.contains(INDEX_TYPE_IVF),
                       fmt::format("parameters must contains {}", INDEX_TYPE_IVF));

        CHECK_ARGUMENT(params[INDEX_TYPE_IVF].contains(IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT),
                       fmt::format("parameters[{}] must contains {}",
                                   INDEX_TYPE_IVF,
                                   IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT));
        obj.scan_buckets_count = params[INDEX_TYPE_IVF][IVF_SEARCH_PARAM_SCAN_BUCKETS_COUNT];
        return obj;
    }

public:
    int64_t scan_buckets_count{30};

private:
    IVFSearchParameters() = default;
};

}  // namespace vsag
