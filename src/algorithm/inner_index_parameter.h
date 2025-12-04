
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

#include "parameter.h"
#include "typing.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER(InnerIndexParameter);
DEFINE_POINTER2(ExtraInfoDataCellParam, ExtraInfoDataCellParameter);
DEFINE_POINTER2(FlattenInterfaceParam, FlattenInterfaceParameter);
DEFINE_POINTER2(AttributeInvertedInterfaceParam, AttributeInvertedInterfaceParameter);

class InnerIndexParameter : public Parameter {
public:
    explicit InnerIndexParameter() = default;

    ~InnerIndexParameter() override = default;

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const vsag::ParamPtr& other) const override;

public:
    bool use_reorder{false};
    FlattenInterfaceParamPtr precise_codes_param{nullptr};

    bool use_attribute_filter{false};

    uint64_t build_thread_count{1};

    bool store_raw_vector{false};
    FlattenInterfaceParamPtr raw_vector_param{nullptr};

    ExtraInfoDataCellParamPtr extra_info_param{nullptr};

    AttributeInvertedInterfaceParamPtr attr_inverted_interface_param{nullptr};
};
}  // namespace vsag
