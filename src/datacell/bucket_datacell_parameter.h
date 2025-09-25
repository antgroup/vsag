
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

#include "io/io_parameter.h"
#include "parameter.h"
#include "quantization/quantizer_parameter.h"

namespace vsag {

class BucketDataCellParameter : public Parameter {
public:
    explicit BucketDataCellParameter();

    void
    FromJson(const JsonType& json) override;

    JsonType
    ToJson() const override;

    bool
    CheckCompatibility(const ParamPtr& other) const override;

public:
    QuantizerParamPtr quantizer_parameter{nullptr};

    IOParamPtr io_parameter{nullptr};

    bool use_residual_{false};

    int64_t buckets_count{1};
};

using BucketDataCellParamPtr = std::shared_ptr<BucketDataCellParameter>;

}  // namespace vsag
