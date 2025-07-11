
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

#include "bucket_datacell_parameter.h"

#include <fmt/format.h>

#include "inner_string_params.h"

namespace vsag {
BucketDataCellParameter::BucketDataCellParameter() = default;

void
BucketDataCellParameter::FromJson(const JsonType& json) {
    CHECK_ARGUMENT(json.contains(IO_PARAMS_KEY),
                   fmt::format("bucket interface parameters must contains {}", IO_PARAMS_KEY));
    this->io_parameter = IOParameter::GetIOParameterByJson(json[IO_PARAMS_KEY]);

    CHECK_ARGUMENT(
        json.contains(QUANTIZATION_PARAMS_KEY),
        fmt::format("bucket interface parameters must contains {}", QUANTIZATION_PARAMS_KEY));
    this->quantizer_parameter =
        QuantizerParameter::GetQuantizerParameterByJson(json[QUANTIZATION_PARAMS_KEY]);

    if (json.contains(BUCKETS_COUNT_KEY)) {
        this->buckets_count = json[BUCKETS_COUNT_KEY];
    }

    if (json.contains(BUCKET_USE_RESIDUAL)) {
        this->use_residual_ = json[BUCKET_USE_RESIDUAL];
    }
}

JsonType
BucketDataCellParameter::ToJson() {
    JsonType json;
    json[IO_PARAMS_KEY] = this->io_parameter->ToJson();
    json[BUCKET_USE_RESIDUAL] = this->use_residual_;
    json[QUANTIZATION_PARAMS_KEY] = this->quantizer_parameter->ToJson();
    json[BUCKETS_COUNT_KEY] = this->buckets_count;
    return json;
}
}  // namespace vsag
