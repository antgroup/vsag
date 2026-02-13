
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

#include "inner_string_params.h"
#include "io_parameter.h"
#include "utils/pointer_define.h"
#include "vsag/readerset.h"
namespace vsag {
DEFINE_POINTER2(ReaderIOParam, ReaderIOParameter);
/**
 * @brief ReaderIOParameter is a class that represents the parameters for a reader in the vsag project.
 * It inherits from IOParameter and is used to define the specific parameters required for reading operations.
 */
class ReaderIOParameter : public IOParameter {
public:
    ReaderIOParameter() : IOParameter(IO_TYPE_VALUE_READER_IO) {
    }

    JsonType
    ToJson() const override {
        JsonType json;
        json[TYPE_KEY].SetString(IO_TYPE_VALUE_READER_IO);
        return json;
    }

    void
    FromJson(const JsonType& json) override {
    }

    std::shared_ptr<Reader> reader;
};
}  // namespace vsag
