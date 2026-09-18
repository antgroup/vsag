// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>

#include "inner_string_params.h"
#include "io/common/io_parameter.h"
#include "utils/pointer_define.h"
#include "vsag/readerset.h"

namespace vsag {

DEFINE_POINTER2(ExternalStorageIOParam, ExternalStorageIOParameter);

class ExternalStorageIOParameter : public IOParameter {
public:
    ExternalStorageIOParameter() : IOParameter(IO_TYPE_VALUE_EXTERNAL_STORAGE_IO) {
    }

    JsonType
    ToJson() const override {
        JsonType json;
        json[TYPE_KEY].SetString(IO_TYPE_VALUE_EXTERNAL_STORAGE_IO);
        json[IO_EXTERNAL_STORAGE_KEY].SetString(external_storage);
        AppendCommonConfig(json);
        return json;
    }

    void
    FromJson(const JsonType& json) override {
        if (json.Contains(IO_EXTERNAL_STORAGE_KEY)) {
            CHECK_ARGUMENT(json[IO_EXTERNAL_STORAGE_KEY].IsString(),
                           "external_storage must be a string");
            external_storage = json[IO_EXTERNAL_STORAGE_KEY].GetString();
            CHECK_ARGUMENT(not external_storage.empty(), "external_storage must not be empty");
        }
    }

    std::string external_storage;
    // Internal binding state, populated together on a private copy by bind_external_storage.
    // The reusable input parameter remains unchanged. Do not mutate a pair during initialization;
    // the backend retains its own shared ownership after initialization completes.
    ReaderPtr reader;
    WriterPtr writer;
};

}  // namespace vsag
