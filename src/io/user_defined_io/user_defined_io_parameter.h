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

DEFINE_POINTER2(UserDefinedIOParam, UserDefinedIOParameter);

class UserDefinedIOParameter : public IOParameter {
public:
    UserDefinedIOParameter() : IOParameter(IO_TYPE_VALUE_USER_DEFINED_IO) {
    }

    JsonType
    ToJson() const override {
        JsonType json;
        json[TYPE_KEY].SetString(IO_TYPE_VALUE_USER_DEFINED_IO);
        json[IO_USER_DEFINED_IO_KEY].SetString(user_defined_io);
        AppendCommonConfig(json);
        return json;
    }

    void
    FromJson(const JsonType& json) override {
        if (json.Contains(IO_USER_DEFINED_IO_KEY)) {
            CHECK_ARGUMENT(json[IO_USER_DEFINED_IO_KEY].IsString(),
                           "user_defined_io must be a string");
            user_defined_io = json[IO_USER_DEFINED_IO_KEY].GetString();
            CHECK_ARGUMENT(not user_defined_io.empty(), "user_defined_io must not be empty");
        }
        // Common config (read cache, prefetch hint) is loaded centrally by
        // IOParameter::GetIOParameterByJson on every io_ptr, matching the other IOParameter
        // subclasses: no FromJson override parses it. Loading it here as well would be a second
        // owner of the same keys.
    }

    // Registered name of the Reader/Writer pair. Mutating this after a WithStorage binding does not
    // rebind existing copies: a bound copy keeps the pair it captured, while an unbound original is
    // still resolved by name at Initialize time.
    std::string user_defined_io;
    // Return a bound copy without exposing mutable binding state on the reusable parameter.
    // The copy semantics are deliberate: the original stays unbound so it can be reused for another
    // construction, and Initialize rejects a parameter that was never bound.
    UserDefinedIOParamPtr
    WithStorage(const UserDefinedIOPair& storage) const {
        auto bound = std::make_shared<UserDefinedIOParameter>(*this);
        bound->reader_ = storage.reader;
        bound->writer_ = storage.writer;
        return bound;
    }

    const ReaderPtr&
    GetReader() const {
        return reader_;
    }

    const WriterPtr&
    GetWriter() const {
        return writer_;
    }

private:
    ReaderPtr reader_;
    WriterPtr writer_;
};

}  // namespace vsag
