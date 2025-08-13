
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

#include <string.h>

#include "vsag/binaryset.h"
#include "vsag/readerset.h"

namespace fixtures {

class TestReader : public vsag::Reader {
public:
    TestReader(vsag::Binary binary) : binary_(std::move(binary)) {
    }

    void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        memcpy((char*)dest, binary_.data.get() + offset, len);
    }

    void
    AsyncRead(uint64_t offset, uint64_t len, void* dest, vsag::CallBack callback) override {
        memcpy((char*)dest, binary_.data.get() + offset, len);
        callback(vsag::IOErrorCode::IO_SUCCESS, "success");
    }

    [[nodiscard]] uint64_t
    Size() const override {
        return binary_.size;
    }

private:
    vsag::Binary binary_;
};

}  // namespace fixtures
