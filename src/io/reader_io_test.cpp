
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

#include "reader_io.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "basic_io_test.h"
#include "safe_allocator.h"


class TestReader: public vsag::Reader {
public:
    TestReader(uint8_t* data): data_(data) {}

    void Read(uint64_t offset, uint64_t len, void *dest) override {
        memcpy(dest, data_ + offset, len);
    }

    void AsyncRead(uint64_t offset, uint64_t len, void *dest, vsag::CallBack callback) override {
        Read(offset, len, dest);
        callback(vsag::IOErrorCode::IO_SUCCESS, "success");
    }
private:
    const uint8_t* data_{nullptr};
};


TEST_CASE("ReaderIO Read Test", "[ut][ReaderIO]") {

}