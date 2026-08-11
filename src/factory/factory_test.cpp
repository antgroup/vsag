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

#include "vsag/factory.h"

#include <fstream>
#include <nlohmann/json.hpp>

#include "impl/logger/logger.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "typing.h"
#include "unittest.h"
#include "vsag/errors.h"

TEST_CASE("Factory reports removed indexes as unsupported", "[ut][factory]") {
    for (const auto* name : {"hnsw", "fresh_hnsw", "diskann"}) {
        auto index =
            vsag::Factory::CreateIndex(name, R"({"dtype":"float32","metric_type":"l2","dim":4})");
        INFO(name);
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::UNSUPPORTED_INDEX);
    }
}

TEST_CASE("Factory creates SINDI V2", "[ut][factory]") {
    auto parameters = vsag::JsonType::Parse(R"(
        {
            "dtype": "sparse",
            "metric_type": "ip",
            "dim": 256,
            "index_param": {
                "term_id_limit": 1000,
                "window_size": 10000,
                "doc_prune_ratio": 0.0,
                "use_quantization": false,
                "use_reorder": false,
                "term_io": {
                    "type": "memory_io"
                }
            }
        }
        )");

    auto index = vsag::Factory::CreateIndex("sindi_v2", parameters.Dump());
    REQUIRE(index.has_value());
    REQUIRE(index.value()->GetIndexType() == vsag::IndexType::SINDI_V2);
}

TEST_CASE("Create Local File Reader", "[ut][factory]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    const std::string filename = "/tmp/test_local_file_reader.bin";
    {
        std::ofstream file(filename, std::ios::binary);
        const std::string content = "HelloWorldTestData";
        file.write(content.c_str(), content.size());
        file.close();
    }

    SECTION("Sync read without offset") {
        auto reader = vsag::Factory::CreateLocalFileReader(filename, 0, 18);
        char buffer[6] = {0};

        reader->Read(0, 5, buffer);
        REQUIRE(std::string(buffer) == "Hello");

        reader->Read(5, 5, buffer);
        REQUIRE(std::string(buffer) == "World");
    }

    SECTION("Sync read with base offset") {
        auto reader = vsag::Factory::CreateLocalFileReader(filename, 5, 5);
        char buffer[6] = {0};

        reader->Read(0, 5, buffer);
        REQUIRE(std::string(buffer) == "World");
    }

    SECTION("Async read without explicit pool") {
        auto reader = vsag::Factory::CreateLocalFileReader(filename, 10, 4);
        char buffer[5] = {0};
        std::promise<void> completion_promise;
        auto completion_future = completion_promise.get_future();
        bool callback_called = false;

        reader->AsyncRead(0, 4, buffer, [&](vsag::IOErrorCode code, const std::string& msg) {
            REQUIRE(code == vsag::IOErrorCode::IO_SUCCESS);
            REQUIRE(msg == "success");
            callback_called = true;
            completion_promise.set_value();
        });

        auto status = completion_future.wait_for(std::chrono::seconds(1));
        REQUIRE(status == std::future_status::ready);

        REQUIRE(callback_called);
        REQUIRE(std::string(buffer) == "Test");
    }

    SECTION("Check size calculation") {
        auto reader1 = vsag::Factory::CreateLocalFileReader(filename, 0, 18);
        REQUIRE(reader1->Size() == 18);

        auto reader2 = vsag::Factory::CreateLocalFileReader(filename, 5, 5);
        REQUIRE(reader2->Size() == 5);
    }
    std::remove(filename.c_str());
}
