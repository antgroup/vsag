
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

#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

#include "impl/logger/logger.h"
#include "impl/thread_pool/safe_thread_pool.h"
#include "typing.h"
#include "vsag/errors.h"

TEST_CASE("Create Index with Full Parameters", "[ut][factory]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    SECTION("hnsw") {
        auto parameters = vsag::JsonType::Parse(R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 512,
            "hnsw": {
                "max_degree": 16,
                "ef_construction": 100
            }
        }
        )");

        auto index = vsag::Factory::CreateIndex("hnsw", parameters.Dump());
        REQUIRE(index.has_value());
    }

    SECTION("diskann") {
        auto parameters = vsag::JsonType::Parse(R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 256,
            "diskann": {
                "ef_construction": 200,
                "max_degree": 16,
                "pq_dims": 32,
                "pq_sample_rate": 0.5
            }
        }
        )");

        auto index = vsag::Factory::CreateIndex("diskann", parameters.Dump());
        REQUIRE(index.has_value());
    }
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

TEST_CASE("Create HNSW with Incomplete Parameters", "[ut][factory]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    auto standard_parameters = vsag::JsonType::Parse(R"(
            {
                "dtype": "float32",
                "metric_type": "l2",
                "dim": 512,
                "hnsw": {
                    "max_degree": 16,
                    "ef_construction": 100
                }
            }
            )");

    SECTION("dtype is not provided") {
        standard_parameters.Erase("dtype");
        auto index = vsag::Factory::CreateIndex("hnsw", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("metric_type is not provided") {
        standard_parameters.Erase("metric_type");
        auto index = vsag::Factory::CreateIndex("hnsw", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("dim is not provided") {
        standard_parameters.Erase("dim");
        auto index = vsag::Factory::CreateIndex("hnsw", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("hnsw is not provided") {
        standard_parameters.Erase("hnsw");
        auto index = vsag::Factory::CreateIndex("hnsw", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("max_degree is not provided") {
        standard_parameters["hnsw"].Erase("max_degree");
        auto index = vsag::Factory::CreateIndex("hnsw", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("ef_construction is not provided") {
        standard_parameters["hnsw"].Erase("ef_construction");
        auto index = vsag::Factory::CreateIndex("hnsw", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("Create Diskann with Incomplete Parameters", "[ut][factory]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    auto standard_parameters = vsag::JsonType::Parse(R"(
            {
                "dim": 256,
                "dtype": "float32",
                "metric_type": "l2",
                "diskann": {
                    "max_degree": 16,
                    "ef_construction": 200,
                    "pq_dims": 32,
                    "pq_sample_rate": 0.5
                }
            }
            )");

    SECTION("dtype is not provided") {
        standard_parameters.Erase("dtype");
        auto index = vsag::Factory::CreateIndex("diskann", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("metric_type is not provided") {
        standard_parameters.Erase("metric_type");
        auto index = vsag::Factory::CreateIndex("diskann", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("dim is not provided") {
        standard_parameters.Erase("dim");
        auto index = vsag::Factory::CreateIndex("diskann", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("diskann is not provided") {
        standard_parameters.Erase("diskann");
        auto index = vsag::Factory::CreateIndex("diskann", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("max_degree is not provided") {
        standard_parameters["diskann"].Erase("max_degree");
        auto index = vsag::Factory::CreateIndex("diskann", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("ef_construction is not provided") {
        standard_parameters["diskann"].Erase("ef_construction");
        auto index = vsag::Factory::CreateIndex("diskann", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("pq_dims is not provided") {
        standard_parameters["diskann"].Erase("pq_dims");
        auto index = vsag::Factory::CreateIndex("diskann", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("pq_sample_rate is not provided") {
        standard_parameters["diskann"].Erase("pq_sample_rate");
        auto index = vsag::Factory::CreateIndex("diskann", standard_parameters.Dump());
        REQUIRE_FALSE(index.has_value());
        REQUIRE(index.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}
