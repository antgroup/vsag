
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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "algorithm/bruteforce/bruteforce.h"
#include "algorithm/inner_index_interface.h"
#include "data_type.h"
#include "datacell/flatten_datacell_parameter.h"
#include "framework/test_dataset.h"
#include "framework/test_dataset_pool.h"
#include "framework/test_logger.h"
#include "impl/allocator/default_allocator.h"
#include "index_common_param.h"
#include "test_index.h"
#include "typing.h"
#include "vsag/constants.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/index.h"
#include "vsag/vsag.h"

using namespace vsag;

struct WarpParam {
    std::string base_quantization_type = "fp32";
    std::string base_io_type = "memory_io";
};

namespace fixtures {
class WarpTestIndex : public fixtures::TestIndex {
public:
    static std::string
    GenerateWarpBuildParametersString(const std::string& metric_type,
                                      int64_t dim,
                                      const WarpParam& param);

    static std::string
    GenerateWarpSearchParametersString();

    static TestDatasetPool pool;

    static std::vector<int> dims;

    constexpr static uint64_t base_count = 1000;

    constexpr static const char* search_param_tmp = R"(
        {{
            "warp": {{
            }}
        }})";
};

TestDatasetPool WarpTestIndex::pool{};
std::vector<int> WarpTestIndex::dims = fixtures::get_common_used_dims(1, RandomValue(0, 999));

std::string
WarpTestIndex::GenerateWarpBuildParametersString(const std::string& metric_type,
                                                 int64_t dim,
                                                 const WarpParam& param) {
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "{}",
            "base_io_type": "{}"
        }}
    }}
    )";
    auto build_parameters_str = fmt::format(
        parameter_temp, metric_type, dim, param.base_quantization_type, param.base_io_type);
    return build_parameters_str;
}

std::string
WarpTestIndex::GenerateWarpSearchParametersString() {
    return fmt::format(search_param_tmp);
}

}  // namespace fixtures

TEST_CASE_PERSISTENT_FIXTURE(fixtures::WarpTestIndex, "Warp Add Test", "[ft][warp]") {
    auto metric_type = GENERATE("ip");
    std::string base_quantization_str = GENERATE("fp32");
    WarpParam warp_param;
    warp_param.base_quantization_type = base_quantization_str;
    const std::string name = "warp";
    auto search_param = GenerateWarpSearchParametersString();
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GenerateWarpBuildParametersString(metric_type, dim, warp_param);
        auto index = TestFactory(name, param, true);
        REQUIRE(index->GetIndexType() == vsag::IndexType::WARP);
        auto dataset =
            pool.GetDatasetAndCreate(dim, base_count, metric_type, false, 0.8, 0, 16, "multi");
        TestAddIndex(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
        TestRangeSearch(index, dataset, search_param, 0.99, 10, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::WarpTestIndex,
                             "Warp Serialize File",
                             "[ft][warp][serialization]") {
    auto origin_size = vsag::Options::Instance().block_size_limit();
    auto size = GENERATE(1024 * 1024 * 2);
    auto metric_type = GENERATE("ip");
    std::string base_quantization_str = GENERATE("fp32");
    WarpParam warp_param;
    warp_param.base_quantization_type = base_quantization_str;
    const std::string name = "warp";
    auto search_param = GenerateWarpSearchParametersString();
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        vsag::Options::Instance().set_block_size_limit(size);
        auto param = GenerateWarpBuildParametersString(metric_type, dim, warp_param);
        auto index = TestFactory(name, param, true);
        SECTION("serialize empty index") {
            auto index2 = TestFactory(name, param, true);
            auto serialize_binary = index->Serialize();
            REQUIRE(serialize_binary.has_value());
            auto deserialize_index = index2->Deserialize(serialize_binary.value());
            REQUIRE(deserialize_index.has_value());
        }
        auto dataset =
            pool.GetDatasetAndCreate(dim, base_count, metric_type, false, 0.8, 0, 16, "multi");
        TestBuildIndex(index, dataset, true);
        SECTION("serialize/deserialize by binary") {
            auto index2 = TestFactory(name, param, true);
            TestSerializeBinarySet(index, index2, dataset, search_param, true);
        }
        SECTION("serialize/deserialize by readerset") {
            auto index2 = TestFactory(name, param, true);
            TestSerializeReaderSet(index, index2, dataset, search_param, name, true);
        }
        SECTION("serialize/deserialize by file") {
            auto index2 = TestFactory(name, param, true);
            TestSerializeFile(index, index2, dataset, search_param, true);
        }
    }
    vsag::Options::Instance().set_block_size_limit(origin_size);
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::WarpTestIndex, "Warp IP Multiple Dims", "[ft][warp]") {
    WarpParam warp_param;
    warp_param.base_quantization_type = "fp32";
    const std::string name = "warp";
    auto search_param = GenerateWarpSearchParametersString();
    for (auto dim : {16, 128, 256}) {
        auto param = GenerateWarpBuildParametersString("ip", dim, warp_param);
        auto index = TestFactory(name, param, true);
        auto dataset = pool.GetDatasetAndCreate(dim, base_count, "ip", false, 0.8, 0, 16, "multi");
        TestBuildIndex(index, dataset, true);
        TestKnnSearch(index, dataset, search_param, 0.99, true);
    }
}

TEST_CASE_PERSISTENT_FIXTURE(fixtures::WarpTestIndex, "Warp Mark Remove", "[ft][remove][warp]") {
    auto metric_type = GENERATE("ip");
    std::string base_quantization_str = GENERATE("fp32");
    WarpParam warp_param;
    warp_param.base_quantization_type = base_quantization_str;
    const std::string name = "warp";
    auto search_param = GenerateWarpSearchParametersString();
    for (auto& dim : dims) {
        INFO(fmt::format("metric_type={}, dim={}", metric_type, dim));
        auto param = GenerateWarpBuildParametersString(metric_type, dim, warp_param);
        auto index = TestFactory(name, param, true);
        auto dataset =
            pool.GetDatasetAndCreate(dim, base_count, metric_type, false, 0.8, 0, 16, "multi");
        TestBuildIndex(index, dataset, true);

        auto base_num = dataset->base_->GetNumElements();
        const auto* ids = dataset->base_->GetIds();
        REQUIRE(index->GetNumElements() == base_num);
        REQUIRE(index->GetNumberRemoved() == 0);

        // FORCE_REMOVE is not supported by WARP
        auto force_result = index->Remove(ids[0], vsag::RemoveMode::FORCE_REMOVE);
        REQUIRE_FALSE(force_result.has_value());

        // mark remove half of the base data
        int64_t remove_count = base_num / 2;
        std::vector<int64_t> remove_ids(ids, ids + remove_count);
        auto remove_result = index->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);
        REQUIRE(remove_result.has_value());
        REQUIRE(remove_result.value() == remove_count);
        REQUIRE(index->GetNumElements() == base_num - remove_count);
        REQUIRE(index->GetNumberRemoved() == remove_count);

        // removing the same ids again should remove nothing
        auto duplicate_remove = index->Remove(remove_ids, vsag::RemoveMode::MARK_REMOVE);
        REQUIRE(duplicate_remove.has_value());
        REQUIRE(duplicate_remove.value() == 0);

        // removing an id that is not present is a no-op
        std::vector<int64_t> absent_ids = {base_num + 10000};
        auto absent_remove = index->Remove(absent_ids, vsag::RemoveMode::MARK_REMOVE);
        REQUIRE(absent_remove.has_value());
        REQUIRE(absent_remove.value() == 0);
        REQUIRE(index->GetNumElements() == base_num - remove_count);
        REQUIRE(index->GetNumberRemoved() == remove_count);

        // removed ids must not appear in search results
        for (int64_t i = 0; i < remove_count; ++i) {
            auto query = fixtures::get_one_query(dataset->base_, static_cast<int>(i));
            auto search_result = index->KnnSearch(query, 10, search_param);
            REQUIRE(search_result.has_value());
            for (int64_t j = 0; j < search_result.value()->GetDim(); ++j) {
                REQUIRE(search_result.value()->GetIds()[j] != ids[i]);
            }
        }
    }
}

// Concurrent, discriminating regression test for the WARP add-document race.
//
// Bug (historical): add_one_doc read the running vector count and wrote
// doc offsets as two non-atomic steps. With build_thread_count > 1 each
// document's add_func runs on a thread-pool worker, so two concurrent docs
// could read the same stale start index and record wrong offsets.
//
// This test verifies that concurrent multi-vector Build with build_thread_count > 1
// preserves correct per-doc vector mapping via the BruteForce multi-vector
// (WARP) implementation.
//
// Discrimination: every document with label L is filled with the constant
// fingerprint fp(L) = float(L) in all dims (stored exactly by the fp32
// quantizer, no normalization). After a concurrent Build, GetVectorByIds(L)
// must return a vector whose value rounds back to L.
TEST_CASE("Warp concurrent multi-doc add preserves per-doc vector mapping",
          "[ut][warp][concurrent]") {
    const int64_t dim = 32;
    const uint64_t num_docs = 600;       // many docs -> more interleavings
    const uint32_t build_threads = 8;    // > 1 enables the concurrent path
    const int iterations = 8;            // repeat to exercise nondeterminism

    // Create a WARP (BruteForce multi-vector) index via factory with
    // build_thread_count > 1 to exercise concurrent add path.
    auto build_param_str = fmt::format(R"({{
        "dtype": "float32",
        "metric_type": "l2",
        "dim": {},
        "index_param": {{
            "base_quantization_type": "fp32",
            "base_io_type": "memory_io",
            "build_thread_count": {}
        }}
    }})", dim, build_threads);

    for (int iter = 0; iter < iterations; ++iter) {
        auto index_result = vsag::Factory::CreateIndex("warp", build_param_str);
        REQUIRE(index_result.has_value());
        auto index = index_result.value();

        // Doc with label L (= i + 1) has a variable number of sub-vectors, every
        // sub-vector filled with float(L) in all dims. Distinct labels -> distinct
        // fingerprints, so any mis-mapped offset entry is detectable.
        std::vector<int64_t> labels(num_docs);
        std::vector<std::vector<float>> storage(num_docs);
        std::vector<vsag::MultiVector> mvs(num_docs);
        std::mt19937 gen(20260617U + static_cast<uint32_t>(iter));
        std::uniform_int_distribution<uint32_t> len_dist(1, 7);
        for (uint64_t i = 0; i < num_docs; ++i) {
            int64_t label = static_cast<int64_t>(i + 1);
            uint32_t len = len_dist(gen);
            labels[i] = label;
            storage[i].assign(static_cast<size_t>(len) * static_cast<size_t>(dim),
                              static_cast<float>(label));
            mvs[i].len_ = len;
            mvs[i].vectors_ = storage[i].data();
        }

        auto dataset = vsag::Dataset::Make();
        dataset->NumElements(static_cast<int64_t>(num_docs))
            ->Dim(dim)
            ->MultiVectorDim(dim)
            ->Ids(labels.data())
            ->MultiVectors(mvs.data())
            ->Owner(false);

        auto build_result = index->Build(dataset);
        REQUIRE(build_result.has_value());
        REQUIRE(index->GetNumElements() == static_cast<int64_t>(num_docs));

        // Discriminating check: each label must map back to its own fingerprint.
        int mismatches = 0;
        int64_t first_bad_label = -1;
        int64_t first_bad_got = -1;
        for (uint64_t i = 0; i < num_docs; ++i) {
            int64_t label = labels[i];
            auto got = index->GetVectorByIds(&label, 1, nullptr);
            REQUIRE(got.has_value());
            const float* v = got.value()->GetFloat32Vectors();
            REQUIRE(v != nullptr);
            bool ok = true;
            for (int d = 0; d < dim; ++d) {
                if (std::lround(v[d]) != label) {
                    ok = false;
                    break;
                }
            }
            if (not ok) {
                if (mismatches == 0) {
                    first_bad_label = label;
                    first_bad_got = std::lround(v[0]);
                }
                ++mismatches;
            }
        }
        INFO("iteration=" << iter << " mismatches=" << mismatches << " first_bad_label="
                          << first_bad_label << " first_bad_got=" << first_bad_got);
        REQUIRE(mismatches == 0);
    }
}
