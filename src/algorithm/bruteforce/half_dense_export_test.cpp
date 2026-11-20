// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "algorithm/hgraph/hgraph.h"
#include "algorithm/pyramid/pyramid.h"
#include "half_dense_test_fixture.h"

namespace {
vsag::InnerIndexPtr
ExportIndex(bool pyramid, bool native, bool bf16, int mode) {
    vsag::IndexCommonParam common;
    common.dim_ = 64;
    common.data_type_ =
        native ? (bf16 ? vsag::DataTypes::DATA_TYPE_BF16 : vsag::DataTypes::DATA_TYPE_FP16)
               : vsag::DataTypes::DATA_TYPE_FLOAT;
    common.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    // The index retains a raw allocator pointer; keep its shared owner alive.
    static const auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    common.allocator_ = allocator;
    auto json = vsag::JsonType::Parse(
        R"({"base_quantization_type":"fp32","max_degree":8,"ef_construction":32,"build_thread_count":1})");
    json["store_raw_vector"].SetBool(true);
    if (mode != 0) {
        json["base_quantization_type"].SetString("rabitq");
        json["precise_quantization_type"].SetString("rabitq");
        json["rabitq_bits_per_dim_base"].SetInt(2);
        json["rabitq_bits_per_dim_precise"].SetInt(6);
        json["use_reorder"].SetBool(true);
        json["store_raw_vector"].SetBool(mode == 1);
    }
    if (pyramid) {
        json["hierarchies"].SetJson(vsag::JsonType::Parse(R"(["site"])"));
        json["index_min_size"].SetInt(8);
        auto param = vsag::Pyramid::CheckAndMappingExternalParam(json, common);
        auto result = std::make_shared<vsag::Pyramid>(param, common);
        result->InitFeatures();
        return result;
    }
    json["graph_type"].SetString("nsw");
    if (mode >= 2) {
        json["reorder_source"].SetString("base");
        json["base_io_type"].SetString("memory_io");
        json["base_supplement_io_type"].SetString("memory_io");
    }
    if (mode == 3) {
        json["rabitq_fused_datacell"].SetBool(true);
        json["graph_storage_type"].SetString("flat");
    }
    if (mode == 4) {
        json["support_duplicate"].SetBool(true);
        json["deduplicate_storage"].SetBool(true);
    }
    auto param = vsag::HGraph::CheckAndMappingExternalParam(json, common);
    auto result = std::make_shared<vsag::HGraph>(param, common);
    result->InitFeatures();
    return result;
}
}  // namespace

TEST_CASE("Public Float32 row bridge respects native and split decode sources",
          "[ut][half_input][half_export]") {
    constexpr int64_t dim = 64;
    constexpr int64_t count = 64;
    for (bool pyramid : {false, true}) {
        for (bool bf16 : {false, true}) {
            for (int mode = 0; mode <= (pyramid ? 2 : 4); ++mode) {
                CAPTURE(pyramid, bf16, mode);
                auto actual = ExportIndex(pyramid, true, bf16, mode);
                auto reference = ExportIndex(pyramid, false, bf16, mode);
                std::vector<int64_t> ids(count);
                std::vector<uint16_t> half(count * dim);
                std::vector<float> values(count * dim);
                std::vector<std::string> paths(count, "tenant/leaf");
                for (int64_t i = 0; i < count; ++i) {
                    ids[i] = 1000 + i;
                    for (int64_t j = 0; j < dim; ++j) {
                        const uint16_t bits = static_cast<uint16_t>((bf16 ? 0x3F00 : 0x3800) +
                                                                    ((i * 31 + j * 17) % 120));
                        half[i * dim + j] = bits;
                        values[i * dim + j] = half_dense_test::Decode(bits, bf16);
                    }
                }
                auto native = vsag::Dataset::Make()
                                  ->Dim(dim)
                                  ->NumElements(count)
                                  ->Ids(ids.data())
                                  ->Float16Vectors(half.data())
                                  ->Paths("site", paths.data())
                                  ->Owner(false);
                auto fp32 = vsag::Dataset::Make()
                                ->Dim(dim)
                                ->NumElements(count)
                                ->Ids(ids.data())
                                ->Float32Vectors(values.data())
                                ->Paths("site", paths.data())
                                ->Owner(false);
                REQUIRE(actual->Build(native).empty());
                REQUIRE(reference->Build(fp32).empty());
                if (mode == 2 or mode == 4) {
                    // Legal ordinary split storage has filter+supplement >= 2
                    // total bits. Existing DecodeOne only supports total bits=1.
                    std::vector<float> rejected(dim);
                    REQUIRE_THROWS(actual->GetFloat32VectorByInnerId(0, rejected.data()));
                    auto binary = actual->Serialize();
                    actual = ExportIndex(pyramid, true, bf16, mode);
                    actual->Deserialize(binary);
                    REQUIRE_THROWS(actual->GetFloat32VectorByInnerId(0, rejected.data()));
                    continue;
                }

                std::vector<float> lhs(dim);
                std::vector<float> rhs(dim);
                // Inner id zero is the first inserted row. Check the new boundary
                // directly even when a lossy index intentionally has no raw-vector API.
                actual->GetFloat32VectorByInnerId(0, lhs.data());
                reference->GetFloat32VectorByInnerId(0, rhs.data());
                for (int64_t j = 0; j < dim; ++j) {
                    REQUIRE(std::isfinite(lhs[j]));
                    REQUIRE(std::abs(lhs[j] - rhs[j]) < (mode >= 2 ? 0.15F : 0.01F));
                }
                // Same-index ABI oracle avoids independently trained RaBitQ models:
                // selected fused native decode already returns Float32.
                if (mode >= 2) {
                    std::vector<float> same_source(dim);
                    actual->GetVectorByInnerId(0, same_source.data());
                    for (int64_t j = 0; j < dim; ++j) {
                        REQUIRE(lhs[j] == same_source[j]);
                    }
                }
                if (mode <= 1) {
                    auto exported = actual->GetDataByIds(ids.data(), 1);
                    REQUIRE(exported->GetFloat32Vectors() != nullptr);
                    for (int64_t j = 0; j < dim; ++j) {
                        REQUIRE(std::abs(exported->GetFloat32Vectors()[j] - values[j]) < 0.01F);
                    }
                }
                if (not pyramid and mode <= 1) {
                    auto same_native = vsag::Dataset::Make()
                                           ->Dim(dim)
                                           ->NumElements(1)
                                           ->Float16Vectors(half.data())
                                           ->Owner(false);
                    auto same_fp32 = vsag::Dataset::Make()
                                         ->Dim(dim)
                                         ->NumElements(1)
                                         ->Float32Vectors(values.data())
                                         ->Owner(false);
                    // force_update=false exercises the existing native Decode->distance ABI.
                    REQUIRE(actual->UpdateVector(ids[0], same_native, false));
                    REQUIRE(reference->UpdateVector(ids[0], same_fp32, false));
                }
                auto binary = actual->Serialize();
                actual = ExportIndex(pyramid, true, bf16, mode);
                actual->Deserialize(binary);
                actual->GetFloat32VectorByInnerId(0, lhs.data());
                for (int64_t j = 0; j < dim; ++j) {
                    REQUIRE(std::isfinite(lhs[j]));
                    REQUIRE(std::abs(lhs[j] - rhs[j]) < (mode >= 2 ? 0.15F : 0.01F));
                }
            }
        }
    }
}
