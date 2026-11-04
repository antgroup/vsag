// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <vsag/vsag.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#if __cplusplus != 201703L
#error "This consumer must compile as C++17, independently of the library standard"
#endif

namespace {
void
require(bool condition, const char* message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}
}  // namespace

int
main() {
    try {
        vsag::init();
        const std::string parameters = R"({"dtype":"float32","metric_type":"l2","dim":4,
            "index_param":{"base_quantization_type":"fp32","max_degree":8,
                           "ef_construction":32}})";
        const std::string search_parameters = R"({"hgraph":{"ef_search":32}})";
        int64_t ids[] = {10, 20, 30};
        float vectors[] = {0, 0, 0, 0, 1, 2, 3, 4, 8, 8, 8, 8};
        auto base = vsag::Dataset::Make();
        base->NumElements(3)->Dim(4)->Ids(ids)->Float32Vectors(vectors)->Owner(false);
        auto created = vsag::Factory::CreateIndex("hgraph", parameters);
        require(created.has_value(), "create index failed");
        auto index = created.value();
        auto built = index->Build(base);
        require(built.has_value() and built.value().empty(), "build failed");
        require(index->GetNumElements() == 3, "wrong element count");
        auto query = vsag::Dataset::Make();
        query->NumElements(1)->Dim(4)->Float32Vectors(vectors + 4)->Owner(false);
        auto check_search = [&](const vsag::IndexPtr& candidate) {
            auto result = candidate->KnnSearch(query, 1, search_parameters);
            require(result.has_value(), "search failed");
            require(result.value()->GetDim() == 1, "wrong result count");
            require(result.value()->GetIds()[0] == 20, "wrong nearest neighbor");
            require(std::abs(result.value()->GetDistances()[0]) < 1e-5F, "wrong distance");
        };
        check_search(index);
        auto serialized = index->Serialize();
        require(serialized.has_value(), "serialize failed");
        auto restored = vsag::Factory::CreateIndex("hgraph", parameters);
        require(restored.has_value(), "create restored index failed");
        require(restored.value()->Deserialize(serialized.value()).has_value(),
                "deserialize failed");
        require(restored.value()->GetNumElements() == 3, "wrong restored element count");
        check_search(restored.value());
        // Both indexes and their owned buffers are destroyed before successful process exit.
        std::cout << "Installed C++17 client passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
