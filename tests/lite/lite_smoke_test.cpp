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

#include <vsag/vsag.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t DIM = 8;

void
Expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

vsag::DatasetPtr
MakeDataset(int64_t first_id,
            int64_t count,
            std::vector<int64_t>& ids,
            std::vector<float>& vectors) {
    ids.resize(static_cast<size_t>(count));
    vectors.resize(static_cast<size_t>(count * DIM));
    for (int64_t row = 0; row < count; ++row) {
        ids[static_cast<size_t>(row)] = first_id + row;
        for (int64_t col = 0; col < DIM; ++col) {
            vectors[static_cast<size_t>(row * DIM + col)] =
                static_cast<float>((first_id + row) * 0.1 + col);
        }
    }
    return vsag::Dataset::Make()
        ->NumElements(count)
        ->Dim(DIM)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false);
}

vsag::DatasetPtr
MakeQuery(const float* vector) {
    return vsag::Dataset::Make()->NumElements(1)->Dim(DIM)->Float32Vectors(vector)->Owner(false);
}

std::string
BuildParameters() {
    return R"({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 8,
        "lazy_hgraph": {
            "transition_threshold": 4,
            "hgraph": {
                "base_quantization_type": "fp32",
                "max_degree": 8,
                "ef_construction": 32,
                "build_thread_count": 1,
                "support_force_remove": true
            }
        }
    })";
}

const char*
SearchParameters() {
    return R"({"hgraph":{"ef_search":32}})";
}

void
ExpectNearest(const vsag::IndexPtr& index, const float* query, int64_t expected_id) {
    auto result = index->KnnSearch(MakeQuery(query), 1, SearchParameters());
    Expect(result.has_value(), "KNN search failed");
    Expect(result.value()->GetIds()[0] == expected_id, "unexpected nearest-neighbor id");
}

}  // namespace

int
main() {
    try {
        vsag::init();

        auto created = vsag::Factory::CreateIndex("lazy_hgraph", BuildParameters());
        Expect(created.has_value(), "failed to create LazyHGraph");
        auto index = created.value();

        std::vector<int64_t> ids;
        std::vector<float> vectors;
        auto dataset = MakeDataset(100, 6, ids, vectors);
        auto add_result = index->Add(dataset);
        Expect(add_result.has_value(), "Add failed");
        Expect(add_result.value().empty(), "Add rejected one or more vectors");
        Expect(index->GetNumElements() == 6, "unexpected element count after Add");
        ExpectNearest(index, vectors.data(), 100);

        auto update_id = index->UpdateId(100, 900);
        Expect(update_id.has_value() && update_id.value(), "UpdateId failed");

        std::vector<int64_t> replacement_ids{900};
        std::vector<float> replacement(static_cast<size_t>(DIM), -3.0F);
        auto replacement_dataset = vsag::Dataset::Make()
                                       ->NumElements(1)
                                       ->Dim(DIM)
                                       ->Ids(replacement_ids.data())
                                       ->Float32Vectors(replacement.data())
                                       ->Owner(false);
        auto update_vector = index->UpdateVector(900, replacement_dataset, true);
        Expect(update_vector.has_value() && update_vector.value(), "UpdateVector failed");
        ExpectNearest(index, replacement.data(), 900);

        auto remove_result = index->Remove(101, vsag::RemoveMode::MARK_REMOVE);
        Expect(remove_result.has_value() && remove_result.value() == 1, "Remove failed");
        Expect(index->GetNumElements() == 5, "unexpected element count after Remove");

        auto binary = index->Serialize();
        Expect(binary.has_value(), "Serialize failed");
        auto restored = vsag::Factory::CreateIndex("lazy_hgraph", BuildParameters());
        Expect(restored.has_value(), "failed to create restore target");
        auto deserialize_result = restored.value()->Deserialize(binary.value());
        Expect(deserialize_result.has_value(), "Deserialize failed");
        const auto restored_count = restored.value()->GetNumElements();
        Expect(restored_count == 5,
               "unexpected element count after Deserialize: " + std::to_string(restored_count));
        Expect(!restored.value()->CheckIdExist(101), "marked deletion was not restored");
        ExpectNearest(restored.value(), replacement.data(), 900);

        auto excluded = vsag::Factory::CreateIndex("ivf", R"({"dtype":"float32"})");
        Expect(!excluded.has_value(), "Lite build unexpectedly exposed IVF");

        std::cout << "VSAG Lite smoke test passed: CRUD, search, and persistence" << std::endl;
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "VSAG Lite smoke test failed: " << exception.what() << std::endl;
        return 1;
    }
}
