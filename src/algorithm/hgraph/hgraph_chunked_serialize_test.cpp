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

#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "hgraph.h"
#include "impl/allocator/safe_allocator.h"
#include "index/index_impl.h"
#include "index_common_param.h"
#include "unittest.h"
#include "vsag/serialize_writer.h"

namespace {

// plain writer: verbatim passthrough, codec "none"
class BufferSerializeWriter : public vsag::SerializeWriter {
public:
    void
    Write(const char* data, uint64_t size) override {
        buffer_.append(data, size);
    }

    std::string buffer_;
};

vsag::IndexCommonParam
MakeCommonParam(int64_t dim) {
    vsag::IndexCommonParam common_param;
    common_param.dim_ = dim;
    common_param.metric_ = vsag::MetricType::METRIC_TYPE_L2SQR;
    common_param.data_type_ = vsag::DataTypes::DATA_TYPE_FLOAT;
    common_param.allocator_ = vsag::SafeAllocator::FactoryDefaultAllocator();
    return common_param;
}

vsag::JsonType
MakeHGraphJson() {
    return vsag::JsonType::Parse(R"({
        "base_quantization_type": "fp32",
        "max_degree": 8,
        "ef_construction": 32,
        "build_thread_count": 1
    })");
}

std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>
MakeHGraphIndex(const vsag::IndexCommonParam& common_param) {
    return std::make_shared<vsag::IndexImpl<vsag::HGraph>>(MakeHGraphJson(), common_param);
}

struct TestData {
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    int64_t dim{0};
    int64_t count{0};
};

TestData
MakeTestData(int64_t dim, int64_t count) {
    TestData data;
    data.dim = dim;
    data.count = count;
    data.vectors.resize(dim * count);
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (auto& v : data.vectors) {
        v = dist(rng);
    }
    for (int64_t i = 0; i < count; ++i) {
        data.ids.push_back(i);
    }
    return data;
}

vsag::DatasetPtr
MakeDataset(TestData& data) {
    auto dataset = vsag::Dataset::Make();
    dataset->NumElements(data.count)
        ->Dim(data.dim)
        ->Ids(data.ids.data())
        ->Float32Vectors(data.vectors.data())
        ->Owner(false);
    return dataset;
}

// knn results of both indexes must match on every sampled base vector as query
void
RequireSameKnnResults(const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& expected,
                      const std::shared_ptr<vsag::IndexImpl<vsag::HGraph>>& actual,
                      TestData& data) {
    const auto* params = R"({"hgraph": {"ef_search": 64}})";
    for (int64_t i = 0; i < data.count; i += 17) {
        auto query = vsag::Dataset::Make();
        query->NumElements(1)
            ->Dim(data.dim)
            ->Float32Vectors(data.vectors.data() + i * data.dim)
            ->Owner(false);
        auto expected_result = expected->KnnSearch(query, 5, params);
        auto actual_result = actual->KnnSearch(query, 5, params);
        REQUIRE(expected_result.has_value());
        REQUIRE(actual_result.has_value());
        REQUIRE(expected_result.value()->GetDim() == actual_result.value()->GetDim());
        for (int64_t k = 0; k < expected_result.value()->GetDim(); ++k) {
            REQUIRE(expected_result.value()->GetIds()[k] == actual_result.value()->GetIds()[k]);
        }
    }
}

}  // namespace

TEST_CASE("HGraph Chunked Plain File Readable by Existing Deserialize",
          "[ut][hgraph][chunked_serialize]") {
    auto data = MakeTestData(32, 200);
    auto common_param = MakeCommonParam(data.dim);
    auto index = MakeHGraphIndex(common_param);
    REQUIRE(index->Build(MakeDataset(data)).has_value());

    uint64_t chunk_size = 0;
    SECTION("small chunk size forces multiple chunks") {
        chunk_size = 4096;
    }
    SECTION("default chunk size keeps single chunks") {
        chunk_size = vsag::DEFAULT_SERIALIZE_CHUNK_SIZE;
    }

    // with a plain writer the body bytes equal the sequential format, and the
    // extra footer keys (chunked layout) must be ignored by the existing
    // reader, so the whole file stays readable in place
    BufferSerializeWriter writer;
    REQUIRE(index->Serialize(writer, chunk_size).has_value());

    std::istringstream file_stream(writer.buffer_, std::ios::in | std::ios::binary);
    auto loaded = MakeHGraphIndex(MakeCommonParam(data.dim));
    REQUIRE(loaded->Deserialize(file_stream).has_value());
    REQUIRE(loaded->GetNumElements() == data.count);

    RequireSameKnnResults(index, loaded, data);
}
