
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

#include <fstream>
#include <iostream>

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 1000;
    int64_t dim = 128;
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    auto base = vsag::Dataset::Make();
    // Transfer the ownership of the data (ids, vectors) to the base.
    base->NumElements(num_vectors)->Dim(dim)->Ids(ids)->Float32Vectors(vectors);

    /******************* Create HGraph Index (base_quantization = sq4) *****************/
    auto hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "max_degree": 32,
            "ef_construction": 100,
            "store_raw_vector": true,
            "base_quantization_type": "sq4"
        }
    }
    )";
    auto index1 = vsag::Factory::CreateIndex("hgraph", hgraph_build_parameters).value();

    /******************* Build Index *****************/
    index1->Build(base);

    /******************* Tune the Index into HGraph Index (base_quantization = sq8) *****************/
    auto alter_hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "max_degree": 32,
            "ef_construction": 100,
            "store_raw_vector": true,
            "base_quantization_type": "sq8"
        }
    }
    )";
    index1->Tune(alter_hgraph_build_parameters);

    /******************* Serialize with sq8 *****************/
    auto index_path = "/tmp/vsag-persistent-streaming-tune-index.index";
    std::ofstream out_stream(index_path);
    auto serialize_result = index1->Serialize(out_stream);
    out_stream.close();

    /******************* index1 is equal to index2 after Deserialize index2 *****************/
    auto index2 = vsag::Factory::CreateIndex("hgraph", alter_hgraph_build_parameters).value();
    std::ifstream in_stream(index_path);
    index2->Deserialize(in_stream);
    in_stream.close();

    return 0;
}
