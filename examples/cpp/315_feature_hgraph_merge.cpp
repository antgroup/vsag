
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

#include <iostream>

void
query_index(vsag::IndexPtr index,
            vsag::DatasetPtr query,
            const std::string& hgraph_search_parameters,
            int64_t topk) {
    auto result = index->KnnSearch(query, topk, hgraph_search_parameters).value();

    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        std::cout << result->GetIds()[i] << ": " << result->GetDistances()[i] << std::endl;
    }
}

int
main(int argc, char** argv) {
    vsag::init();

    /******************* Prepare Base Dataset *****************/
    int64_t num_vectors = 10000;
    int64_t dim = 128;
    int64_t index_count = 10;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> datas(num_vectors * dim);
    std::mt19937 rng(47);

    int64_t per_index_size = num_vectors / index_count;
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }

    /******************* Create HGraph Index *****************/
    std::string hgraph_build_parameters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "index_param": {
            "base_quantization_type": "sq8",
            "max_degree": 26,
            "ef_construction": 100
        }
    }
    )";

    /******************* Train HGraph Model *****************/
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto empty_index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();
    auto train_data = vsag::Dataset::Make();
    train_data->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->Owner(false);
    empty_index->Train(train_data);
    auto model = empty_index->ExportModel().value();

    /******************* Build HGraph Index *****************/
    std::vector<vsag::IndexPtr> indexes;
    for (int i = 0; i < index_count; ++i) {
        auto index = model->Clone().value();
        auto base = vsag::Dataset::Make();
        base->NumElements(per_index_size)
            ->Dim(dim)
            ->Ids(ids.data() + per_index_size * i)
            ->Float32Vectors(datas.data() + per_index_size * i * dim)
            ->Owner(false);
        if (auto build_result = index->Build(base); not build_result.has_value()) {
            std::cerr << "Failed to build index: internalError" << std::endl;
            exit(-1);
        }
        indexes.push_back(index);
    }

    /******************* Prepare Query Dataset *****************/
    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }
    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);
    auto hgraph_search_parameters = R"(
    {
        "hgraph": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;

    /******************* KnnSearch For HGraph Index Before Merge *****************/
    std::cout << "Before Merge:" << std::endl;
    query_index(indexes[0], query, hgraph_search_parameters, topk);

    /******************* Merge HGraph Index *****************/
    std::vector<vsag::MergeUnit> merge_units;
    for (int i = 1; i < index_count; ++i) {
        vsag::MergeUnit merge_unit{.index = indexes[i], .id_map_func = [](int64_t id) {
                                       return std::tuple<bool, int64_t>(true, id);
                                   }};
        merge_units.push_back(merge_unit);
    }
    indexes[0]->Merge(merge_units);

    /******************* KnnSearch For HGraph Index After Merge *****************/
    std::cout << "After Merge:" << std::endl;
    query_index(indexes[0], query, hgraph_search_parameters, topk);

    engine.Shutdown();
    return 0;
}
