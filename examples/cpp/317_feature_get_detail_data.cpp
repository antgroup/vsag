
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

std::string
transform_type(vsag::IndexDetailDataType type) {
    switch (type) {
        case vsag::IndexDetailDataType::TYPE_SCALAR_INT64:
            return "scalar_int64";
        case vsag::IndexDetailDataType::TYPE_SCALAR_DOUBLE:
            return "scalar_double";
        case vsag::IndexDetailDataType::TYPE_SCALAR_BOOL:
            return "scalar_bool";
        case vsag::IndexDetailDataType::TYPE_1DArray_INT64:
            return "1d_array_int64";
        case vsag::IndexDetailDataType::TYPE_2DArray_INT64:
            return "2d_array_int64";
        default:
            return "unknown";
    }
}

void
print_detail_info(const vsag::IndexDetailInfo& info) {
    std::cout << "{" << std::endl;
    std::cout << "    name: " << info.name << std::endl;
    std::cout << "    type: " << transform_type(info.type) << std::endl;
    std::cout << "    description: " << info.description << std::endl;
    std::cout << "}" << std::endl;
}

std::string
trans_2d_array_int64(const std::vector<std::vector<int64_t>>& array, int64_t raw, int64_t col) {
    std::string result = "[";

    for (int64_t i = 0; i < raw; ++i) {
        result += "[";
        for (int64_t j = 0; j < col; ++j) {
            result += std::to_string(array[i][j]) + ",";
        }
        result.pop_back();
        result += "],";
    }
    result.pop_back();
    result += "]";
    return result;
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
        ids[i] = i + 100;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        datas[i] = distrib_real(rng);
    }

    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(datas.data())
        ->Owner(false);

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

    /******************* Build HGraph *****************/
    vsag::Resource resource(vsag::Engine::CreateDefaultAllocator(), nullptr);
    vsag::Engine engine(&resource);
    auto index = engine.CreateIndex("hgraph", hgraph_build_parameters).value();
    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "After Build(), Index HGraph contains: " << index->GetNumElements()
                  << std::endl;
    } else if (build_result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "Failed to build index: internalError" << std::endl;
        exit(-1);
    }

    /********************* Get Index Detail Info *****************/
    if (auto detail_infos = index->GetIndexDetailInfos(); detail_infos.has_value()) {
        for (const auto& info : detail_infos.value()) {
            print_detail_info(info);
        }
    } else {
        std::cerr << "Failed to get index detail info: " << detail_infos.error().message
                  << std::endl;
        exit(-1);
    }

    /*********************** Get Index Detail Datas *****************/
    vsag::IndexDetailInfo info;
    if (auto detail_datas = index->GetDetailDataByName("num_elements", info);
        detail_datas.has_value()) {
        std::cout << "num_elements(type): " << transform_type(info.type) << std::endl;
        std::cout << "num_elements(value): " << detail_datas.value()->GetDataScalarInt64()
                  << std::endl;
    } else {
        std::cerr << "Failed to get index detail datas: " << detail_datas.error().message
                  << std::endl;
        exit(-1);
    }

    if (auto detail_datas = index->GetDetailDataByName("label_table", info);
        detail_datas.has_value()) {
        std::cout << "label_table(type): " << transform_type(info.type) << std::endl;
        std::cout << "label_table(value)[:10]: "
                  << trans_2d_array_int64(detail_datas.value()->GetData2DArrayInt64(), 10, 2)
                  << std::endl;
    } else {
        std::cerr << "Failed to get index detail datas: " << detail_datas.error().message
                  << std::endl;
        exit(-1);
    }

    engine.Shutdown();
    return 0;
}
