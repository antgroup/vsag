//
// Created by root on 25-4-15.
//

#include <vsag/vsag.h>
#include <fstream>
#include <iostream>
void ReadBin(const std::string& filename, std::vector<float>& data, int64_t total) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return;
    }

    data.resize(total);
    file.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
    if (file.fail()) {
        std::cout << "Failed to read file: " << filename << std::endl;
    }
    file.close();
}

void WriteJson(const std::string& filename, const std::string& json_str) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cout << "Failed to open file: " << filename << std::endl;
        return;
    }
    file << json_str;
    if (file.fail()) {
        std::cout << "Failed to write file: " << filename << std::endl;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <version>" << std::endl;
        return 1;
    }
    std::string version = argv[1];
    int64_t dim = 512;
    int64_t total = 10000;
    std::vector<float> data(total * dim);
    std::vector<int64_t> ids(total);
    std::iota(ids.begin(), ids.end(), 0);
    std::string filename = "../vsag_index/random_512d_10K.bin";
    ReadBin(filename, data, total * dim);
    auto base = vsag::Dataset::Make();
    base->NumElements(total)->Dim(dim)->Ids(ids.data())->Float32Vectors(data.data())
        ->Owner(false);

    auto hnsw_build_paramesters = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 512,
            "hnsw": {
                "max_degree": 32,
                "ef_construction": 300
            }
        }
    )";

    auto hgraph_build_paramesters = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 512,
            "index_param": {
                "base_quantization_type": "sq8",
                "max_degree": 64,
                "ef_construction": 300
            }
        }
    )";

    auto hnsw_search_paramesters = R"(
        {
            "hnsw": {
                "ef_search": 100
            }
        }
    )";

    auto hgraph_search_paramesters = R"(
        {
            "hgraph": {
                "ef_search": 100
            }
        }
    )";

    WriteJson(version + "_hnsw_build.json", hnsw_build_paramesters);
    WriteJson(version + "_hgraph_build.json", hgraph_build_paramesters);
    WriteJson(version + "_hnsw_search.json", hnsw_search_paramesters);
    WriteJson(version + "_hgraph_search.json", hgraph_search_paramesters);

    auto hnsw_index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();
    hnsw_index->Build(base);
    std::ofstream hnsw_file(version + "_hnsw.index", std::ios::binary);
    hnsw_index->Serialize(hnsw_file);

    auto hgraph_index = vsag::Factory::CreateIndex("hgraph", hgraph_build_paramesters).value();
    hgraph_index->Build(base);
    std::ofstream hgraph_file(version + "_hgraph.index", std::ios::binary);
    hgraph_index->Serialize(hgraph_file);
}