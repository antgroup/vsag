#include <fmt/format.h>
#include <omp.h>
#include <vsag/vsag.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <cstring>

#include "typing.h"


const void * open_file(const char* file_name) {
    int fd = open(file_name, O_RDONLY);
    if (fd == -1) {
        perror("open");
        exit(-1);
    }

    // 获取文件大小
    off_t file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);

    // 映射文件
    void* mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        perror("mmap");
        close(fd);
        exit(-1);
    }
    return mapped;
}


std::vector<char> read_file(const char* file_name, int row, int col, int code_size) {
    std::vector<char> buffer;
    buffer.resize(row * col * code_size + 2 * sizeof(int32_t));
    std::fstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_name << std::endl;
        exit(-1);
    }
    file.read(buffer.data(), row * col * code_size +  + 2 * sizeof(int32_t));
    return buffer;
}

int getIntersectionSize(const int32_t* a32, const int64_t* a64, size_t topk) {
    // Step 1: Convert int32_t array to a set of int64_t
    std::unordered_set<int64_t> setA;
    for (size_t i = 0; i < topk; ++i) {
        setA.insert(static_cast<int64_t>(a32[i]));
    }

    // Step 2: Convert int64_t array to a set
    std::unordered_set<int64_t> setB;
    for (size_t i = 0; i < topk; ++i) {
        setB.insert(a64[i]);
    }

    // Step 3: Count the number of common elements
    int count = 0;
    for (const auto& val : setA) {
        if (setB.find(val) != setB.end()) {
            ++count;
        }
    }
    return count;
}

int
main(int argc, char** argv) {
    /******************* Prepare Base Dataset *****************/
    constexpr static const char * file_path = "hnsw_index_{}";
    std::string query_file = "/root/vsag/data/msmar_query.fbin";
    std::string base_file = "/root/vsag/data/msmar_base.fbin";
    std::string gt_file = "/root/vsag/data/msmarc_groundtruth.ibin";

    auto vectors = (const float *)open_file(base_file.c_str());
    auto vectors_int = (const int32_t *)vectors;
    int64_t num_vectors = vectors_int[0];
    int64_t dim = vectors_int[1];
    vectors += 2;
    std::cout << "num_vectors: " << num_vectors << ", dim: " << dim << std::endl;
    bool is_train = true;
    int64_t block = 1000;
    int64_t start_block = 801;
    int64_t per_block_size = num_vectors / block;


    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    std::vector<int64_t> ids(num_vectors);
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    auto hnsw_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 1024,
        "hnsw": {
            "max_degree": 64,
            "ef_construction": 500
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();
    std::string before_file;
    if (start_block != 0) {
        before_file = fmt::format(file_path, start_block);
        if (std::filesystem::exists(before_file)) {
            std::ifstream file(before_file);
            index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();
            index->Deserialize(file);
            file.close();
        } else {
            std::cout << "index file " << before_file << " not exist" << std::endl;
            exit(-1);
        }
    }
    /******************* Build HNSW Index *****************/
    for (int i = start_block; i < block && is_train; ++i) {
        std::cout << "Building block " << i << std::endl;
        auto start = i * per_block_size;
        auto end = std::min((i + 1) * per_block_size, num_vectors);
#pragma omp parallel for
        for (int j = start; j < end; ++j) {
            auto vector = vsag::Dataset::Make();
            vector->NumElements(1)->Dim(dim)->Ids(ids.data() + j)->Float32Vectors(vectors + j * dim)->Owner(false);
            auto status = index->Add(vector);
        }
        if (i % 10 == 0 || i == block - 1) {
            auto after_file = fmt::format(file_path, i + 1);                                                                                                                               
            std::ofstream file(after_file);                                                                                                                                               
            index->Serialize(file);                                                                                                                                                       
            file.close();                                                                                                                                                                
            std::filesystem::remove(before_file);
            before_file = after_file;
        }
    }

    // hnsw_search_parameters is the configuration for searching in an HNSW index.
    // The "hnsw" section contains parameters specific to the search operation:
    // - "ef_search": The size of the dynamic list used for nearest neighbor search, which influences both recall and search speed.
    std::vector<int> efsearchs{ 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000 };
    constexpr static const char* hnsw_search_tempalte = R"(
    {{
        "hnsw": {{
            "ef_search": {}
        }}
    }}
    )";
    int64_t topk = 100;
    int64_t correct = 0;
    int64_t query_size = 1000;
    std::vector<vsag::DatasetPtr> results_ids;
    auto query_vectors = read_file(query_file.c_str(), query_size, dim, sizeof(float));
    auto query_data = (const float *)query_vectors.data();
    query_data += 2;
    auto query_int = (const int *)query_vectors.data();
    query_size = query_int[0];
    dim = query_int[1];
    std::cout << "query_size: " << query_size << ", dim: " << dim << std::endl;
    vsag::JsonType json;
    for (const auto& efsearch : efsearchs) {
        auto hnsw_search_parameters = fmt::format(hnsw_search_tempalte, efsearch);
        auto time_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < query_size; ++i) {
            auto query = vsag::Dataset::Make();
            query->NumElements(1)->Dim(dim)->Float32Vectors(query_data + dim * i)->Owner(false);
            auto knn_result = index->KnnSearch(query, topk, hnsw_search_parameters);
            results_ids.push_back(knn_result.value());
        }
        auto time_end = std::chrono::high_resolution_clock::now();
        auto time_cost =
            std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count() /
            (float)1000;

        int32_t gt_topk = 100;
        auto gt = read_file(gt_file.c_str(), query_size, gt_topk, sizeof(int32_t));
        auto gt_ids = (int32_t*)gt.data();
        if (gt_ids[0] != query_size) {
            std::cout << "Ground truth size mismatch: expected " << query_size << ", got "
                      << gt_ids[0] << std::endl;
        }
        if (gt_ids[1] != gt_topk) {
            std::cout << "Ground truth topk mismatch: expected " << gt_topk << ", got " << gt_ids[1]
                      << std::endl;
        }
        gt_ids += 2;
        for (int i = 0; i < query_size; ++i) {
            correct += getIntersectionSize(gt_ids + i * gt_topk, results_ids[i]->GetIds(), topk);
        }
        float recall = (float)correct / (query_size * topk);
        float qps = (float)query_size / time_cost;
        float rt = (float)time_cost / query_size;
        json[efsearch]["Recall"] = recall;
        json[efsearch]["QPS"] = qps;
        json[efsearch]["RT"] = rt;
    }
    std::cout << json.dump() << std::endl;

    return 0;
}