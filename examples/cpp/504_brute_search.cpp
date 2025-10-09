#include <vsag/vsag.h>
#include <iostream>
#include "SimpleBruteSearch.h" // 包含自定义的索引

int main(int argc, char** argv) {

    vsag::init();

    int64_t num_vectors = 1000;
    int64_t dim = 128;
    std::vector<int64_t> ids(num_vectors);
    std::vector<float> vectors(num_vectors * dim);

    // 准备随机数据。
    std::mt19937 rng(47);
    std::uniform_real_distribution<float> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    // 创建Dataset对象
    auto base = vsag::Dataset::Make();
    base->NumElements(num_vectors)
        ->Dim(dim)
        ->Ids(ids.data())
        ->Float32Vectors(vectors.data())
        ->Owner(false); 


    // 使用简单BruteForce索引
    auto index = vsag::CreateSimpleBruteSearchIndex(dim, "l2");


    if (auto build_result = index->Build(base); build_result.has_value()) {
        std::cout << "构建完成后，自定义BruteForce索引包含: " << static_cast<vsag::SimpleBruteSearchIndex*>(index.get())->GetNumElements() << " 个向量" << std::endl;
    }

    std::vector<float> query_vector(dim);
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    auto query = vsag::Dataset::Make();
    query->NumElements(1)->Dim(dim)->Float32Vectors(query_vector.data())->Owner(false);

    //执行KNN搜索
    int64_t topk = 10;
    std::string parameters = "";
    auto knn_result = index->KnnSearch(query, topk, parameters);

    //输出搜索结果
    if (knn_result.has_value()) {
        auto result = knn_result.value();
        std::cout << "KNN搜索结果: " << std::endl;
        for (int64_t i = 0; i < result->GetNumElements(); ++i) {
            std::cout << "ID: " << result->GetIds()[i] << ", 距离: " << result->GetDistances()[i] << std::endl;
        }
    }

    return 0;
}