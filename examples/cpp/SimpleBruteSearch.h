#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <algorithm>

#include <vsag/vsag.h>
#include <vsag/index.h>
#include <vsag/dataset.h>
#include <vsag/bitset.h>
#include <vsag/errors.h>
#include <vsag/expected.hpp>

namespace vsag {

// 简单的Brute Force索引实现
class SimpleBruteSearchIndex : public Index {
public:
    int64_t dim_;
    int64_t num_elements_;
    std::string metric_type_;
    std::vector<int64_t> ids_;
    std::vector<float> vectors_;

    SimpleBruteSearchIndex(int64_t dim, const std::string& metric_type)
        : dim_(dim), metric_type_(metric_type) {}

    // 构建索引（实际上只是存储数据）
    tl::expected<std::vector<int64_t>, Error> Build(const DatasetPtr& base) override {

        dim_=base->GetDim();
        num_elements_=base->GetNumElements();

        ids_.resize(num_elements_);

        vectors_.resize(num_elements_ * dim_);
        const auto* base_ids = base->GetIds();
        const auto* base_vectors = base->GetFloat32Vectors();
        std::copy(base_ids, base_ids + num_elements_, ids_.begin());
        std::copy(base_vectors, base_vectors + num_elements_ * dim_, vectors_.begin());
        return std::vector<int64_t>();
    }
    
    // 获取索引类型
    IndexType GetIndexType() override {
        return IndexType::BRUTEFORCE;
    }
    

    int64_t GetNumElements() const override {
        return num_elements_;
    }
    
    // KNN搜索实现——非空版（504_brute_search.cpp使用的版本）
   tl::expected<DatasetPtr, Error> KnnSearch(
        const DatasetPtr& query, 
        int64_t k, 
        const std::string& parameters, 
        BitsetPtr invalid = nullptr) const override {
        
        // 创建结果数据集
        auto result = Dataset::Make();

        const float* query_vec = query->GetFloat32Vectors();
        
        // 使用优先队列存储距离和id，只保留k个最近邻。
        std::priority_queue<std::pair<float, int64_t>> priority_queue; // 默认最大距离的元素在顶部
        
        // 遍历所有向量，计算与查询向量的距离
        for (int64_t i = 0; i < num_elements_; ++i) {
            // 计算查询向量与当前向量之间的L2距离。
            float distance = l2sqr(query_vec, vectors_.data() + i * dim_, dim_);

    
            if (priority_queue.size() < k) {
                // 如果队列中的元素少于k个，则直接添加当前元素。
                std::pair<float, int64_t> distance_id_pair;
                distance_id_pair.first = distance;
                distance_id_pair.second = ids_[i];
                priority_queue.push(distance_id_pair);
            } 
            else {
                // 队列已有k个元素，只有当前距离比队列中最大距离小时才会替换
                if (distance < priority_queue.top().first) {
                    // 移除队列中距离最大的元素
                    priority_queue.pop();
                    // 添加当前元素
                    std::pair<float, int64_t> distance_id_pair;
                    distance_id_pair.first = distance;
                    distance_id_pair.second = ids_[i];
                    priority_queue.push(distance_id_pair);
                }
            }
        }

        // 准备结果数组
        int64_t result_count = priority_queue.size();
        std::vector<int64_t> result_ids(result_count);
        std::vector<float> result_dists(result_count);
        
        // 从优先队列中提取结果
        // 最大距离的元素在顶部，我们需要从后往前填充数组。
        for (int64_t i = result_count - 1; i >= 0; --i) {
            // 获取队列顶部元素（当前队列中距离最大的）
            std::pair<float, int64_t> top_element = priority_queue.top();
            
            // 存储到结果数组
            result_dists[i] = top_element.first;
            result_ids[i] = top_element.second;
            
            // 移除已处理的元素
            priority_queue.pop();
        }

        // 设置结果数据集
        result->NumElements(result_count);
        // 设置结果ID数组
        result->Ids(result_ids.data());
        result->Distances(result_dists.data());
        result->Owner(false);

        return result;
    }
    
    // KNN重载的空函数
    tl::expected<DatasetPtr, Error> KnnSearch(
        const DatasetPtr& query, 
        int64_t k, 
        const std::string& parameters, 
        const std::function<bool(int64_t)>& filter) const override {
        auto result = Dataset::Make();
        return result;
    }
    //空函数，避免因纯虚函数未实现报错
    [[nodiscard]] tl::expected<DatasetPtr, Error> RangeSearch(
        const DatasetPtr& query, 
        float radius, 
        const std::string& parameters, 
        int64_t limited_size = -1) const override {
        auto result = Dataset::Make();
        return result;
    }
    
   //空函数
    [[nodiscard]] tl::expected<DatasetPtr, Error> RangeSearch(
        const DatasetPtr& query, 
        float radius, 
        const std::string& parameters, 
        BitsetPtr invalid, 
        int64_t limited_size = -1) const override {
        auto result = Dataset::Make();
        return result;
    }
    
   //空函数
    tl::expected<DatasetPtr, Error> RangeSearch(
        const DatasetPtr& query, 
        float radius, 
        const std::string& parameters, 
        const std::function<bool(int64_t)>& filter, 
        int64_t limited_size = -1) const override {
        auto result = Dataset::Make();
        return result;
    }

   //空函数，返回无能为力
    [[nodiscard]] tl::expected<BinarySet, Error> Serialize() const override {
        return tl::make_unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, "Serialize not implemented"));
    }
    
   //空函数，返回无能为力
    tl::expected<void, Error> Deserialize(const BinarySet& binary_set) override {
        return tl::make_unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, "Deserialize not implemented"));
    }
    
   //空函数，返回无能为力
    tl::expected<void, Error> Deserialize(const ReaderSet& reader_set) override {
        return tl::make_unexpected(Error(ErrorType::UNSUPPORTED_INDEX_OPERATION, "Deserialize not implemented"));
    }
    
   //空函数，返回0。
    [[nodiscard]] int64_t GetMemoryUsage() const override {
        return 0;
    }

    // 计算两个向量之间的L2距离平方
    float l2sqr(const void* vec1, const void* vec2, int64_t dim) const {
        auto* v1 = static_cast<const float*>(vec1);
        auto* v2 = static_cast<const float*>(vec2);

        float res = 0;
        for (int64_t i = 0; i < dim; i++) {
            float t = *v1 - *v2;
            v1++;
            v2++;
            res += t * t;
        }

        return res;
    }

};

// 工厂函数，用于创建SimpleBruteSearchIndex实例
inline std::shared_ptr<Index> CreateSimpleBruteSearchIndex(int64_t dim, const std::string& metric_type = "l2") {
    return std::make_shared<SimpleBruteSearchIndex>(dim, metric_type);
}

} // namespace vsag