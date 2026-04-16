//
// Created by root on 12/16/24.
//

#include "test_dataset_pool.h"

namespace fixtures {
TestDatasetPtr
TestDatasetPool::GetDatasetAndCreate(uint64_t dim,
                                     uint64_t count,
                                     const std::string& metric_str,
                                     bool with_path,
                                     float valid_ratio,
                                     uint64_t extra_info_size,
                                     int64_t id_shift) {
    auto key = key_gen(dim, count, metric_str, with_path, valid_ratio, extra_info_size, id_shift);
    if (this->pool_.find(key) == this->pool_.end()) {
        this->dim_counts_.emplace_back(dim, count);
        this->pool_[key] = TestDataset::CreateTestDataset(dim,
                                                          count,
                                                          metric_str,
                                                          with_path,
                                                          valid_ratio,
                                                          "dense",
                                                          extra_info_size,
                                                          false,
                                                          id_shift);
    }
    return this->pool_.at(key);
}
std::string
TestDatasetPool::key_gen(int64_t dim,
                         uint64_t count,
                         const std::string& metric_str,
                         bool with_path,
                         float filter_ratio,
                         uint64_t extra_info_size,
                         int64_t id_shift) {
    return std::to_string(dim) + "_" + std::to_string(count) + "_" + metric_str + "_" +
           std::to_string(with_path) + "_" + std::to_string(filter_ratio) +
           std::to_string(extra_info_size) + "_" + std::to_string(id_shift);
}

TestDatasetPtr
TestDatasetPool::GetNanDataset(const std::string& metric_str) {
    auto key = NAN_DATASET + metric_str;
    if (this->pool_.find(key) == this->pool_.end()) {
        this->pool_[key] = TestDataset::CreateNanDataset(metric_str);
    }
    return this->pool_.at(key);
}

TestDatasetPtr
TestDatasetPool::GetSparseDatasetAndCreate(uint64_t count, uint64_t dim, float valid_ratio) {
    auto key = "sparse_" + std::to_string(count) + "_" + std::to_string(dim) + "_" +
               std::to_string(valid_ratio);
    if (this->pool_.find(key) == this->pool_.end()) {
        this->pool_[key] =
            TestDataset::CreateTestDataset(dim, count, "l2", false, valid_ratio, "sparse");
    }
    return this->pool_.at(key);
}

TestDatasetPtr
TestDatasetPool::GetDuplicateDataset(uint64_t dim,
                                     uint64_t count,
                                     const std::string& metric_str,
                                     bool with_path,
                                     float valid_ratio,
                                     uint64_t extra_info_size) {
    auto key =
        key_gen(dim, count, metric_str, with_path, valid_ratio, extra_info_size) + "_duplicate";
    if (this->pool_.find(key) == this->pool_.end()) {
        this->dim_counts_.emplace_back(dim, count);
        this->pool_[key] = TestDataset::CreateTestDataset(
            dim, count, metric_str, with_path, valid_ratio, "dense", extra_info_size, true);
    }
    return this->pool_.at(key);
}

}  // namespace fixtures
