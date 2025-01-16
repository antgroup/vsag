
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

#pragma once

#include "base_filter_functor.h"
#include "brute_force_parameter.h"
#include "common.h"
#include "data_cell/flatten_datacell.h"
#include "index_feature_list.h"
#include "label_table.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {
class BruteForce : public Index {
public:
    explicit BruteForce(const BruteForceParameter& param, const IndexCommonParam& common_param);

    ~BruteForce() override;

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& data) override {
        SAFE_CALL(return this->build(data));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& data) override {
        SAFE_CALL(return this->add(data));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        std::function<bool(int64_t)> func = [&invalid](int64_t id) -> bool {
            int64_t bit_index = id & ROW_ID_MASK;
            return invalid->Test(bit_index);
        };
        if (invalid == nullptr) {
            func = nullptr;
        }
        SAFE_CALL(return this->knn_search(query, k, parameters, func));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, filter));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, nullptr, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        BitsetOrCallbackFilter filter(invalid);
        SAFE_CALL(return this->range_search(query, radius, parameters, &filter, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size) const override {
        BitsetOrCallbackFilter callback(filter);
        SAFE_CALL(return this->range_search(query, radius, parameters, &callback, limited_size));
    }

    tl::expected<float, Error>
    CalcDistanceById(const float* vector, int64_t id) const override {
        SAFE_CALL(return this->calculate_distance_by_id(vector, id));
    };

    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(this->serialize(out_stream));
        return {};
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(this->deserialize(in_stream));
        return {};
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(this->deserialize(binary_set));
        return {};
    };

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(this->deserialize(reader_set));
        return {};
    }

    [[nodiscard]] int64_t
    GetNumElements() const override {
        return this->total_count_;
    }

    [[nodiscard]] int64_t
    GetMemoryUsage() const override;

    [[nodiscard]] uint64_t
    EstimateMemory(uint64_t num_elements) const override;

    [[nodiscard]] bool
    CheckFeature(IndexFeature feature) const override;

private:
    std::vector<int64_t>
    build(const DatasetPtr& data);

    std::vector<int64_t>
    add(const DatasetPtr& data);

    DatasetPtr
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               const std::function<bool(int64_t)>& filter) const;

    DatasetPtr
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 BaseFilterFunctor* filter_ptr,
                 int64_t limited_size) const;

    float
    calculate_distance_by_id(const float* vector, int64_t id) const;

    [[nodiscard]] BinarySet
    serialize() const;

    void
    serialize(std::ostream& out_stream) const;

    void
    serialize(StreamWriter& writer) const;

    void
    deserialize(std::istream& in_stream);

    void
    deserialize(const BinarySet& binary_set);

    void
    deserialize(const ReaderSet& reader_set);

    void
    deserialize(StreamReader& reader);

    uint64_t
    cal_serialize_size() const;

    Vector<DatasetPtr>
    split_dataset_by_duplicate_label(const DatasetPtr& dataset,
                                     std::vector<LabelType>& failed_ids) const;

    void
    init_feature_list();

private:
    FlattenInterfacePtr inner_codes_{nullptr};

    LabelTablePtr label_table_;
    std::shared_ptr<Allocator> allocator_{nullptr};

    int64_t dim_{0};

    uint64_t total_count_{0};

    IndexFeatureList feature_list_{};
};
}  // namespace vsag
