
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

#include "algorithm/hgraph.h"
#include "hgraph_index_zparameters.h"
#include "index_common_param.h"
#include "typing.h"
#include "vsag/index.h"

namespace vsag {
class HGraphIndex : public Index {
public:
    HGraphIndex(const HGraphIndexParameter& param, const IndexCommonParam& common_param);

    ~HGraphIndex() override;

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& data) override {
        SAFE_CALL(return this->hgraph_->Build(data));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& data) override {
        SAFE_CALL(return this->hgraph_->Add(data));
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
        SAFE_CALL(return this->hgraph_->KnnSearch(query, k, parameters, func));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const std::function<bool(int64_t)>& filter) const override {
        SAFE_CALL(return this->hgraph_->KnnSearch(query, k, parameters, filter));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                int64_t limited_size = -1) const override {
        SAFE_CALL(
            return this->hgraph_->RangeSearch(query, radius, parameters, nullptr, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid,
                int64_t limited_size = -1) const override {
        BitsetOrCallbackFilter filter(invalid);
        SAFE_CALL(
            return this->hgraph_->RangeSearch(query, radius, parameters, &filter, limited_size));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const std::function<bool(int64_t)>& filter,
                int64_t limited_size) const override {
        BitsetOrCallbackFilter callback(filter);
        SAFE_CALL(
            return this->hgraph_->RangeSearch(query, radius, parameters, &callback, limited_size));
    }

    tl::expected<float, Error>
    CalcDistanceById(const float* vector, int64_t id) const override {
        SAFE_CALL(return this->hgraph_->CalculateDistanceById(vector, id));
    };

    tl::expected<DatasetPtr, Error>
    CalDistanceById(const float* vector, const int64_t* ids, int64_t count) const override {
        SAFE_CALL(return this->hgraph_->CalDistanceById(vector, ids, count));
    };

    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->hgraph_->Serialize());
    }

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->hgraph_->Serialize(out_stream));
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(return this->hgraph_->Deserialize(in_stream));
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->hgraph_->Deserialize(binary_set));
    };

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->hgraph_->Deserialize(reader_set));
    }

    [[nodiscard]] int64_t
    GetNumElements() const override {
        return this->hgraph_->GetNumElements();
    }

    [[nodiscard]] int64_t
    GetMemoryUsage() const override {
        return this->hgraph_->GetMemoryUsage();
    }

    [[nodiscard]] uint64_t
    EstimateMemory(uint64_t num_elements) const override {
        return this->hgraph_->EstimateMemory(num_elements);
    }

    [[nodiscard]] bool
    CheckFeature(IndexFeature feature) const override {
        return this->hgraph_->CheckFeature(feature);
    }

    [[nodiscard]] bool
    CheckIdExist(int64_t id) const override {
        return this->hgraph_->CheckIdExist(id);
    }

private:
    std::unique_ptr<HGraph> hgraph_{nullptr};

    std::shared_ptr<Allocator> allocator_{nullptr};
};
}  // namespace vsag
