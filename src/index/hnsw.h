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

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../algorithm/hnswlib/hnswlib.h"
#include "../common.h"
#include "../default_allocator.h"
#include "../impl/conjugate_graph.h"
#include "../logger.h"
#include "../utils.h"
#include "vsag/binaryset.h"
#include "vsag/errors.h"
#include "vsag/index.h"
#include "vsag/readerset.h"

namespace vsag {

class HNSW : public Index {
public:
    HNSW(std::shared_ptr<hnswlib::SpaceInterface> space_interface,
         int M,
         int ef_construction,
         bool use_static = false,
         bool use_reversed_edges = false,
         bool use_conjugate_graph = false,
         int sq_num_bits = -1,
         float alpha = 1.0,
         std::string extra_file = "",
         Allocator* allocator = nullptr);

    ~HNSW();

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override {
        SAFE_CALL(return this->add(base));
    }

    tl::expected<bool, Error>
    Remove(int64_t id) override {
        SAFE_CALL(return this->remove(id));
    }

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, invalid));
    }

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid = nullptr,
                int64_t limited_size = -1) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, invalid, limited_size));
    }

    tl::expected<uint32_t, Error>
    Feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) override {
        SAFE_CALL(return this->feedback(query, k, parameters, global_optimum_tag_id));
    };

    tl::expected<uint32_t, Error>
    Pretrain(const std::vector<int64_t>& base_tag_ids,
             uint32_t k,
             const std::string& parameters) override {
        SAFE_CALL(return this->pretrain(base_tag_ids, k, parameters));
    };

    tl::expected<uint32_t, Error>
    Train(const DatasetPtr& base) override {
    };

    virtual tl::expected<float, Error>
    CalcDistanceById(const float* vector, int64_t id) const override {
        SAFE_CALL(return alg_hnsw->getDistanceByLabel(id, vector));
    };

    tl::expected<DatasetPtr, Error>
    BruteForce(const DatasetPtr& query, int64_t k) const override {
        SAFE_CALL(return this->brute_force(query, k));
    }

    virtual tl::expected<std::pair<double, uint32_t>, Error>
    Test(const DatasetPtr& query) override {
        int8_t* tq;
        int code_size;

        if (sq_num_bits_ == 4) {
            code_size = dim_ / 2;
            tq = new int8_t[code_size];
            alg_hnsw->transform_to_int4(query->GetFloat32Vectors(), tq);
        } else if (sq_num_bits_ == 8) {
            code_size = dim_;
            tq = new int8_t[code_size];
            alg_hnsw->transform_to_int8(query->GetFloat32Vectors(), tq);
        }

        double avg_time = 0;
        double single_time = 0;
        for (int i = 0; i < 1000; i++) {
            single_time = 0;
            auto tb = alg_hnsw->get_encoded_data(rand() % alg_hnsw->getCurrentElementCount(),
                                                 code_size + 8);
            {
                Timer t(single_time);
                if (sq_num_bits_ == 4) {
                    alg_hnsw->INT4_IP((const void*)tq, (const void*)tb, dim_);
                } else if (sq_num_bits_ == 8) {
                    alg_hnsw->INT8_IP_TEST((const void*)tq, (const void*)tb, dim_);
                }
            }
            avg_time += single_time;
        }

        return std::make_pair(avg_time, 0);
    }

public:
    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->serialize(out_stream));
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream) override {
        SAFE_CALL(return this->deserialize(in_stream));
    }

public:
    int64_t
    GetNumElements() const override {
        return alg_hnsw->getCurrentElementCount() - alg_hnsw->getDeletedCount();
    }

    int64_t
    GetMemoryUsage() const override {
        if (use_conjugate_graph_)
            return alg_hnsw->calcSerializeSize() + conjugate_graph_->GetMemoryUsage();
        else
            return alg_hnsw->calcSerializeSize();
    }

    std::string
    GetStats() const override;

    // used to test the integrity of graphs, used only in UT.
    bool
    CheckGraphIntegrity() const;

private:
    tl::expected<std::vector<int64_t>, Error>
    build(const DatasetPtr& base);

    tl::expected<std::vector<int64_t>, Error>
    add(const DatasetPtr& base);

    tl::expected<bool, Error>
    remove(int64_t id);

    tl::expected<DatasetPtr, Error>
    knn_search(const DatasetPtr& query,
               int64_t k,
               const std::string& parameters,
               BitsetPtr invalid = nullptr) const;

    tl::expected<DatasetPtr, Error>
    range_search(const DatasetPtr& query,
                 float radius,
                 const std::string& parameters,
                 BitsetPtr invalid,
                 int64_t limited_size) const;

    tl::expected<uint32_t, Error>
    feedback(const DatasetPtr& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id);

    tl::expected<uint32_t, Error>
    feedback(const DatasetPtr& result, int64_t global_optimum_tag_id, int64_t k);

    tl::expected<DatasetPtr, Error>
    brute_force(const DatasetPtr& query, int64_t k) const;

    tl::expected<uint32_t, Error>
    pretrain(const std::vector<int64_t>& base_tag_ids, uint32_t k, const std::string& parameters);

    tl::expected<BinarySet, Error>
    serialize() const;

    tl::expected<void, Error>
    serialize(std::ostream& out_stream);

    tl::expected<void, Error>
    deserialize(const BinarySet& binary_set);

    tl::expected<void, Error>
    deserialize(const ReaderSet& binary_set);

    tl::expected<void, Error>
    deserialize(std::istream& in_stream);

    BinarySet
    empty_binaryset() const;

private:
    std::shared_ptr<hnswlib::AlgorithmInterface<float>> alg_hnsw;
    std::shared_ptr<hnswlib::SpaceInterface> space;

    bool use_conjugate_graph_;
    std::shared_ptr<ConjugateGraph> conjugate_graph_;

    int64_t dim_;
    bool use_static_ = false;
    bool empty_index_ = false;
    bool use_reversed_edges_ = false;
    int sq_num_bits_ = -1;

    std::shared_ptr<Allocator> allocator_ = nullptr;
    std::string pq_code_file;

    mutable std::mutex stats_mutex_;
    mutable std::map<std::string, WindowResultQueue> result_queues_;
};

}  // namespace vsag
