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

#include <optional>

#include "algorithm/inner_index_interface.h"
#include "bruteforce_parameter.h"
#include "impl/label_table/label_table.h"
#include "typing.h"
#include "utils/pointer_define.h"
#include "vsag/filter.h"

namespace vsag {

class SafeThreadPool;

DEFINE_POINTER2(AttrInvertedInterface, AttributeInvertedInterface);
DEFINE_POINTER(FlattenInterface);

// BruteForce index supports both single-vector and multi-vector (WARP-style) modes
// via FlattenInterface polymorphism (introduced since v0.13)
class BruteForce : public InnerIndexInterface {
public:
    static ParamPtr
    CheckAndMappingExternalParam(const JsonType& external_param,
                                 const IndexCommonParam& common_param);

public:
    explicit BruteForce(const BruteForceParameterPtr& param, const IndexCommonParam& common_param);

    explicit BruteForce(const ParamPtr& param, const IndexCommonParam& common_param)
        : BruteForce(std::dynamic_pointer_cast<BruteForceParameter>(param), common_param){};

    ~BruteForce() override = default;

    std::vector<int64_t>
    Add(const DatasetPtr& data) override;

    std::vector<int64_t>
    Build(const DatasetPtr& data) override;

    float
    CalcDistanceById(const float* vector,
                     int64_t id,
                     bool calculate_precise_distance = true) const override;

    void
    Deserialize(StreamReader& reader) override;

    uint64_t
    EstimateMemory(uint64_t num_elements) const override;

    [[nodiscard]] InnerIndexPtr
    Fork(const IndexCommonParam& param) override {
        return std::make_shared<BruteForce>(this->create_param_ptr_, param);
    }

    void
    GetAttributeSetByInnerId(InnerIdType inner_id, AttributeSet* attr) const override;

    [[nodiscard]] IndexType
    GetIndexType() const override {
        return is_multi_vector_ ? IndexType::WARP : IndexType::BRUTEFORCE;
    }

    std::string
    GetName() const override {
        return is_multi_vector_ ? INDEX_WARP : INDEX_BRUTE_FORCE;
    }

    [[nodiscard]] int64_t
    GetNumElements() const override {
        auto deleted = static_cast<int64_t>(this->delete_count_.load());
        auto total = static_cast<int64_t>(this->total_count_.load());
        return total > deleted ? total - deleted : 0;
    }

    [[nodiscard]] int64_t
    GetNumberRemoved() const override {
        return this->delete_count_.load();
    }

    void
    GetVectorByInnerId(InnerIdType inner_id, float* data) const override;

    void
    InitFeatures() override;

    [[nodiscard]] DatasetPtr
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              const FilterPtr& filter) const override;

    [[nodiscard]] DatasetPtr
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                const FilterPtr& filter,
                int64_t limited_size = -1) const override;

    uint32_t
    Remove(const std::vector<int64_t>& ids, RemoveMode mode = RemoveMode::MARK_REMOVE) override;

    [[nodiscard]] DatasetPtr
    SearchWithRequest(const SearchRequest& request) const override;

    void
    Serialize(StreamWriter& writer) const override;

    void
    Train(const DatasetPtr& data) override;

    void
    UpdateAttribute(int64_t id, const AttributeSet& new_attrs) override;

    void
    UpdateAttribute(int64_t id,
                    const AttributeSet& new_attrs,
                    const AttributeSet& origin_attrs) override;

    int64_t
    GetMemoryUsage() const override;

private:
    /**
     * @brief Grow the capacity of inner_codes_ to at least new_size slots.
     *
     * Capacity grows arithmetically (by a fixed increment of 2^resize_increase_count_bit_) to
     * amortise allocation cost.  Safe to call with a smaller value than the
     * current capacity (no-op).
     */
    void
    resize(uint64_t new_size);

    /**
     * @brief Reserve the next available internal slot for @p label.
     *
     * In single-vector mode the slot index equals total_count_.
     * In multi-vector mode the slot is reserved only if @p label is new;
     * attribute-based dedup is performed when @p attr is non-null.
     *
     * @return the allocated inner id, or std::nullopt if the label was
     *         already present (multi-vector mode only).
     */
    std::optional<InnerIdType>
    claim_slot(int64_t label, const AttributeSet* attr);

    /**
     * @brief Write one vector into inner_codes_ at position @p inner_id.
     *
     * The caller must have already called claim_slot() and resize().
     */
    void
    add_one(const float* data, InnerIdType inner_id);

    /**
     * @brief Recalculate and cache the memory-usage counter.
     *
     * Called after Add/Build/Remove to keep GetMemoryUsage() cheap.
     */
    void
    cal_memory_usage();

    /// Build the multi-vector label table and train the quantizer.
    void
    train_multi_vector(const DatasetPtr& data);

    /// Insert multi-vector (WARP) data; returns failed external ids.
    std::vector<int64_t>
    add_multi_vector(const DatasetPtr& data);

    /**
     * @brief Create a Computer object sized for @p query, used by
     *        KnnSearch / RangeSearch in multi-vector mode.
     */
    ComputerInterfacePtr
    make_search_computer(const DatasetPtr& query) const;

private:
    FlattenInterfacePtr inner_codes_{nullptr};  // quantized or raw vector storage

    std::atomic<uint64_t> delete_count_{0};  // number of soft-deleted vectors

    uint64_t resize_increase_count_bit_{DEFAULT_RESIZE_BIT};  // log2 of capacity increment

    mutable std::shared_mutex global_mutex_;  // protects reads during resize
    mutable std::shared_mutex add_mutex_;     // serialises Add / Remove

    std::atomic<InnerIdType> max_capacity_{0};  // allocated slot count

    bool is_multi_vector_{false};  // true ⇒ WARP / multi-vector mode

    static constexpr uint64_t DEFAULT_RESIZE_BIT = 10;  // default increment = 1024
};
}  // namespace vsag
