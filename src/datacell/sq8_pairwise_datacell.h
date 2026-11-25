// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>

#include "flatten_datacell.h"
#include "flatten_pairwise_distance_interface.h"
#include "quantization/scalar_quantization/scalar_quantizer.h"

namespace vsag {

template <typename Layout>
class SQ8PairwiseDataCell final
    : public FlattenDataCell<SQ8Quantizer<MetricType::METRIC_TYPE_L2SQR>, Layout>,
      public FlattenPairwiseDistanceInterface {
    using QuantizerType = SQ8Quantizer<MetricType::METRIC_TYPE_L2SQR>;
    using Base = FlattenDataCell<QuantizerType, Layout>;

public:
    using Base::Base;

    std::unique_ptr<FlattenPairwiseDistanceComputer>
    FactoryPairwiseDistanceComputer() override {
        return nullptr;
    }

    bool
    ComputePairwiseTile(std::span<const InnerIdType> sources,
                        std::span<const InnerIdType> targets,
                        float* output,
                        std::optional<uint64_t> first_diagonal) const override {
        if (this->external_read_mutex_)
            return false;
        if (sources.empty() || targets.empty())
            return true;
        CHECK_ARGUMENT(!first_diagonal || (*first_diagonal <= targets.size() &&
                                           sources.size() - 1 <= targets.size() - *first_diagonal),
                       "invalid SQ8 tile diagonal");
        const auto total = this->TotalCount();
        auto acquire = [&](InnerIdType id) {
            CHECK_ARGUMENT(id < total, "invalid SQ8 pairwise ID");
            auto lease = this->layout_->Acquire(id);
            CHECK_ARGUMENT(lease, "missing SQ8 pairwise code");
            return lease;
        };
        Vector<typename Layout::Lease> queries(this->allocator_);
        queries.reserve(sources.size());
        for (auto id : sources) queries.push_back(acquire(id));
        const auto& quantizer = static_cast<const QuantizerType&>(*this->quantizer_);
        // Reuse a block of target codes across the source rows. Leases also own records
        // that straddle two MemoryBlockIO blocks.
        constexpr size_t block_size = 64;
        std::array<typename Layout::Lease, block_size> leases;
        std::array<const uint8_t*, block_size> codes;
        const size_t columns =
            first_diagonal ? *first_diagonal + sources.size() - 1 : targets.size();
        for (size_t begin = 0; begin < columns; begin += block_size) {
            const size_t end = std::min(columns, begin + block_size);
            for (size_t col = begin; col < end; ++col) {
                leases[col - begin] = acquire(targets[col]);
                codes[col - begin] = leases[col - begin].Data();
            }
            const size_t first_row =
                first_diagonal && begin >= *first_diagonal ? begin - *first_diagonal + 1 : 0;
            for (size_t row = first_row; row < sources.size(); ++row) {
                const size_t n =
                    first_diagonal ? std::min(end, *first_diagonal + row) - begin : end - begin;
                const auto* source = queries[row].Data();
                auto* dest = output + row * targets.size() + begin;
                size_t j = 0;
                for (; j + 4 <= n; j += 4)
                    quantizer.ComputeCodeDistances4(source, codes.data() + j, dest + j);
                for (; j < n; ++j) dest[j] = this->quantizer_->Compute(source, codes[j]);
            }
        }
        return true;
    }
};
}  // namespace vsag
