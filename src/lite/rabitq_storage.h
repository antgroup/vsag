// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "lite/rabitq_constants.h"
#include "lite/rabitq_transform.h"
#include "simd/kernels/rabitq_pack.h"

// Encoded record data structures and the SoA (structure-of-arrays) storage
// container. Pure data with Append / Replace / RemoveSwap semantics.

namespace vsag::lite::detail::rabitq {

struct EncodedMetadata {
    float norm{}, code_norm{}, error{}, filter_norm{}, filter_error{}, lower_bound_error{};
};

struct EncodedView {
    const uint8_t* filter;
    const uint8_t* supplement;
    EncodedMetadata metadata;
};

struct Encoded {
    std::vector<uint8_t> filter, supplement, scalar;
    float norm{}, code_norm{}, error{}, filter_norm{}, filter_error{}, lower_bound_error{};
};

struct EncodedRecords {
    explicit EncodedRecords(uint64_t input_dim) : dim(input_dim) {
    }

    [[nodiscard]] uint64_t
    FilterBytes() const {
        return ((dim + 7) / 8) * K_FILTER_BITS;
    }

    [[nodiscard]] uint64_t
    SupplementBytes() const {
        return ((dim + 7) / 8) * K_SUPPLEMENT_BITS;
    }

    [[nodiscard]] uint64_t
    Size() const {
        return metadata.size();
    }

    void
    Reserve(uint64_t count) {
        metadata.reserve(count);
        filters.reserve(count * FilterBytes());
        supplements.reserve(count * SupplementBytes());
    }

    void
    Resize(uint64_t count) {
        metadata.resize(count);
        filters.resize(count * FilterBytes());
        supplements.resize(count * SupplementBytes());
    }

    void
    Append(Encoded code) {
        codec_require(
            code.filter.size() == FilterBytes() and code.supplement.size() == SupplementBytes(),
            "invalid encoded record");
        metadata.push_back({code.norm,
                            code.code_norm,
                            code.error,
                            code.filter_norm,
                            code.filter_error,
                            code.lower_bound_error});
        filters.insert(filters.end(), code.filter.begin(), code.filter.end());
        supplements.insert(supplements.end(), code.supplement.begin(), code.supplement.end());
    }

    void
    Replace(uint64_t slot, Encoded code) {
        codec_require(slot < Size() and code.filter.size() == FilterBytes() and
                          code.supplement.size() == SupplementBytes(),
                      "invalid encoded replacement");
        metadata[slot] = {code.norm,
                          code.code_norm,
                          code.error,
                          code.filter_norm,
                          code.filter_error,
                          code.lower_bound_error};
        std::copy(code.filter.begin(), code.filter.end(), filters.data() + slot * FilterBytes());
        std::copy(code.supplement.begin(),
                  code.supplement.end(),
                  supplements.data() + slot * SupplementBytes());
    }

    void
    RemoveSwap(uint64_t slot) {
        codec_require(slot < Size(), "encoded removal outside storage");
        const uint64_t last = Size() - 1;
        if (slot != last) {
            metadata[slot] = metadata[last];
            std::copy_n(filters.data() + last * FilterBytes(),
                        FilterBytes(),
                        filters.data() + slot * FilterBytes());
            std::copy_n(supplements.data() + last * SupplementBytes(),
                        SupplementBytes(),
                        supplements.data() + slot * SupplementBytes());
        }
        metadata.pop_back();
        filters.resize(last * FilterBytes());
        supplements.resize(last * SupplementBytes());
    }

    [[nodiscard]] EncodedView
    At(uint64_t id) const {
        codec_require(id < Size(), "encoded record outside storage");
        return {filters.data() + id * FilterBytes(),
                supplements.data() + id * SupplementBytes(),
                metadata[id]};
    }

    uint64_t dim;
    std::vector<EncodedMetadata> metadata;
    std::vector<uint8_t> filters;
    std::vector<uint8_t> supplements;
};

}  // namespace vsag::lite::detail::rabitq
