// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

namespace vsag {

// Optional multi-vector token capability. Both encoded operands must belong
// to the same owning data cell's quantizer/model. The owning shared_ptr keeps
// this interface alive throughout a build or incremental update.
class TokenCodeInterface {
public:
    virtual ~TokenCodeInterface() = default;
    virtual void
    EncodeToken(const float* vector, uint8_t* code) = 0;
    virtual float
    ComputeTokenCodes(const uint8_t* lhs, const uint8_t* rhs) = 0;
};

}  // namespace vsag
