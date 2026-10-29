#pragma once

#include <cstdint>

namespace vsag {
using SQ4LocalSharedComputeType =
    float (*)(const float*, const uint8_t*, const uint8_t*, uint64_t, bool);
using SQ4LocalSharedComputeBatch4Type =
    void (*)(const float*, const uint8_t* const*, uint64_t, bool, float*);

SQ4LocalSharedComputeType
GetSQ4LocalSharedCompute();
SQ4LocalSharedComputeBatch4Type
GetSQ4LocalSharedComputeBatch4();

namespace avx2 {
float
SQ4LocalSharedCompute(
    const float* query, const uint8_t* first, const uint8_t* second, uint64_t dim, bool l2);
}
namespace avx512 {
float
SQ4LocalSharedCompute(
    const float* query, const uint8_t* first, const uint8_t* second, uint64_t dim, bool l2);
void
SQ4LocalSharedComputeBatch4(
    const float* query, const uint8_t* const* codes, uint64_t dim, bool l2, float* distances);
}  // namespace avx512
}  // namespace vsag
