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

/**
 * @file rabitq_simd.h
 * @brief SIMD-accelerated RaBitQ (Rapid Bit Quantization) operations.
 *
 * This file provides functions for RaBitQ quantization and distance computation,
 * supporting multiple SIMD instruction sets including SSE, AVX, AVX2, AVX512,
 * NEON, SVE, and generic implementations. RaBitQ is used for fast approximate
 * nearest neighbor search with binary quantization.
 */

#pragma once

#include <cstdint>

namespace vsag {

/**
 * @brief AVX512 with VPOPCNTDQ extension namespace for RaBitQ operations.
 */
namespace avx512vpopcntdq {

/**
 * @brief Compute binary inner product between SQ4U codes and bits.
 *
 * @param codes The SQ4U quantized codes.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @return The computed binary inner product.
 */
uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

}  // namespace avx512vpopcntdq

/**
 * @brief AVX512 namespace for RaBitQ operations.
 */
namespace avx512 {

/**
 * @brief Compute inner product between float vector and binary bits.
 *
 * @param vector The float vector.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @param inv_sqrt_d Precomputed 1/sqrt(dim) factor.
 * @return The computed inner product.
 */
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

/**
 * @brief Compute binary inner product between SQ4U codes and bits.
 *
 * @param codes The SQ4U quantized codes.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @return The computed binary inner product.
 */
uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

/**
 * @brief Perform Kaczmarz walk on data vector.
 *
 * @param data The data vector to transform.
 * @param len The length of the data vector.
 */
void
KacsWalk(float* data, uint64_t len);

/**
 * @brief Rescale vector elements by a scalar value.
 *
 * @param data The data vector to rescale.
 * @param dim The dimension of the vector.
 * @param val The scaling factor.
 */
void
VecRescale(float* data, uint64_t dim, float val);

/**
 * @brief Apply Fast Hadamard Transform rotation to data vector.
 *
 * @param data The data vector to transform.
 * @param dim_ The dimension of the vector.
 */
void
FHTRotate(float* data, uint64_t dim_);

/**
 * @brief Flip sign of vector elements based on flip bits.
 *
 * @param flip The flip bits pattern.
 * @param data The data vector to modify.
 * @param dim The dimension of the vector.
 */
void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

/**
 * @brief Perform rotation operation on data vector.
 *
 * @param data The data vector to rotate.
 * @param idx The rotation index.
 * @param dim_ The dimension of the vector.
 * @param step The rotation step size.
 */
void
RotateOp(float* data, int idx, int dim_, int step);

}  // namespace avx512

/**
 * @brief AVX2 namespace for RaBitQ operations.
 */
namespace avx2 {

/**
 * @brief Compute inner product between float vector and binary bits.
 *
 * @param vector The float vector.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @param inv_sqrt_d Precomputed 1/sqrt(dim) factor.
 * @return The computed inner product.
 */
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

/**
 * @brief Apply Fast Hadamard Transform rotation to data vector.
 *
 * @param data The data vector to transform.
 * @param dim_ The dimension of the vector.
 */
void
FHTRotate(float* data, uint64_t dim_);

/**
 * @brief Perform rotation operation on data vector.
 *
 * @param data The data vector to rotate.
 * @param idx The rotation index.
 * @param dim_ The dimension of the vector.
 * @param step The rotation step size.
 */
void
RotateOp(float* data, int idx, int dim_, int step);

/**
 * @brief Rescale vector elements by a scalar value.
 *
 * @param data The data vector to rescale.
 * @param dim The dimension of the vector.
 * @param val The scaling factor.
 */
void
VecRescale(float* data, uint64_t dim, float val);

/**
 * @brief Perform Kaczmarz walk on data vector.
 *
 * @param data The data vector to transform.
 * @param len The length of the data vector.
 */
void
KacsWalk(float* data, uint64_t len);

}  // namespace avx2

/**
 * @brief AVX namespace for RaBitQ operations.
 */
namespace avx {

/**
 * @brief Compute inner product between float vector and binary bits.
 *
 * @param vector The float vector.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @param inv_sqrt_d Precomputed 1/sqrt(dim) factor.
 * @return The computed inner product.
 */
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

/**
 * @brief Rescale vector elements by a scalar value.
 *
 * @param data The data vector to rescale.
 * @param dim The dimension of the vector.
 * @param val The scaling factor.
 */
void
VecRescale(float* data, uint64_t dim, float val);

/**
 * @brief Apply Fast Hadamard Transform rotation to data vector.
 *
 * @param data The data vector to transform.
 * @param dim_ The dimension of the vector.
 */
void
FHTRotate(float* data, uint64_t dim_);

/**
 * @brief Flip sign of vector elements based on flip bits.
 *
 * @param flip The flip bits pattern.
 * @param data The data vector to modify.
 * @param dim The dimension of the vector.
 */
void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

/**
 * @brief Perform rotation operation on data vector.
 *
 * @param data The data vector to rotate.
 * @param idx The rotation index.
 * @param dim_ The dimension of the vector.
 * @param step The rotation step size.
 */
void
RotateOp(float* data, int idx, int dim_, int step);

/**
 * @brief Perform Kaczmarz walk on data vector.
 *
 * @param data The data vector to transform.
 * @param len The length of the data vector.
 */
void
KacsWalk(float* data, uint64_t len);

}  // namespace avx

/**
 * @brief SSE namespace for RaBitQ operations.
 */
namespace sse {

/**
 * @brief Compute inner product between float vector and binary bits.
 *
 * @param vector The float vector.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @param inv_sqrt_d Precomputed 1/sqrt(dim) factor.
 * @return The computed inner product.
 */
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

/**
 * @brief Apply Fast Hadamard Transform rotation to data vector.
 *
 * @param data The data vector to transform.
 * @param dim_ The dimension of the vector.
 */
void
FHTRotate(float* data, uint64_t dim_);

/**
 * @brief Perform rotation operation on data vector.
 *
 * @param data The data vector to rotate.
 * @param idx The rotation index.
 * @param dim_ The dimension of the vector.
 * @param step The rotation step size.
 */
void
RotateOp(float* data, int idx, int dim_, int step);

/**
 * @brief Rescale vector elements by a scalar value.
 *
 * @param data The data vector to rescale.
 * @param dim The dimension of the vector.
 * @param val The scaling factor.
 */
void
VecRescale(float* data, uint64_t dim, float val);

/**
 * @brief Perform Kaczmarz walk on data vector.
 *
 * @param data The data vector to transform.
 * @param len The length of the data vector.
 */
void
KacsWalk(float* data, uint64_t len);

}  // namespace sse

/**
 * @brief Generic (scalar) namespace for RaBitQ operations.
 */
namespace generic {

/**
 * @brief Compute inner product between float vector and binary bits.
 *
 * @param vector The float vector.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @param inv_sqrt_d Precomputed 1/sqrt(dim) factor.
 * @return The computed inner product.
 */
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

/**
 * @brief Compute inner product between float vector and SQ codes.
 *
 * @param vector The float vector.
 * @param codes The SQ quantized codes.
 * @param dim The dimension of the vectors.
 * @return The computed inner product.
 */
float
RaBitQFloatSQIP(const float* vector, const uint8_t* codes, uint64_t dim);

/**
 * @brief Compute binary inner product between SQ4U codes and bits.
 *
 * @param codes The SQ4U quantized codes.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @return The computed binary inner product.
 */
uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

/**
 * @brief Perform Kaczmarz walk on data vector.
 *
 * @param data The data vector to transform.
 * @param len The length of the data vector.
 */
void
KacsWalk(float* data, uint64_t len);

/**
 * @brief Rescale vector elements by a scalar value.
 *
 * @param data The data vector to rescale.
 * @param dim The dimension of the vector.
 * @param val The scaling factor.
 */
void
VecRescale(float* data, uint64_t dim, float val);

/**
 * @brief Apply Fast Hadamard Transform rotation to data vector.
 *
 * @param data The data vector to transform.
 * @param dim_ The dimension of the vector.
 */
void
FHTRotate(float* data, uint64_t dim_);

/**
 * @brief Flip sign of vector elements based on flip bits.
 *
 * @param flip The flip bits pattern.
 * @param data The data vector to modify.
 * @param dim The dimension of the vector.
 */
void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

/**
 * @brief Perform rotation operation on data vector.
 *
 * @param data The data vector to rotate.
 * @param idx The rotation index.
 * @param dim_ The dimension of the vector.
 * @param step The rotation step size.
 */
void
RotateOp(float* data, int idx, int dim_, int step);

}  // namespace generic

/**
 * @brief NEON namespace for RaBitQ operations.
 */
namespace neon {

/**
 * @brief Compute inner product between float vector and binary bits.
 *
 * @param vector The float vector.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @param inv_sqrt_d Precomputed 1/sqrt(dim) factor.
 * @return The computed inner product.
 */
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

/**
 * @brief Compute binary inner product between SQ4U codes and bits.
 *
 * @param codes The SQ4U quantized codes.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @return The computed binary inner product.
 */
uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

/**
 * @brief Perform Kaczmarz walk on data vector.
 *
 * @param data The data vector to transform.
 * @param len The length of the data vector.
 */
void
KacsWalk(float* data, uint64_t len);

/**
 * @brief Rescale vector elements by a scalar value.
 *
 * @param data The data vector to rescale.
 * @param dim The dimension of the vector.
 * @param val The scaling factor.
 */
void
VecRescale(float* data, uint64_t dim, float val);

/**
 * @brief Apply Fast Hadamard Transform rotation to data vector.
 *
 * @param data The data vector to transform.
 * @param dim_ The dimension of the vector.
 */
void
FHTRotate(float* data, uint64_t dim_);

/**
 * @brief Flip sign of vector elements based on flip bits.
 *
 * @param flip The flip bits pattern.
 * @param data The data vector to modify.
 * @param dim The dimension of the vector.
 */
void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

/**
 * @brief Perform rotation operation on data vector.
 *
 * @param data The data vector to rotate.
 * @param idx The rotation index.
 * @param dim_ The dimension of the vector.
 * @param step The rotation step size.
 */
void
RotateOp(float* data, int idx, int dim_, int step);

}  // namespace neon

/**
 * @brief SVE (Scalable Vector Extension) namespace for RaBitQ operations.
 */
namespace sve {

/**
 * @brief Compute inner product between float vector and binary bits.
 *
 * @param vector The float vector.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @param inv_sqrt_d Precomputed 1/sqrt(dim) factor.
 * @return The computed inner product.
 */
float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d);

/**
 * @brief Compute binary inner product between SQ4U codes and bits.
 *
 * @param codes The SQ4U quantized codes.
 * @param bits The binary bits vector.
 * @param dim The dimension of the vectors.
 * @return The computed binary inner product.
 */
uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

/**
 * @brief Perform Kaczmarz walk on data vector.
 *
 * @param data The data vector to transform.
 * @param len The length of the data vector.
 */
void
KacsWalk(float* data, uint64_t len);

/**
 * @brief Rescale vector elements by a scalar value.
 *
 * @param data The data vector to rescale.
 * @param dim The dimension of the vector.
 * @param val The scaling factor.
 */
void
VecRescale(float* data, uint64_t dim, float val);

/**
 * @brief Apply Fast Hadamard Transform rotation to data vector.
 *
 * @param data The data vector to transform.
 * @param dim_ The dimension of the vector.
 */
void
FHTRotate(float* data, uint64_t dim_);

/**
 * @brief Flip sign of vector elements based on flip bits.
 *
 * @param flip The flip bits pattern.
 * @param data The data vector to modify.
 * @param dim The dimension of the vector.
 */
void
FlipSign(const uint8_t* flip, float* data, uint64_t dim);

/**
 * @brief Perform rotation operation on data vector.
 *
 * @param data The data vector to rotate.
 * @param idx The rotation index.
 * @param dim_ The dimension of the vector.
 * @param step The rotation step size.
 */
void
RotateOp(float* data, int idx, int dim_, int step);

}  // namespace sve

/**
 * @brief Function pointer type for computing inner product between float vector and binary bits.
 */
using RaBitQFloatBinaryType = float (*)(const float* vector,
                                        const uint8_t* bits,
                                        uint64_t dim,
                                        float inv_sqrt_d);

/**
 * @brief Function pointer type for computing binary inner product between SQ4U codes and bits.
 */
using RaBitQSQ4UBinaryType = uint32_t (*)(const uint8_t* codes, const uint8_t* bits, uint64_t dim);

/**
 * @brief Function pointer type for computing inner product between float vector and SQ codes.
 */
using RaBitQFloatSQType = float (*)(const float* vector, const uint8_t* codes, uint64_t dim);

/**
 * @brief Function pointer type for Fast Hadamard Transform rotation.
 */
using FHTRotateType = void (*)(float* data, uint64_t dim_);

/**
 * @brief Function pointer type for Kaczmarz walk transformation.
 */
using KacsWalkType = void (*)(float* data, uint64_t len);

/**
 * @brief Function pointer type for vector rescaling.
 */
using VecRescaleType = void (*)(float* data, uint64_t dim, float val);

/**
 * @brief Function pointer type for sign flipping operation.
 */
using FlipSignType = void (*)(const uint8_t* flip, float* data, uint64_t dim);

/**
 * @brief Function pointer type for rotation operation.
 */
using RotateOpType = void (*)(float* data, int idx, int dim_, int step);

/**
 * @brief Global function pointer for float-binary inner product.
 * @see RaBitQFloatBinaryType
 */
extern RaBitQFloatBinaryType RaBitQFloatBinaryIP;

/**
 * @brief Global function pointer for float-SQ inner product.
 * @see RaBitQFloatSQType
 */
extern RaBitQFloatSQType RaBitQFloatSQIP;

/**
 * @brief Global function pointer for SQ4U-binary inner product.
 * @see RaBitQSQ4UBinaryType
 */
extern RaBitQSQ4UBinaryType RaBitQSQ4UBinaryIP;

/**
 * @brief Global function pointer for Fast Hadamard Transform rotation.
 * @see FHTRotateType
 */
extern FHTRotateType FHTRotate;

/**
 * @brief Global function pointer for Kaczmarz walk transformation.
 * @see KacsWalkType
 */
extern KacsWalkType KacsWalk;

/**
 * @brief Global function pointer for vector rescaling.
 * @see VecRescaleType
 */
extern VecRescaleType VecRescale;

/**
 * @brief Global function pointer for sign flipping operation.
 * @see FlipSignType
 */
extern FlipSignType FlipSign;

/**
 * @brief Global function pointer for rotation operation.
 * @see RotateOpType
 */
extern RotateOpType RotateOp;

}  // namespace vsag