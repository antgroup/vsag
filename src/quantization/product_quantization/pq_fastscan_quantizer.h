
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
 * @file pq_fastscan_quantizer.h
 * @brief PQ FastScan Quantizer implementation for optimized SIMD-based vector compression.
 */

#pragma once

#include "index_common_param.h"
#include "inner_string_params.h"
#include "pq_fastscan_quantizer_parameter.h"
#include "quantization/quantizer.h"

namespace vsag {

/***
 * @brief PQ FastScan Quantizer uses 4-bit packed storage for optimized distance computation.
 *
 * code layout (32 vectors packed):
 * +--------+--------+-----+--------+
 * | sub0   | sub1   | ... | subN   |
 * | [32*4b]| [32*4b]|     | [32*4b]|
 * +--------+--------+-----+--------+
 *
 * query layout:
 * +------------------+--------+--------+
 * | lookup-table     | diff   | lower  |
 * | [pq_dim*16*4B]   | [4B]   | [4B]   |
 * +------------------+--------+--------+
 *
 * - pqfs-code: 4 bits per subspace, packed for 32 vectors at once
 * - Each subspace has 16 centroids (2^4 = 16 possible indices)
 * - Optimized for SIMD batch distance computation
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class PQFastScanQuantizer : public Quantizer<PQFastScanQuantizer<metric>> {
public:
    /**
     * @brief Constructs a PQFastScanQuantizer with specified dimensions.
     * @param dim Vector dimension.
     * @param pq_dim Number of subspaces for product quantization.
     * @param allocator Allocator for memory management.
     */
    explicit PQFastScanQuantizer(int dim, int64_t pq_dim, Allocator* allocator);

    /**
     * @brief Constructs a PQFastScanQuantizer from parameter object.
     * @param param PQ FastScan quantizer specific parameters.
     * @param common_param Common index parameters.
     */
    PQFastScanQuantizer(const PQFastScanQuantizerParamPtr& param,
                        const IndexCommonParam& common_param);

    /**
     * @brief Constructs a PQFastScanQuantizer from base quantizer parameter.
     * @param param Quantizer parameters (will be cast to PQFastScanQuantizerParamPtr).
     * @param common_param Common index parameters.
     */
    PQFastScanQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~PQFastScanQuantizer() override = default;

    /**
     * @brief Trains the quantizer using the provided data.
     * @param data Training data array.
     * @param count Number of vectors in training data.
     * @return true if training succeeded, false otherwise.
     */
    bool
    TrainImpl(const float* data, uint64_t count);

    /**
     * @brief Encodes a single vector into PQ FastScan codes.
     * @param data Input vector data.
     * @param codes Output buffer for encoded codes.
     * @return true if encoding succeeded, false otherwise.
     */
    bool
    EncodeOneImpl(const float* data, uint8_t* codes);

    /**
     * @brief Encodes a batch of vectors into PQ FastScan codes.
     * @param data Input vector data array.
     * @param codes Output buffer for encoded codes.
     * @param count Number of vectors to encode.
     * @return true if encoding succeeded, false otherwise.
     */
    bool
    EncodeBatchImpl(const float* data, uint8_t* codes, uint64_t count);

    /**
     * @brief Decodes PQ FastScan codes back to a vector.
     * @param codes Input encoded codes.
     * @param data Output buffer for decoded vector.
     * @return true if decoding succeeded, false otherwise.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, float* data);

    /**
     * @brief Decodes a batch of PQ FastScan codes back to vectors.
     * @param codes Input encoded codes array.
     * @param data Output buffer for decoded vectors.
     * @param count Number of vectors to decode.
     * @return true if decoding succeeded, false otherwise.
     */
    bool
    DecodeBatchImpl(const uint8_t* codes, float* data, uint64_t count);

    /**
     * @brief Computes distance between two encoded vectors.
     * @param codes1 First encoded vector.
     * @param codes2 Second encoded vector.
     * @return Computed distance.
     */
    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);

    /**
     * @brief Prepares a computer for query processing.
     * @param query Query vector data.
     * @param computer Output computer object for distance computation.
     */
    void
    ProcessQueryImpl(const float* query, Computer<PQFastScanQuantizer>& computer) const;

    /**
     * @brief Computes distance from query to encoded vector.
     * @param computer Computer object containing query lookup table.
     * @param codes Encoded vector.
     * @param dists Output distance value.
     */
    void
    ComputeDistImpl(Computer<PQFastScanQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    /**
     * @brief Computes distances for a batch of encoded vectors.
     * @param computer Computer object containing query lookup table.
     * @param count Number of vectors to process.
     * @param codes Encoded vectors array.
     * @param dists Output distance array.
     */
    void
    ScanBatchDistImpl(Computer<PQFastScanQuantizer<metric>>& computer,
                      uint64_t count,
                      const uint8_t* codes,
                      float* dists) const;

    /**
     * @brief Serializes the quantizer to a stream.
     * @param writer Stream writer for output.
     */
    void
    SerializeImpl(StreamWriter& writer);

    /**
     * @brief Deserializes the quantizer from a stream.
     * @param reader Stream reader for input.
     */
    void
    DeserializeImpl(StreamReader& reader);

    /**
     * @brief Releases resources held by a computer object.
     * @param computer Computer object to release.
     */
    void
    ReleaseComputerImpl(Computer<PQFastScanQuantizer<metric>>& computer) const;

    /**
     * @brief Returns the name of the quantizer type.
     * @return String identifier "pqfs".
     */
    [[nodiscard]] inline std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_PQFS;
    }

    /**
     * @brief Packages 32 encoded vectors into optimized storage format.
     * @param codes Input codes array (32 vectors).
     * @param packaged_codes Output buffer for packaged codes.
     * @param valid_size Number of valid vectors (may be less than 32 for last batch).
     */
    void
    Package32(const uint8_t* codes, uint8_t* packaged_codes, int64_t valid_size) const override;

    /**
     * @brief Unpacks optimized storage format back to individual codes.
     * @param packaged_codes Input packaged codes (32 vectors).
     * @param codes Output buffer for unpacked codes.
     */
    void
    Unpack32(const uint8_t* packaged_codes, uint8_t* codes) const override;

private:
    /**
     * @brief Gets pointer to codebook data for a specific subspace and centroid.
     * @param subspace_idx Index of the subspace.
     * @param centroid_num Index of the centroid within the subspace.
     * @return Pointer to the centroid data.
     */
    [[nodiscard]] inline const float*
    get_codebook_data(int64_t subspace_idx, int64_t centroid_num) const {
        return this->codebooks_.data() + subspace_idx * subspace_dim_ * CENTROIDS_PER_SUBSPACE +
               centroid_num * subspace_dim_;
    }

public:
    static constexpr int64_t PQ_BITS = 4L;  /// Bits per PQ code (4 for FastScan)
    static constexpr int64_t CENTROIDS_PER_SUBSPACE =
        16L;                                            /// Number of centroids per subspace (2^4)
    static constexpr int64_t BLOCK_SIZE_PACKAGE = 32L;  /// Number of vectors packed together
    static constexpr int32_t MAPPER[32] = {
        0,  16, 8,  24, 1,  17, 9,  25, 2,  18, 10, 26, 3,  19, 11, 27, 4,
        20, 12, 28, 5,  21, 13, 29, 6,  22, 14, 30, 7,  23, 15, 31};  /// Index mapping for interleaving

public:
    int64_t pq_dim_{1};        /// Number of subspaces
    int64_t subspace_dim_{1};  /// Dimension of each subspace (dim / pq_dim)

    Vector<float> codebooks_;  /// Codebook centroids (pq_dim * 16 * subspace_dim)
};

}  // namespace vsag
