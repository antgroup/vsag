
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
 * @file product_quantizer.h
 * @brief Product Quantizer implementation for vector compression.
 */

#pragma once

#include "index_common_param.h"
#include "inner_string_params.h"
#include "product_quantizer_parameter.h"
#include "quantization/quantizer.h"

namespace vsag {

/***
 * @brief Product Quantizer divides vectors into subspaces and quantizes each subspace.
 *
 * code layout:
 * +------+------+------+------+
 * | sub0 | sub1 | ...  | subN |
 * | [1B] | [1B] |      | [1B] |
 * +------+------+------+------+
 *
 * query layout (lookup table):
 * +------------------+------------------+          +------------------+
 * | lut[0][0..255]   | lut[1][0..255]   |   ...    | lut[N][0..255]   |
 * | [256 floats]     | [256 floats]     |          | [256 floats]     |
 * +------------------+------------------+          +------------------+
 *
 * - pq-code: 1 byte per subspace, index into codebook (required)
 * - pq_dim: number of subspaces
 * - Each subspace has 256 centroids (2^8 = 256 possible indices)
 * - Lookup table: precomputed distances for query to all centroids
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class ProductQuantizer : public Quantizer<ProductQuantizer<metric>> {
public:
    /**
     * @brief Constructs a ProductQuantizer with specified dimensions.
     * @param dim Vector dimension.
     * @param pq_dim Number of subspaces for product quantization.
     * @param allocator Allocator for memory management.
     */
    explicit ProductQuantizer(int dim, int64_t pq_dim, Allocator* allocator);

    /**
     * @brief Constructs a ProductQuantizer from parameter object.
     * @param param Product quantizer specific parameters.
     * @param common_param Common index parameters.
     */
    ProductQuantizer(const ProductQuantizerParamPtr& param, const IndexCommonParam& common_param);

    /**
     * @brief Constructs a ProductQuantizer from base quantizer parameter.
     * @param param Quantizer parameters (will be cast to ProductQuantizerParamPtr).
     * @param common_param Common index parameters.
     */
    ProductQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~ProductQuantizer() = default;

    /**
     * @brief Trains the quantizer using the provided data.
     * @param data Training data array.
     * @param count Number of vectors in training data.
     * @return true if training succeeded, false otherwise.
     */
    bool
    TrainImpl(const float* data, uint64_t count);

    /**
     * @brief Encodes a single vector into PQ codes.
     * @param data Input vector data.
     * @param codes Output buffer for encoded codes.
     * @return true if encoding succeeded, false otherwise.
     */
    bool
    EncodeOneImpl(const float* data, uint8_t* codes);

    /**
     * @brief Decodes PQ codes back to a vector.
     * @param codes Input encoded codes.
     * @param data Output buffer for decoded vector.
     * @return true if decoding succeeded, false otherwise.
     */
    bool
    DecodeOneImpl(const uint8_t* codes, float* data);

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
    ProcessQueryImpl(const float* query, Computer<ProductQuantizer>& computer) const;

    /**
     * @brief Computes distance from query to encoded vector.
     * @param computer Computer object containing query lookup table.
     * @param codes Encoded vector.
     * @param dists Output distance value.
     */
    void
    ComputeDistImpl(Computer<ProductQuantizer>& computer, const uint8_t* codes, float* dists) const;

    /**
     * @brief Computes distances from query to four encoded vectors in batch.
     * @param computer Computer object containing query lookup table.
     * @param codes1 First encoded vector.
     * @param codes2 Second encoded vector.
     * @param codes3 Third encoded vector.
     * @param codes4 Fourth encoded vector.
     * @param dists1 Output distance for first vector.
     * @param dists2 Output distance for second vector.
     * @param dists3 Output distance for third vector.
     * @param dists4 Output distance for fourth vector.
     */
    void
    ComputeDistsBatch4Impl(Computer<ProductQuantizer<metric>>& computer,
                           const uint8_t* codes1,
                           const uint8_t* codes2,
                           const uint8_t* codes3,
                           const uint8_t* codes4,
                           float& dists1,
                           float& dists2,
                           float& dists3,
                           float& dists4) const;

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
     * @brief Returns the name of the quantizer type.
     * @return String identifier "pq".
     */
    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_PQ;
    }

private:
    /**
     * @brief Gets pointer to codebook data for a specific subspace and centroid.
     * @param subspace_idx Index of the subspace.
     * @param centroid_num Index of the centroid within the subspace.
     * @return Pointer to the centroid data.
     */
    [[nodiscard]] const float*
    get_codebook_data(int64_t subspace_idx, int64_t centroid_num) const {
        return this->codebooks_.data() + subspace_idx * subspace_dim_ * CENTROIDS_PER_SUBSPACE +
               centroid_num * subspace_dim_;
    }

    /**
     * @brief Transposes codebooks for optimized distance computation.
     */
    void
    transpose_codebooks();

public:
    static constexpr int64_t PQ_BITS = 8L;                   /// Bits per PQ code (fixed at 8)
    static constexpr int64_t CENTROIDS_PER_SUBSPACE = 256L;  /// Number of centroids per subspace

public:
    int64_t pq_dim_{1};        /// Number of subspaces
    int64_t subspace_dim_{1};  /// Dimension of each subspace (dim / pq_dim)

    Vector<float> codebooks_;          /// Codebook centroids (pq_dim * 256 * subspace_dim)
    Vector<float> reverse_codebooks_;  /// Transposed codebooks for optimized lookup
};

}  // namespace vsag
