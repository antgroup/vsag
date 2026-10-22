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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

#include "common.h"
#include "impl/transform/fht_kac_rotate_transformer.h"
#include "index_common_param.h"
#include "quantization/computer.h"
#include "simd/normalize.h"
#include "simd/sq_local_split_simd.h"
#include "sq_local_split_quantizer_parameter.h"

namespace vsag {

/**
 * Rotated local scalar quantizer with split codes:  d_hat = v_min + delta * c
 *
 * - d is randomly rotated with the same FHT + random-sign rotation RaBitQ uses, so the
 *   rotated coordinates are close to uniform / isotropic; a per-vector min/step range is
 *   then a much tighter fit than a global (per-coordinate) range.
 * - c is (x + y) bits per dimension, split into the high x bits ("filter" record, stored
 *   together with the per-vector metadata) and the low y bits ("supplement" record).
 * - Inner product decomposition (q = rotated query, c = (hi << y) | lo):
 *
 *       <q, d_hat> = v_min * sum(q) + delta * <q, c>
 *
 *   When the query is quantized as q_hat = q_min + delta_q * c_q, the per-vector
 *   constant sum(c) is required:
 *
 *       <q_hat, d_hat> = dim * q_min * v_min + q_min * delta * sum(c)
 *                        + v_min * delta_q * sum(c_q) + delta_q * delta * <c_q, c>
 *
 * The filter (high bits) alone yields a lower bound on the inner product.
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SQLocalSplitQuantizer {
public:
    // per-vector metadata: float minimum, float delta, float |d_hat|^2, uint32 sum(c)
    static constexpr uint64_t META_SIZE = 16;

    class Computer : public ComputerInterface {
    public:
        Computer(uint64_t dim, uint64_t query_bits, Allocator* allocator)
            : rotated_query_(allocator), query_codes_(allocator) {
            rotated_query_.resize(dim);
            if (query_bits > 0) {
                query_codes_.resize(dim);
            }
        }

        // rotated FP32 query
        Vector<float> rotated_query_;
        float sum_q{0.0F};
        float sum_positive_q{0.0F};
        float norm_q_sqr{0.0F};

        // optional SQ8 quantized rotated query (one byte per dimension)
        float q_minimum{0.0F};
        float q_delta{0.0F};
        uint64_t q_sum_codes{0};
        uint64_t q_positive_sum_codes{0};
        Vector<uint8_t> query_codes_;
    };

    SQLocalSplitQuantizer(int64_t dim,
                          uint64_t filter_bits,
                          uint64_t supplement_bits,
                          uint64_t query_bits,
                          Allocator* allocator)
        : dim_(dim),
          filter_bits_(filter_bits),
          base_bits_(filter_bits + supplement_bits),
          supplement_bits_(supplement_bits),
          // internal representation: 0 means a float query, 8 means an SQ8 query
          query_bits_(query_bits == 32 ? 0 : query_bits),
          allocator_(allocator) {
        CHECK_ARGUMENT(dim_ > 0, "sq_local_split requires a positive dimension");
        CHECK_ARGUMENT(filter_bits_ >= 1 and supplement_bits_ >= 1 and base_bits_ <= 8,
                       "sq_local_split requires 1 <= x, 1 <= y and x + y <= 8");
        CHECK_ARGUMENT(query_bits == 32 or query_bits == 8,
                       "sq_local_split query bits must be 32 (float) or 8 (SQ8)");
        this->filter_code_size_ = META_SIZE + (dim_ * filter_bits_ + 7) / 8 + 1;
        this->supplement_code_size_ = (dim_ * supplement_bits_ + 7) / 8 + 1;
        this->full_code_size_ = (dim_ * base_bits_ + 7) / 8;
        this->scalar_code_size_ = META_SIZE + dim_;

        this->rotator_ = std::make_unique<FhtKacRotator>(allocator_, dim_);
        // Fixed-seed rotation; deterministic and cheap, so no data-dependent training is needed.
        this->rotator_->Train();
    }

    SQLocalSplitQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param)
        : SQLocalSplitQuantizer(
              common_param.dim_,
              std::dynamic_pointer_cast<SQLocalSplitQuantizerParameter>(param) != nullptr
                  ? std::dynamic_pointer_cast<SQLocalSplitQuantizerParameter>(param)->FilterBits()
                  : 3,
              std::dynamic_pointer_cast<SQLocalSplitQuantizerParameter>(param) != nullptr
                  ? std::dynamic_pointer_cast<SQLocalSplitQuantizerParameter>(param)->PreciseBits()
                  : 5,
              std::dynamic_pointer_cast<SQLocalSplitQuantizerParameter>(param) != nullptr
                  ? std::dynamic_pointer_cast<SQLocalSplitQuantizerParameter>(param)->QueryBits()
                  : 32,
              common_param.allocator_.get()) {
        CHECK_ARGUMENT(common_param.data_type_ == DataTypes::DATA_TYPE_FLOAT,
                       "sq_local_split supports only float32");
        CHECK_ARGUMENT(common_param.metric_ == MetricType::METRIC_TYPE_L2SQR or
                           common_param.metric_ == MetricType::METRIC_TYPE_IP or
                           common_param.metric_ == MetricType::METRIC_TYPE_COSINE,
                       "sq_local_split supports only L2, IP and COSINE");
        CHECK_ARGUMENT(common_param.metric_ == metric, "sq_local_split metric mismatch");
    }

    [[nodiscard]] int64_t
    GetDim() const {
        return dim_;
    }

    [[nodiscard]] uint64_t
    FilterBits() const {
        return filter_bits_;
    }

    [[nodiscard]] uint64_t
    BaseBits() const {
        return base_bits_;
    }

    [[nodiscard]] uint64_t
    SupplementBits() const {
        return supplement_bits_;
    }

    [[nodiscard]] uint64_t
    QueryBits() const {
        return query_bits_;
    }

    [[nodiscard]] uint64_t
    GetFilterCodeSize() const {
        return filter_code_size_;
    }

    [[nodiscard]] uint64_t
    GetSupplementCodeSize() const {
        return supplement_code_size_;
    }

    [[nodiscard]] uint64_t
    GetFullCodeSize() const {
        return full_code_size_;
    }

    /** Build-time record: metadata followed by one code byte per dimension. */
    [[nodiscard]] uint64_t
    GetScalarCodeSize() const {
        return scalar_code_size_;
    }

    void
    Train(const float* /*data*/, uint64_t /*count*/) {
        // rotation is deterministic; nothing data-dependent to learn
    }

    // ------------------------------------------------------------------ encode/decode

    bool
    Encode(const float* data, uint8_t* filter_code, uint8_t* supplement_code) const {
        CHECK_ARGUMENT(data != nullptr and filter_code != nullptr and supplement_code != nullptr,
                       "sq_local_split encode requires non-null buffers");
        Vector<float> rotated(allocator_);
        rotated.resize(dim_);
        this->rotate(data, rotated.data());

        float minimum = rotated[0];
        float maximum = rotated[0];
        for (int64_t i = 0; i < dim_; ++i) {
            minimum = std::min(minimum, rotated[i]);
            maximum = std::max(maximum, rotated[i]);
        }
        const double range = static_cast<double>(maximum) - static_cast<double>(minimum);
        const float delta = static_cast<float>(range / static_cast<double>(MaxCode()));

        std::memset(filter_code, 0, filter_code_size_);
        std::memset(supplement_code, 0, supplement_code_size_);
        const uint64_t hi_mask = (1ULL << filter_bits_) - 1;
        const uint64_t lo_mask = (1ULL << supplement_bits_) - 1;
        uint64_t sum_codes = 0;
        double norm_sqr = 0.0;
        for (int64_t i = 0; i < dim_; ++i) {
            const double scaled =
                delta == 0.0F ? 0.0 : (static_cast<double>(rotated[i]) - minimum) / delta;
            const auto code = static_cast<uint32_t>(
                std::clamp(std::round(scaled), 0.0, static_cast<double>(MaxCode())));
            const auto hi = static_cast<uint32_t>((code >> supplement_bits_) & hi_mask);
            const auto lo = static_cast<uint32_t>(code & lo_mask);
            SetBits(filter_code + META_SIZE, i * filter_bits_, filter_bits_, hi);
            SetBits(supplement_code, i * supplement_bits_, supplement_bits_, lo);
            sum_codes += code;
            const double value = static_cast<double>(minimum) + static_cast<double>(delta) * code;
            norm_sqr += value * value;
        }
        CHECK_ARGUMENT(sum_codes <= std::numeric_limits<uint32_t>::max(),
                       "sq_local_split dim is too large for the sum-of-codes constant");
        WriteMeta(filter_code,
                  minimum,
                  delta,
                  static_cast<float>(norm_sqr),
                  static_cast<uint32_t>(sum_codes));
        return true;
    }

    /** Build-time code: metadata followed by one code byte per dimension. */
    bool
    EncodeScalar(const float* data, uint8_t* scalar_code) const {
        CHECK_ARGUMENT(data != nullptr and scalar_code != nullptr,
                       "sq_local_split scalar encode requires non-null buffers");
        Vector<float> rotated(allocator_);
        rotated.resize(dim_);
        this->rotate(data, rotated.data());
        float minimum = rotated[0];
        float maximum = rotated[0];
        for (int64_t i = 1; i < dim_; ++i) {
            minimum = std::min(minimum, rotated[i]);
            maximum = std::max(maximum, rotated[i]);
        }
        const double range = static_cast<double>(maximum) - static_cast<double>(minimum);
        const float delta = static_cast<float>(range / static_cast<double>(MaxCode()));
        uint64_t sum_codes = 0;
        double norm_sqr = 0.0;
        for (int64_t i = 0; i < dim_; ++i) {
            const double scaled =
                delta == 0.0F ? 0.0 : (static_cast<double>(rotated[i]) - minimum) / delta;
            const auto code = static_cast<uint32_t>(
                std::clamp(std::round(scaled), 0.0, static_cast<double>(MaxCode())));
            scalar_code[META_SIZE + i] = static_cast<uint8_t>(code);
            sum_codes += code;
            const double value = static_cast<double>(minimum) + static_cast<double>(delta) * code;
            norm_sqr += value * value;
        }
        WriteMeta(scalar_code,
                  minimum,
                  delta,
                  static_cast<float>(norm_sqr),
                  static_cast<uint32_t>(sum_codes));
        return true;
    }

    /** Converts a build-time scalar code into the packed filter + supplement records. */
    void
    PackScalarToSplit(const uint8_t* scalar_code,
                      uint8_t* filter_code,
                      uint8_t* supplement_code) const {
        std::memcpy(filter_code, scalar_code, META_SIZE);
        std::memset(filter_code + META_SIZE, 0, filter_code_size_ - META_SIZE);
        std::memset(supplement_code, 0, supplement_code_size_);
        // Accumulate whole bytes instead of writing bit by bit: one shift-or per dimension.
        PackStream(
            scalar_code + META_SIZE, filter_code + META_SIZE, dim_, filter_bits_, supplement_bits_);
        PackStream(scalar_code + META_SIZE, supplement_code, dim_, supplement_bits_, 0);
    }

    /** Packs `dim` codes, optionally shifted right by `shift`, `bits` bits per code. */
    static void
    PackStream(const uint8_t* codes, uint8_t* out, int64_t dim, uint64_t bits, uint64_t shift) {
        const uint64_t mask = (1ULL << bits) - 1ULL;
        uint64_t acc = 0;
        uint64_t pending = 0;
        for (int64_t i = 0; i < dim; ++i) {
            const auto code = static_cast<uint64_t>((codes[i] >> shift) & mask);
            acc |= code << pending;
            pending += bits;
            while (pending >= 8) {
                *out++ = static_cast<uint8_t>(acc & 0xFFULL);
                acc >>= 8;
                pending -= 8;
            }
        }
        if (pending > 0) {
            *out = static_cast<uint8_t>(acc & 0xFFULL);
        }
    }

    bool
    Decode(const uint8_t* filter_code, const uint8_t* supplement_code, float* out) const {
        CHECK_ARGUMENT(filter_code != nullptr and out != nullptr,
                       "sq_local_split decode requires a filter record");
        const float minimum = MetaMinimum(filter_code);
        const float delta = MetaDelta(filter_code);
        Vector<float> rotated(allocator_);
        rotated.resize(dim_);
        for (int64_t i = 0; i < dim_; ++i) {
            const uint32_t code = (GetBits(filter_code + META_SIZE, i * filter_bits_, filter_bits_)
                                   << supplement_bits_) |
                                  GetBits(supplement_code, i * supplement_bits_, supplement_bits_);
            rotated[i] = std::fma(static_cast<float>(code), delta, minimum);
        }
        this->rotator_->InverseTransform(rotated.data(), out);
        return true;
    }

    // ------------------------------------------------------------------ query

    [[nodiscard]] std::shared_ptr<Computer>
    FactoryComputer(const void* query) const {
        auto computer = std::make_shared<Computer>(dim_, query_bits_, allocator_);
        this->ProcessQuery(*computer, static_cast<const float*>(query));
        return computer;
    }

    void
    ProcessQuery(Computer& computer, const float* query) const {
        CHECK_ARGUMENT(query != nullptr, "sq_local_split query must not be null");
        for (int64_t i = 0; i < dim_; ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, query + i, sizeof(bits));
            CHECK_ARGUMENT((bits & 0x7F800000U) != 0x7F800000U,
                           "sq_local_split requires finite queries");
        }
        Vector<float> normalized(allocator_);
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            normalized.resize(dim_);
            Normalize(query, normalized.data(), dim_);
            query = normalized.data();
        }
        this->rotator_->Transform(query, computer.rotated_query_.data(), nullptr);
        this->FillQuerySums(computer);
        if (query_bits_ > 0) {
            this->quantize_query(computer);
        }
    }

    /** Rebuilds a query computer from a packed code (already in the rotated space). */
    void
    RestoreComputer(Computer& computer,
                    const uint8_t* filter_code,
                    const uint8_t* supplement_code) const {
        const float minimum = MetaMinimum(filter_code);
        const float delta = MetaDelta(filter_code);
        if (static_cast<int64_t>(computer.rotated_query_.size()) != dim_) {
            computer.rotated_query_.resize(dim_);
        }
        for (int64_t i = 0; i < dim_; ++i) {
            const uint32_t code = (GetBits(filter_code + META_SIZE, i * filter_bits_, filter_bits_)
                                   << supplement_bits_) |
                                  GetBits(supplement_code, i * supplement_bits_, supplement_bits_);
            computer.rotated_query_[i] = std::fma(static_cast<float>(code), delta, minimum);
        }
        this->FillQuerySums(computer);
        computer.q_minimum = 0.0F;
        computer.q_delta = 0.0F;
        computer.q_sum_codes = 0;
        computer.q_positive_sum_codes = 0;
    }

    /** Rebuilds a query computer from a build-time scalar code. */
    void
    RestoreComputerFromScalar(Computer& computer, const uint8_t* scalar_code) const {
        const float minimum = MetaMinimum(scalar_code);
        const float delta = MetaDelta(scalar_code);
        if (static_cast<int64_t>(computer.rotated_query_.size()) != dim_) {
            computer.rotated_query_.resize(dim_);
        }
        for (int64_t i = 0; i < dim_; ++i) {
            computer.rotated_query_[i] =
                std::fma(static_cast<float>(scalar_code[META_SIZE + i]), delta, minimum);
        }
        this->FillQuerySums(computer);
        computer.q_minimum = 0.0F;
        computer.q_delta = 0.0F;
        computer.q_sum_codes = 0;
        computer.q_positive_sum_codes = 0;
    }

    // ------------------------------------------------------------------ distances

    [[nodiscard]] float
    ComputeFullDistance(const Computer& computer,
                        const uint8_t* filter_code,
                        const uint8_t* supplement_code) const {
        return this->MetricDistance(this->InnerProduct(computer, filter_code, supplement_code),
                                    computer.norm_q_sqr,
                                    MetaNormSqr(filter_code));
    }

    /** Code-to-code distance between two build-time scalar codes. */
    [[nodiscard]] float
    ComputeScalarPairDistance(const uint8_t* scalar1, const uint8_t* scalar2) const {
        const float m1 = MetaMinimum(scalar1);
        const float d1 = MetaDelta(scalar1);
        const float m2 = MetaMinimum(scalar2);
        const float d2 = MetaDelta(scalar2);
        const float dot =
            SQByteByteIP(scalar1 + META_SIZE, scalar2 + META_SIZE, static_cast<uint64_t>(dim_));
        const double ip = static_cast<double>(dim_) * m1 * m2 +
                          static_cast<double>(m1) * d2 * MetaSumCodes(scalar2) +
                          static_cast<double>(m2) * d1 * MetaSumCodes(scalar1) +
                          static_cast<double>(d1) * d2 * dot;
        return this->MetricDistance(
            static_cast<float>(ip), MetaNormSqr(scalar1), MetaNormSqr(scalar2));
    }

    /** Distance against a build-time scalar code (one byte per dimension). */
    [[nodiscard]] float
    ComputeScalarDistance(const Computer& computer, const uint8_t* scalar_code) const {
        const float minimum = MetaMinimum(scalar_code);
        const float delta = MetaDelta(scalar_code);
        const float dot = SQFloatByteIP(
            computer.rotated_query_.data(), scalar_code + META_SIZE, static_cast<uint64_t>(dim_));
        const float ip = minimum * computer.sum_q + delta * dot;
        return this->MetricDistance(ip, computer.norm_q_sqr, MetaNormSqr(scalar_code));
    }

    [[nodiscard]] float
    ComputeFilterDistanceLowerBound(const Computer& computer, const uint8_t* filter_code) const {
        const float upper_ip = this->InnerProductFilterOnly(computer, filter_code) +
                               this->FilterErrorBound(computer, filter_code);
        return this->MetricDistance(upper_ip, computer.norm_q_sqr, MetaNormSqr(filter_code));
    }

    [[nodiscard]] float
    ComputePairDistance(const uint8_t* filter1,
                        const uint8_t* supplement1,
                        const uint8_t* filter2,
                        const uint8_t* supplement2) const {
        const float m1 = MetaMinimum(filter1);
        const float d1 = MetaDelta(filter1);
        const float m2 = MetaMinimum(filter2);
        const float d2 = MetaDelta(filter2);
        const float dot = SQPackedPackedIP(filter1 + META_SIZE,
                                           supplement1,
                                           filter2 + META_SIZE,
                                           supplement2,
                                           static_cast<uint64_t>(dim_),
                                           filter_bits_,
                                           supplement_bits_);
        const double ip = static_cast<double>(dim_) * m1 * m2 +
                          static_cast<double>(m1) * d2 * MetaSumCodes(filter2) +
                          static_cast<double>(m2) * d1 * MetaSumCodes(filter1) +
                          static_cast<double>(d1) * d2 * dot;
        return this->MetricDistance(
            static_cast<float>(ip), MetaNormSqr(filter1), MetaNormSqr(filter2));
    }

    /** Inner product of the (float) query and the reconstructed vector. */
    [[nodiscard]] float
    InnerProduct(const Computer& computer,
                 const uint8_t* filter_code,
                 const uint8_t* supplement_code) const {
        const float minimum = MetaMinimum(filter_code);
        const float delta = MetaDelta(filter_code);
        const uint8_t* hi_codes = filter_code + META_SIZE;
        if (query_bits_ == 0) {
            const float dot = SQFloatPackedIP(computer.rotated_query_.data(),
                                              hi_codes,
                                              supplement_code,
                                              static_cast<uint64_t>(dim_),
                                              filter_bits_,
                                              supplement_bits_);
            return minimum * computer.sum_q + delta * dot;
        }
        const float dot = SQBytePackedIP(computer.query_codes_.data(),
                                         hi_codes,
                                         supplement_code,
                                         static_cast<uint64_t>(dim_),
                                         filter_bits_,
                                         supplement_bits_);
        const float q_min = computer.q_minimum;
        const float q_delta = computer.q_delta;
        return static_cast<float>(static_cast<double>(dim_) * q_min * minimum +
                                  q_min * delta * MetaSumCodes(filter_code) +
                                  minimum * q_delta * computer.q_sum_codes + q_delta * delta * dot);
    }

    /** Filter-only (high x bits) inner product: the low bits are dropped. */
    [[nodiscard]] float
    InnerProductFilterOnly(const Computer& computer, const uint8_t* filter_code) const {
        const double minimum = MetaMinimum(filter_code);
        const double delta = MetaDelta(filter_code);
        const double scale = static_cast<double>(1ULL << supplement_bits_);
        const uint8_t* hi_codes = filter_code + META_SIZE;
        if (query_bits_ == 0) {
            double dot = 0.0;
            for (int64_t i = 0; i < dim_; ++i) {
                dot += static_cast<double>(computer.rotated_query_[i]) *
                       GetBits(hi_codes, i * filter_bits_, filter_bits_);
            }
            return static_cast<float>(minimum * computer.sum_q + delta * scale * dot);
        }
        double dot = 0.0;
        for (int64_t i = 0; i < dim_; ++i) {
            dot += static_cast<double>(computer.query_codes_[i]) *
                   GetBits(hi_codes, i * filter_bits_, filter_bits_);
        }
        const double q_min = computer.q_minimum;
        const double q_delta = computer.q_delta;
        return static_cast<float>(static_cast<double>(dim_) * q_min * minimum +
                                  q_min * delta * MetaSumCodes(filter_code) +
                                  minimum * q_delta * computer.q_sum_codes +
                                  q_delta * delta * scale * dot);
    }

    /** Worst-case inner-product gap caused by dropping the low y bits. */
    [[nodiscard]] float
    FilterErrorBound(const Computer& computer, const uint8_t* filter_code) const {
        const double delta = MetaDelta(filter_code);
        const double scale = static_cast<double>(1ULL << supplement_bits_);
        if (query_bits_ == 0) {
            return static_cast<float>(delta * (scale - 1.0) * computer.sum_positive_q);
        }
        return static_cast<float>(delta * computer.q_delta * (scale - 1.0) *
                                  static_cast<double>(computer.q_positive_sum_codes));
    }

    [[nodiscard]] float
    MetricDistance(float inner_product, float norm_q_sqr, float norm_d_sqr) const {
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            return norm_q_sqr - 2.0F * inner_product + norm_d_sqr;
        }
        return 1.0F - inner_product;
    }

    void
    Serialize(StreamWriter& writer) const {
        this->rotator_->Serialize(writer);
    }

    void
    Deserialize(StreamReader& reader) {
        this->rotator_->Deserialize(reader);
    }

    // ------------------------------------------------------------------ metadata accessors

    [[nodiscard]] static float
    MetaMinimum(const uint8_t* code) {
        float value = 0.0F;
        std::memcpy(&value, code, sizeof(float));
        return value;
    }

    [[nodiscard]] static float
    MetaDelta(const uint8_t* code) {
        float value = 0.0F;
        std::memcpy(&value, code + 4, sizeof(float));
        return value;
    }

    [[nodiscard]] static float
    MetaNormSqr(const uint8_t* code) {
        float value = 0.0F;
        std::memcpy(&value, code + 8, sizeof(float));
        return value;
    }

    [[nodiscard]] static uint32_t
    MetaSumCodes(const uint8_t* code) {
        uint32_t value = 0;
        std::memcpy(&value, code + 12, sizeof(uint32_t));
        return value;
    }

    static void
    WriteMeta(uint8_t* code, float minimum, float delta, float norm_sqr, uint32_t sum_codes) {
        std::memcpy(code, &minimum, sizeof(float));
        std::memcpy(code + 4, &delta, sizeof(float));
        std::memcpy(code + 8, &norm_sqr, sizeof(float));
        std::memcpy(code + 12, &sum_codes, sizeof(uint32_t));
    }

    static void
    SetBits(uint8_t* codes, uint64_t bit_offset, uint64_t bits, uint32_t value) {
        for (uint64_t bit = 0; bit < bits; ++bit) {
            const uint64_t position = bit_offset + bit;
            const auto mask = static_cast<uint8_t>(1U << (position & 7U));
            if (((value >> bit) & 1U) != 0U) {
                codes[position >> 3] |= mask;
            } else {
                codes[position >> 3] &= static_cast<uint8_t>(~mask);
            }
        }
    }

    [[nodiscard]] static uint32_t
    GetBits(const uint8_t* codes, uint64_t bit_offset, uint64_t bits) {
        // Branch-free 16-bit load: every record carries one padding byte, so reading two
        // bytes at the last component stays in bounds. Little-endian hosts only (x86/ARM).
        uint32_t raw = 0;
        std::memcpy(&raw, codes + (bit_offset >> 3), sizeof(uint16_t));
        return (raw >> (bit_offset & 7U)) & static_cast<uint32_t>((1ULL << bits) - 1ULL);
    }

    [[nodiscard]] uint32_t
    MaxCode() const {
        return static_cast<uint32_t>((1ULL << base_bits_) - 1ULL);
    }

private:
    void
    rotate(const float* data, float* rotated) const {
        for (int64_t i = 0; i < dim_; ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, data + i, sizeof(bits));
            CHECK_ARGUMENT((bits & 0x7F800000U) != 0x7F800000U,
                           "sq_local_split requires finite input");
        }
        Vector<float> normalized(allocator_);
        if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
            normalized.resize(dim_);
            Normalize(data, normalized.data(), dim_);
            data = normalized.data();
        }
        this->rotator_->Transform(data, rotated, nullptr);
    }

    void
    FillQuerySums(Computer& computer) const {
        float sum_q = 0.0F;
        float sum_positive_q = 0.0F;
        float norm_q = 0.0F;
        for (int64_t i = 0; i < dim_; ++i) {
            const float q = computer.rotated_query_[i];
            sum_q += q;
            sum_positive_q += std::max(q, 0.0F);
            norm_q += q * q;
        }
        computer.sum_q = sum_q;
        computer.sum_positive_q = sum_positive_q;
        computer.norm_q_sqr = norm_q;
    }

    void
    quantize_query(Computer& computer) const {
        float minimum = computer.rotated_query_[0];
        float maximum = computer.rotated_query_[0];
        for (int64_t i = 1; i < dim_; ++i) {
            minimum = std::min(minimum, computer.rotated_query_[i]);
            maximum = std::max(maximum, computer.rotated_query_[i]);
        }
        const float delta = static_cast<float>((static_cast<double>(maximum) - minimum) /
                                               static_cast<double>((1ULL << query_bits_) - 1));
        uint64_t sum_codes = 0;
        uint64_t positive_sum_codes = 0;
        const double max_code = static_cast<double>((1ULL << query_bits_) - 1);
        for (int64_t i = 0; i < dim_; ++i) {
            const double scaled =
                delta == 0.0F ? 0.0
                              : (static_cast<double>(computer.rotated_query_[i]) - minimum) / delta;
            const auto code = static_cast<uint32_t>(std::clamp(std::round(scaled), 0.0, max_code));
            computer.query_codes_[i] = static_cast<uint8_t>(code);
            sum_codes += code;
            if (computer.rotated_query_[i] > 0.0F) {
                positive_sum_codes += code;
            }
        }
        computer.q_minimum = minimum;
        computer.q_delta = delta;
        computer.q_sum_codes = sum_codes;
        computer.q_positive_sum_codes = positive_sum_codes;
    }

private:
    int64_t dim_{0};
    uint64_t filter_bits_{3};
    uint64_t base_bits_{8};
    uint64_t supplement_bits_{5};
    uint64_t query_bits_{0};
    uint64_t filter_code_size_{0};
    uint64_t supplement_code_size_{0};
    uint64_t full_code_size_{0};
    uint64_t scalar_code_size_{0};
    Allocator* allocator_{nullptr};
    std::unique_ptr<FhtKacRotator> rotator_{nullptr};
};

}  // namespace vsag
