
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

#include "local_scalar_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "simd/kernels/local_sq_compute.h"
#include "simd/normalize.h"
#include "simd/sq4_local_simd.h"
#include "simd/sq8_local_simd.h"

namespace vsag {
namespace {
int
checked_dimension(int64_t dim) {
    const bool valid = dim > 0 && dim <= std::numeric_limits<int>::max();
    CHECK_ARGUMENT(valid, "local SQ dimension must be in [1, INT_MAX]");
    return static_cast<int>(dim);
}
}  // namespace

template <int bit>
LocalScalarQuantizer<bit>::LocalScalarQuantizer(int dim, Allocator* allocator)
    : Quantizer<LocalScalarQuantizer<bit>>(dim, allocator) {
    CHECK_ARGUMENT(dim > 0, "local SQ requires a positive dimension");
    this->code_size_ = (this->dim_ * bit + 7) / 8 + 2 * sizeof(float);
    this->query_code_size_ = this->dim_ * sizeof(float);
    this->metric_ = MetricType::METRIC_TYPE_L2SQR;
}

template <int bit>
LocalScalarQuantizer<bit>::LocalScalarQuantizer(const QuantizerParamPtr& param,
                                                const IndexCommonParam& common_param)
    : LocalScalarQuantizer(checked_dimension(common_param.dim_), common_param.allocator_.get()) {
    CHECK_ARGUMENT(common_param.data_type_ == DataTypes::DATA_TYPE_FLOAT,
                   "local SQ supports only float32");
    const bool supported = common_param.metric_ == MetricType::METRIC_TYPE_L2SQR ||
                           common_param.metric_ == MetricType::METRIC_TYPE_IP ||
                           common_param.metric_ == MetricType::METRIC_TYPE_COSINE;
    CHECK_ARGUMENT(supported, "local SQ supports only L2, IP and COSINE");
    this->metric_ = common_param.metric_;
}

template <int bit>
bool
LocalScalarQuantizer<bit>::TrainImpl(const float* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

template <int bit>
bool
LocalScalarQuantizer<bit>::EncodeOneImpl(const float* data, uint8_t* codes) const {
    // Validate before normalization, including when a nonfinite norm would hide bad input.
    for (uint64_t i = 0; i < this->dim_; ++i) {
        uint32_t bits;
        memcpy(&bits, data + i, sizeof(bits));
        CHECK_ARGUMENT((bits & 0x7f800000U) != 0x7f800000U, "local SQ requires finite input");
    }
    Vector<float> normalized(this->allocator_);
    if (this->metric_ == MetricType::METRIC_TYPE_COSINE) {
        normalized.resize(this->dim_);
        Normalize(data, normalized.data(), this->dim_);
        data = normalized.data();
    }
    float minimum = data[0];
    float maximum = data[0];
    for (uint64_t i = 0; i < this->dim_; ++i) {
        uint32_t bits;
        memcpy(&bits, data + i, sizeof(bits));
        CHECK_ARGUMENT((bits & 0x7f800000U) != 0x7f800000U, "local SQ requires finite input");
        minimum = std::min(minimum, data[i]);
        maximum = std::max(maximum, data[i]);
    }
    // Compute the range in double to avoid overflow for finite float endpoints.
    const double range = static_cast<double>(maximum) - minimum;
    const auto step = static_cast<float>(range / ((1U << bit) - 1));
    memcpy(codes, &minimum, sizeof(float));
    memcpy(codes + sizeof(float), &step, sizeof(float));
    memset(codes + 2 * sizeof(float), 0, this->code_size_ - 2 * sizeof(float));
    for (uint64_t i = 0; i < this->dim_; ++i) {
        const double scaled = step == 0.0F ? 0.0 : (static_cast<double>(data[i]) - minimum) / step;
        codes[2 * sizeof(float) + i * bit / 8] |=
            static_cast<uint8_t>(std::clamp(std::round(scaled), 0.0, double((1U << bit) - 1)))
            << (i * bit % 8);
    }
    return true;
}

template <int bit>
bool
LocalScalarQuantizer<bit>::DecodeOneImpl(const uint8_t* codes, float* data) {
    float minimum;
    float step;
    memcpy(&minimum, codes, sizeof(float));
    memcpy(&step, codes + sizeof(float), sizeof(float));
    for (uint64_t i = 0; i < this->dim_; ++i) {
        data[i] = simd::LocalSQDecode(
            simd::LocalSQCode<bit>(codes + 2 * sizeof(float), i), step, minimum);
    }
    return true;
}

template <int bit>
float
LocalScalarQuantizer<bit>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    const bool l2 = this->metric_ == MetricType::METRIC_TYPE_L2SQR;
    if constexpr (bit == 8) {
        return l2 ? SQ8LocalComputeCodesL2Sqr(codes1, codes2, this->dim_)
                  : 1 - SQ8LocalComputeCodesIP(codes1, codes2, this->dim_);
    } else {
        return l2 ? SQ4LocalComputeCodesL2Sqr(codes1, codes2, this->dim_)
                  : 1 - SQ4LocalComputeCodesIP(codes1, codes2, this->dim_);
    }
}

template <int bit>
void
LocalScalarQuantizer<bit>::ProcessQueryImpl(const float* query,
                                            Computer<LocalScalarQuantizer<bit>>& computer) const {
    for (uint64_t i = 0; i < this->dim_; ++i) {
        uint32_t bits;
        memcpy(&bits, query + i, sizeof(bits));
        CHECK_ARGUMENT((bits & 0x7f800000U) != 0x7f800000U, "local SQ requires finite queries");
    }
    if (computer.buf_ == nullptr) {
        computer.buf_ = static_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
    }
    if (this->metric_ == MetricType::METRIC_TYPE_COSINE) {
        Normalize(query, reinterpret_cast<float*>(computer.buf_), this->dim_);
    } else {
        memcpy(computer.buf_, query, this->query_code_size_);
    }
}

template <int bit>
void
LocalScalarQuantizer<bit>::ComputeDistImpl(Computer<LocalScalarQuantizer<bit>>& computer,
                                           const uint8_t* codes,
                                           float* dists) const {
    const auto* query = reinterpret_cast<const float*>(computer.buf_);
    const bool l2 = this->metric_ == MetricType::METRIC_TYPE_L2SQR;
    if constexpr (bit == 8) {
        *dists = l2 ? SQ8LocalComputeL2Sqr(query, codes, this->dim_)
                    : 1 - SQ8LocalComputeIP(query, codes, this->dim_);
    } else {
        *dists = l2 ? SQ4LocalComputeL2Sqr(query, codes, this->dim_)
                    : 1 - SQ4LocalComputeIP(query, codes, this->dim_);
    }
}

template <int bit>
void
LocalScalarQuantizer<bit>::DeserializeImpl(StreamReader& reader) {
    const bool valid = this->dim_ > 0 && this->dim_ <= std::numeric_limits<int>::max() &&
                       this->code_size_ == (this->dim_ * bit + 7) / 8 + 2 * sizeof(float) &&
                       (this->metric_ == MetricType::METRIC_TYPE_L2SQR ||
                        this->metric_ == MetricType::METRIC_TYPE_IP ||
                        this->metric_ == MetricType::METRIC_TYPE_COSINE);
    CHECK_ARGUMENT(valid, "invalid local SQ metadata");
    this->query_code_size_ = this->dim_ * sizeof(float);
}
template class LocalScalarQuantizer<4>;
template class LocalScalarQuantizer<8>;
}  // namespace vsag
