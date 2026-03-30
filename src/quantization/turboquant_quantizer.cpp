
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

#include "turboquant_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "simd/basic_func.h"
#include "simd/fp32_simd.h"
#include "simd/normalize.h"

namespace vsag {

namespace {

constexpr float PI_VALUE = 3.14159265358979323846F;
constexpr float TWO_PI_VALUE = 2.0F * PI_VALUE;
constexpr float ZERO_THRESHOLD = 1e-12F;

float
clamp_value(float value, float low, float high) {
    return std::min(std::max(value, low), high);
}

float
positive_mod(float value, float period) {
    auto mod = std::fmod(value, period);
    if (mod < 0.0F) {
        mod += period;
    }
    return mod;
}

}  // namespace

template <MetricType metric>
TurboQuantizer<metric>::TurboQuantizer(int dim,
                                       uint64_t bits_per_dim,
                                       bool use_fht,
                                       bool enable_qjl,
                                       uint64_t qjl_projection_dim,
                                       Allocator* allocator)
    : Quantizer<TurboQuantizer<metric>>(dim, allocator),
      bits_per_dim_(bits_per_dim),
      use_fht_(use_fht),
      enable_qjl_(enable_qjl),
      qjl_projection_dim_(qjl_projection_dim == 0 ? static_cast<uint64_t>(dim) : qjl_projection_dim) {
    this->metric_ = metric;
    offset_polar_code_ = 0;
    offset_radius_ = get_polar_code_size();
    offset_qjl_code_ = offset_radius_ + sizeof(float);
    offset_raw_norm_ = offset_qjl_code_ + get_qjl_code_size();
    this->code_size_ = offset_raw_norm_ + sizeof(float);
    this->query_code_size_ = this->dim_ * sizeof(float) + sizeof(float);
    if (enable_qjl_) {
        this->query_code_size_ += qjl_projection_dim_ * sizeof(float);
    }

    if (use_fht_) {
        rotator_ = std::make_shared<FhtKacRotator>(allocator, dim);
    } else {
        rotator_ = std::make_shared<RandomOrthogonalMatrix>(allocator, dim);
    }
}

template <MetricType metric>
TurboQuantizer<metric>::TurboQuantizer(const TurboQuantizerParamPtr& param,
                                       const IndexCommonParam& common_param)
    : TurboQuantizer<metric>(common_param.dim_,
                             param->bits_per_dim_,
                             param->use_fht_,
                             param->enable_qjl_,
                             param->qjl_projection_dim_,
                             common_param.allocator_.get()) {
}

template <MetricType metric>
TurboQuantizer<metric>::TurboQuantizer(const QuantizerParamPtr& param,
                                       const IndexCommonParam& common_param)
    : TurboQuantizer<metric>(std::dynamic_pointer_cast<TurboQuantizerParameter>(param),
                             common_param) {
}

template <MetricType metric>
uint64_t
TurboQuantizer<metric>::bits_to_bytes(uint64_t bits) const {
    return (bits + 7) / 8;
}

template <MetricType metric>
uint64_t
TurboQuantizer<metric>::get_polar_code_size() const {
    if (this->dim_ <= 1) {
        return 0;
    }
    return bits_to_bytes((this->dim_ - 1) * bits_per_dim_);
}

template <MetricType metric>
uint64_t
TurboQuantizer<metric>::get_qjl_code_size() const {
    if (not enable_qjl_) {
        return 0;
    }
    return bits_to_bytes(qjl_projection_dim_);
}

template <MetricType metric>
bool
TurboQuantizer<metric>::TrainImpl(const DataType* data, uint64_t count) {
    if (data == nullptr or count == 0) {
        return false;
    }
    if (this->is_trained_) {
        return true;
    }
    rotator_->Train(data, count);
    this->is_trained_ = true;
    return true;
}

template <MetricType metric>
void
TurboQuantizer<metric>::rotate(const DataType* input, std::vector<float>& output) const {
    output.resize(this->dim_);
    rotator_->Transform(input, output.data());
}

template <MetricType metric>
void
TurboQuantizer<metric>::inverse_rotate(const float* input, std::vector<float>& output) const {
    output.resize(this->dim_);
    rotator_->InverseTransform(input, output.data());
}

template <MetricType metric>
void
TurboQuantizer<metric>::cartesian_to_polar(const float* input,
                                           float& radius,
                                           std::vector<float>& angles) const {
    std::vector<float> current(input, input + this->dim_);
    std::vector<float> next;
    angles.clear();

    while (current.size() > 1) {
        next.clear();
        next.reserve((current.size() + 1) / 2);
        auto pair_count = current.size() / 2;
        for (uint64_t i = 0; i < pair_count; ++i) {
            const auto x = current[2 * i];
            const auto y = current[2 * i + 1];
            next.push_back(std::sqrt(x * x + y * y));
            angles.push_back(std::atan2(y, x));
        }
        if (current.size() % 2 == 1) {
            next.push_back(current.back());
        }
        current.swap(next);
    }
    radius = current.empty() ? 0.0F : std::fabs(current[0]);
}

template <MetricType metric>
void
TurboQuantizer<metric>::polar_to_cartesian(float radius,
                                           const std::vector<float>& angles,
                                           std::vector<float>& output) const {
    if (this->dim_ == 0) {
        output.clear();
        return;
    }
    std::vector<std::vector<float>> levels;
    std::vector<float> current{radius};
    levels.push_back(current);
    auto angle_it = angles.rbegin();

    while (current.size() < static_cast<uint64_t>(this->dim_)) {
        std::vector<float> next;
        next.reserve(std::min<uint64_t>(this->dim_, current.size() * 2));
        for (float value : current) {
            if (next.size() >= static_cast<uint64_t>(this->dim_)) {
                break;
            }
            if (angle_it == angles.rend()) {
                next.push_back(value);
                continue;
            }
            auto angle = *angle_it;
            ++angle_it;
            next.push_back(value * std::cos(angle));
            if (next.size() < static_cast<uint64_t>(this->dim_)) {
                next.push_back(value * std::sin(angle));
            }
        }
        current.swap(next);
        levels.push_back(current);
    }

    output.assign(current.begin(), current.end());
    output.resize(this->dim_, 0.0F);
}

template <MetricType metric>
void
TurboQuantizer<metric>::encode_polar_coordinates(const std::vector<float>& angles,
                                                 uint8_t* code) const {
    const auto code_size = get_polar_code_size();
    if (code_size == 0) {
        return;
    }
    std::memset(code, 0, code_size);
    const uint64_t levels = 1ULL << bits_per_dim_;
    uint64_t bit_offset = 0;
    for (uint64_t i = 0; i < angles.size(); ++i) {
        const bool full_circle = (i + 1 == angles.size());
        const float range = full_circle ? TWO_PI_VALUE : PI_VALUE;
        float normalized =
            full_circle ? positive_mod(angles[i], range) : angles[i] + (0.5F * PI_VALUE);
        normalized = clamp_value(normalized, 0.0F, std::nextafter(range, 0.0F));
        auto bucket = static_cast<uint64_t>(normalized / range * static_cast<float>(levels));
        bucket = std::min<uint64_t>(bucket, levels - 1);
        for (uint64_t b = 0; b < bits_per_dim_; ++b) {
            if (((bucket >> b) & 1ULL) != 0) {
                code[(bit_offset + b) / 8] |= static_cast<uint8_t>(1U << ((bit_offset + b) % 8));
            }
        }
        bit_offset += bits_per_dim_;
    }
}

template <MetricType metric>
void
TurboQuantizer<metric>::decode_polar_coordinates(const uint8_t* code, std::vector<float>& angles) const {
    angles.assign(this->dim_ > 0 ? this->dim_ - 1 : 0, 0.0F);
    const uint64_t levels = 1ULL << bits_per_dim_;
    uint64_t bit_offset = 0;
    for (uint64_t i = 0; i < angles.size(); ++i) {
        uint64_t bucket = 0;
        for (uint64_t b = 0; b < bits_per_dim_; ++b) {
            const auto byte = code[(bit_offset + b) / 8];
            const auto bit = (byte >> ((bit_offset + b) % 8)) & 1U;
            bucket |= static_cast<uint64_t>(bit) << b;
        }
        const bool full_circle = (i + 1 == angles.size());
        const float range = full_circle ? TWO_PI_VALUE : PI_VALUE;
        auto center = (static_cast<float>(bucket) + 0.5F) / static_cast<float>(levels);
        auto angle = center * range;
        angles[i] = full_circle ? angle : (angle - 0.5F * PI_VALUE);
        bit_offset += bits_per_dim_;
    }
}

template <MetricType metric>
void
TurboQuantizer<metric>::compute_qjl_projection(const float* data, std::vector<float>& projection) const {
    projection.assign(qjl_projection_dim_, 0.0F);
    if (not enable_qjl_) {
        return;
    }
    for (uint64_t row = 0; row < qjl_projection_dim_; ++row) {
        float sum = 0.0F;
        for (uint64_t col = 0; col < this->dim_; ++col) {
            const uint64_t parity = ((row + 1) * 1315423911ULL + (col + 1) * 2654435761ULL) & 1ULL;
            sum += parity == 0 ? data[col] : -data[col];
        }
        projection[row] = sum;
    }
}

template <MetricType metric>
void
TurboQuantizer<metric>::encode_qjl(const std::vector<float>& projection, uint8_t* code) const {
    const auto code_size = get_qjl_code_size();
    if (code_size == 0) {
        return;
    }
    std::memset(code, 0, code_size);
    for (uint64_t i = 0; i < projection.size(); ++i) {
        if (projection[i] >= 0.0F) {
            code[i / 8] |= static_cast<uint8_t>(1U << (i % 8));
        }
    }
}

template <MetricType metric>
void
TurboQuantizer<metric>::decode_qjl(const uint8_t* code, std::vector<float>& projection) const {
    projection.assign(qjl_projection_dim_, 0.0F);
    for (uint64_t i = 0; i < qjl_projection_dim_; ++i) {
        const auto bit = (code[i / 8] >> (i % 8)) & 1U;
        projection[i] = bit != 0 ? 1.0F : -1.0F;
    }
}

template <MetricType metric>
float
TurboQuantizer<metric>::compute_qjl_correction(const std::vector<float>& base_projection,
                                               const std::vector<float>& query_projection) const {
    if (not enable_qjl_ or qjl_projection_dim_ == 0) {
        return 0.0F;
    }
    float sum = 0.0F;
    for (uint64_t i = 0; i < qjl_projection_dim_; ++i) {
        sum += base_projection[i] * query_projection[i];
    }
    return sum / static_cast<float>(qjl_projection_dim_);
}

template <MetricType metric>
float
TurboQuantizer<metric>::load_float(const uint8_t* codes, uint64_t offset) const {
    float value = 0.0F;
    std::memcpy(&value, codes + offset, sizeof(float));
    return value;
}

template <MetricType metric>
void
TurboQuantizer<metric>::store_float(uint8_t* codes, uint64_t offset, float value) const {
    std::memcpy(codes + offset, &value, sizeof(float));
}

template <MetricType metric>
bool
TurboQuantizer<metric>::EncodeOneImpl(const DataType* data, uint8_t* codes) const {
    std::vector<float> rotated;
    rotate(data, rotated);

    float radius = 0.0F;
    std::vector<float> angles;
    cartesian_to_polar(rotated.data(), radius, angles);
    encode_polar_coordinates(angles, codes + offset_polar_code_);
    store_float(codes, offset_radius_, radius);

    std::vector<float> reconstructed_rotated;
    polar_to_cartesian(radius, angles, reconstructed_rotated);

    std::vector<float> base_projection;
    if (enable_qjl_) {
        std::vector<float> residual(this->dim_, 0.0F);
        for (uint64_t i = 0; i < this->dim_; ++i) {
            residual[i] = rotated[i] - reconstructed_rotated[i];
        }
        compute_qjl_projection(residual.data(), base_projection);
        encode_qjl(base_projection, codes + offset_qjl_code_);
    }

    float raw_norm = 0.0F;
    raw_norm = std::sqrt(std::max(InnerProduct(data, data, &this->dim_), 0.0F));
    store_float(codes, offset_raw_norm_, raw_norm);
    return true;
}

template <MetricType metric>
bool
TurboQuantizer<metric>::DecodeOneImpl(const uint8_t* codes, DataType* data) {
    std::vector<float> angles;
    decode_polar_coordinates(codes + offset_polar_code_, angles);
    const auto radius = load_float(codes, offset_radius_);

    std::vector<float> rotated;
    polar_to_cartesian(radius, angles, rotated);
    std::vector<float> decoded;
    inverse_rotate(rotated.data(), decoded);
    std::copy(decoded.begin(), decoded.end(), data);
    return true;
}

template <MetricType metric>
float
TurboQuantizer<metric>::get_metric_distance(const float* lhs,
                                            const float* rhs,
                                            float lhs_raw_norm,
                                            float rhs_raw_norm) const {
    if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
        return FP32ComputeL2Sqr(lhs, rhs, this->dim_);
    }

    auto similarity = FP32ComputeIP(lhs, rhs, this->dim_);
    if constexpr (metric == MetricType::METRIC_TYPE_COSINE) {
        const auto denominator = lhs_raw_norm * rhs_raw_norm;
        if (denominator > ZERO_THRESHOLD) {
            similarity /= denominator;
        } else {
            similarity = 0.0F;
        }
    }
    return 1.0F - similarity;
}

template <MetricType metric>
float
TurboQuantizer<metric>::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const {
    std::vector<float> lhs(this->dim_, 0.0F);
    std::vector<float> rhs(this->dim_, 0.0F);
    const_cast<TurboQuantizer<metric>*>(this)->DecodeOneImpl(codes1, lhs.data());
    const_cast<TurboQuantizer<metric>*>(this)->DecodeOneImpl(codes2, rhs.data());
    return get_metric_distance(lhs.data(),
                               rhs.data(),
                               load_float(codes1, offset_raw_norm_),
                               load_float(codes2, offset_raw_norm_));
}

template <MetricType metric>
void
TurboQuantizer<metric>::ProcessQueryImpl(const DataType* query,
                                         Computer<TurboQuantizer<metric>>& computer) const {
    if (computer.buf_ == nullptr) {
        computer.buf_ = reinterpret_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
    }
    auto* decoded_query = reinterpret_cast<float*>(computer.buf_);
    std::memcpy(decoded_query, query, this->dim_ * sizeof(float));

    auto projection_offset = this->dim_ * sizeof(float);
    auto raw_norm_offset = projection_offset;
    if (enable_qjl_) {
        std::vector<float> rotated;
        rotate(query, rotated);
        auto* query_projection = reinterpret_cast<float*>(computer.buf_ + projection_offset);
        std::vector<float> projection;
        compute_qjl_projection(rotated.data(), projection);
        std::memcpy(query_projection, projection.data(), qjl_projection_dim_ * sizeof(float));
        raw_norm_offset += qjl_projection_dim_ * sizeof(float);
    }
    const auto raw_norm = std::sqrt(std::max(InnerProduct(query, query, &this->dim_), 0.0F));
    std::memcpy(computer.buf_ + raw_norm_offset, &raw_norm, sizeof(float));
}

template <MetricType metric>
void
TurboQuantizer<metric>::ComputeDistImpl(Computer<TurboQuantizer<metric>>& computer,
                                        const uint8_t* codes,
                                        float* dists) const {
    const auto* decoded_query = reinterpret_cast<const float*>(computer.buf_);
    auto projection_offset = this->dim_ * sizeof(float);
    auto raw_norm_offset = projection_offset;
    const float* query_projection = nullptr;
    if (enable_qjl_) {
        query_projection = reinterpret_cast<const float*>(computer.buf_ + projection_offset);
        raw_norm_offset += qjl_projection_dim_ * sizeof(float);
    }
    float query_raw_norm = 0.0F;
    std::memcpy(&query_raw_norm, computer.buf_ + raw_norm_offset, sizeof(float));

    std::vector<float> decoded(this->dim_, 0.0F);
    const_cast<TurboQuantizer<metric>*>(this)->DecodeOneImpl(codes, decoded.data());

    auto dist = get_metric_distance(decoded.data(),
                                    decoded_query,
                                    load_float(codes, offset_raw_norm_),
                                    query_raw_norm);

    if (enable_qjl_) {
        std::vector<float> base_projection;
        decode_qjl(codes + offset_qjl_code_, base_projection);
        std::vector<float> query_projection_vec(query_projection, query_projection + qjl_projection_dim_);
        if constexpr (metric == MetricType::METRIC_TYPE_L2SQR) {
            dist = std::max(0.0F, dist - compute_qjl_correction(base_projection, query_projection_vec));
        } else {
            dist -= compute_qjl_correction(base_projection, query_projection_vec);
        }
    }
    *dists = dist;
}

template <MetricType metric>
void
TurboQuantizer<metric>::SerializeImpl(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, bits_per_dim_);
    StreamWriter::WriteObj(writer, use_fht_);
    StreamWriter::WriteObj(writer, enable_qjl_);
    StreamWriter::WriteObj(writer, qjl_projection_dim_);
    StreamWriter::WriteObj(writer, offset_polar_code_);
    StreamWriter::WriteObj(writer, offset_radius_);
    StreamWriter::WriteObj(writer, offset_qjl_code_);
    StreamWriter::WriteObj(writer, offset_raw_norm_);
    rotator_->Serialize(writer);
}

template <MetricType metric>
void
TurboQuantizer<metric>::DeserializeImpl(StreamReader& reader) {
    StreamReader::ReadObj(reader, bits_per_dim_);
    StreamReader::ReadObj(reader, use_fht_);
    StreamReader::ReadObj(reader, enable_qjl_);
    StreamReader::ReadObj(reader, qjl_projection_dim_);
    StreamReader::ReadObj(reader, offset_polar_code_);
    StreamReader::ReadObj(reader, offset_radius_);
    StreamReader::ReadObj(reader, offset_qjl_code_);
    StreamReader::ReadObj(reader, offset_raw_norm_);
    if (use_fht_) {
        rotator_ = std::make_shared<FhtKacRotator>(this->allocator_, this->dim_);
    } else {
        rotator_ = std::make_shared<RandomOrthogonalMatrix>(this->allocator_, this->dim_);
    }
    rotator_->Deserialize(reader);
}

TEMPLATE_QUANTIZER(TurboQuantizer)

}  // namespace vsag
