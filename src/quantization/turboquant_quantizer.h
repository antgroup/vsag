
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

#include <memory>
#include <string>
#include <vector>

#include "impl/transform/transformer_headers.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "turboquant_quantizer_parameter.h"

namespace vsag {

template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class TurboQuantizer : public Quantizer<TurboQuantizer<metric>> {
public:
    explicit TurboQuantizer(int dim,
                            uint64_t bits_per_dim,
                            bool use_fht,
                            bool enable_qjl,
                            uint64_t qjl_projection_dim,
                            Allocator* allocator);

    explicit TurboQuantizer(const TurboQuantizerParamPtr& param,
                            const IndexCommonParam& common_param);

    explicit TurboQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    bool
    TrainImpl(const DataType* data, uint64_t count);

    bool
    EncodeOneImpl(const DataType* data, uint8_t* codes) const;

    bool
    DecodeOneImpl(const uint8_t* codes, DataType* data) const;

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    void
    ProcessQueryImpl(const DataType* query, Computer<TurboQuantizer<metric>>& computer) const;

    void
    ComputeDistImpl(Computer<TurboQuantizer<metric>>& computer,
                    const uint8_t* codes,
                    float* dists) const;

    void
    SerializeImpl(StreamWriter& writer);

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_TURBOQUANT;
    }

private:
    [[nodiscard]] uint64_t
    get_polar_code_size() const;

    [[nodiscard]] uint64_t
    get_qjl_code_size() const;

    [[nodiscard]] uint64_t
    get_query_workspace_size() const;

    void
    update_derived_sizes();

    void
    validate_configuration() const;

    [[nodiscard]] float
    get_metric_distance(const float* lhs,
                        const float* rhs,
                        float lhs_raw_norm,
                        float rhs_raw_norm) const;

    void
    rotate(const DataType* input, float* output) const;

    void
    inverse_rotate(const float* input, float* output) const;

    void
    cartesian_to_polar(const float* input, float& radius, std::vector<float>& angles) const;

    void
    polar_to_cartesian(float radius,
                       const std::vector<float>& angles,
                       std::vector<float>& output) const;

    void
    encode_polar_coordinates(const std::vector<float>& angles, uint8_t* code) const;

    void
    decode_polar_coordinates(const uint8_t* code, std::vector<float>& angles) const;

    void
    decode_to_rotated(const uint8_t* codes,
                      float* rotated,
                      float* angle_buffer,
                      float* current_buffer,
                      float* next_buffer) const;

    void
    compute_qjl_projection(const float* data, std::vector<float>& projection) const;

    void
    encode_qjl(const std::vector<float>& projection, uint8_t* code) const;

    [[nodiscard]] float
    compute_qjl_correction(const uint8_t* base_code, const float* query_projection) const;

    [[nodiscard]] float
    load_float(const uint8_t* codes, uint64_t offset) const;

    void
    store_float(uint8_t* codes, uint64_t offset, float value) const;

    [[nodiscard]] uint64_t
    bits_to_bytes(uint64_t bits) const;

    uint64_t bits_per_dim_{4};
    bool use_fht_{true};
    bool enable_qjl_{true};
    uint64_t qjl_projection_dim_{0};

    uint64_t offset_polar_code_{0};
    uint64_t offset_radius_{0};
    uint64_t offset_qjl_code_{0};
    uint64_t offset_raw_norm_{0};

    std::shared_ptr<VectorTransformer> rotator_;
};

}  // namespace vsag
