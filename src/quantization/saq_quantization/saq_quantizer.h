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
#include <vector>

#include "impl/transform/random_orthogonal_transformer.h"
#include "index_common_param.h"
#include "inner_string_params.h"
#include "quantization/quantizer.h"
#include "saq_quantizer_parameter.h"

namespace vsag {

/**
 * Segmented Code Adjustment Quantization (SAQ).
 *
 * Training uses a full-dimensional PCA rotation, then jointly chooses segment boundaries and
 * per-segment bit widths under a fixed average-bit budget. Each segment is scalar-quantized and
 * refined with the CAQ coordinate-adjustment objective. The stored record contains the adjusted
 * scale and a metric-specific squared norm for each segment, followed by its packed scalar codes.
 * L2 stores the original projected norm used by its asymmetric distance formula; cosine stores the
 * reconstructed norm so query-to-code and code-to-code distances have identical semantics.
 */
template <MetricType metric = MetricType::METRIC_TYPE_L2SQR>
class SAQQuantizer : public Quantizer<SAQQuantizer<metric>> {
public:
    struct Segment {
        uint64_t begin{0};
        uint64_t length{0};
        uint64_t bits{0};
        uint64_t metadata_offset{0};
        uint64_t code_offset{0};
        uint64_t code_bytes{0};
    };

    explicit SAQQuantizer(int dim,
                          float avg_bits,
                          uint64_t segment_count,
                          uint64_t adjustment_rounds,
                          bool use_pca,
                          bool random_rotation,
                          Allocator* allocator);

    SAQQuantizer(const SAQQuantizerParamPtr& param, const IndexCommonParam& common_param);

    SAQQuantizer(const QuantizerParamPtr& param, const IndexCommonParam& common_param);

    ~SAQQuantizer() = default;

    bool
    TrainImpl(const float* data, uint64_t count);

    bool
    EncodeOneImpl(const float* data, uint8_t* codes) const;

    bool
    DecodeOneImpl(const uint8_t* codes, float* data) const;

    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) const;

    void
    ProcessQueryImpl(const float* query, Computer<SAQQuantizer>& computer) const;

    void
    ComputeDistImpl(Computer<SAQQuantizer>& computer, const uint8_t* codes, float* dists) const;

    void
    ComputeDistsBatch4Impl(Computer<SAQQuantizer>& computer,
                           const uint8_t* codes1,
                           const uint8_t* codes2,
                           const uint8_t* codes3,
                           const uint8_t* codes4,
                           float& dist1,
                           float& dist2,
                           float& dist3,
                           float& dist4) const;

    bool
    ComputeDistWithThresholdImpl(Computer<SAQQuantizer>& computer,
                                 const uint8_t* codes,
                                 float threshold,
                                 float* dists) const;

    void
    SerializeImpl(StreamWriter& writer) const;

    void
    DeserializeImpl(StreamReader& reader);

    [[nodiscard]] std::string
    NameImpl() const {
        return QUANTIZATION_TYPE_VALUE_SAQ;
    }

    [[nodiscard]] const std::vector<Segment>&
    GetSegments() const {
        return segments_;
    }

private:
    static constexpr uint64_t MAX_BITS = 13;
    static constexpr uint64_t SEGMENT_ALIGNMENT = 64;
    static constexpr uint64_t METADATA_FLOATS = 2;
    static constexpr uint64_t METADATA_BYTES = METADATA_FLOATS * sizeof(float);
    static constexpr uint64_t METADATA_BITS = METADATA_BYTES * 8;
    static constexpr uint64_t RECORD_OVERHEAD_BITS = 3 * sizeof(float) * 8;
    static constexpr uint64_t MAX_TRAINING_SAMPLES = 65536;

    void
    ProjectGlobal(const float* data, float* projected) const;

    void
    Project(const float* data, float* projected) const;

    void
    InverseProject(const float* projected, float* data) const;

    void
    BuildPlan(const Vector<float>& variances);

    void
    BuildDynamicPlan(const Vector<float>& variances);

    void
    BuildFixedSegmentPlan(const Vector<float>& variances);

    void
    InitializeSegmentLayout();

    void
    InitializeRotations(bool train);

    void
    EncodeSegment(const float* data, const Segment& segment, uint8_t* codes) const;

    void
    DecodeSegment(const uint8_t* codes, const Segment& segment, float* data) const;

    [[nodiscard]] float
    ComputeProjectedDistance(const float* query_lookup,
                             const float* query_norms,
                             const float* query_sums,
                             const uint8_t* codes,
                             float threshold,
                             bool enable_threshold,
                             bool* stopped_early) const;

    [[nodiscard]] float
    ComputeSegmentInnerProduct(const float* query_lookup,
                               float query_sum,
                               const uint8_t* codes,
                               const Segment& segment) const;

    static void
    WritePackedCode(uint8_t* codes, uint64_t index, uint64_t length, uint64_t bits, uint16_t value);

    [[nodiscard]] static uint16_t
    ReadPackedCode(const uint8_t* codes, uint64_t index, uint64_t length, uint64_t bits);

private:
    float avg_bits_{SAQQuantizerParameter::DEFAULT_AVG_BITS};
    uint64_t requested_segment_count_{SAQQuantizerParameter::DEFAULT_SEGMENT_COUNT};
    uint64_t adjustment_rounds_{SAQQuantizerParameter::DEFAULT_ADJUSTMENT_ROUNDS};
    bool use_pca_{SAQQuantizerParameter::DEFAULT_USE_PCA};
    bool random_rotation_{SAQQuantizerParameter::DEFAULT_RANDOM_ROTATION};
    uint64_t budget_bits_{0};
    Vector<float> projection_matrix_;
    Vector<float> projected_mean_;
    std::vector<Segment> segments_;
    std::vector<std::shared_ptr<RandomOrthogonalMatrix>> rotations_;
};

}  // namespace vsag
