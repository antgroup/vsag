#include "sq4_local_shared_view.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "inner_string_params.h"
#include "quantization/scalar_quantization/sq8_local_quantizer.h"
#include "simd/sq4_local_shared_simd.h"
#include "simd/sq4_local_simd.h"
#include "vsag_exception.h"

namespace vsag {
namespace {
constexpr uint32_t kSharedViewMagic = 0x3451534c;

bool
CanPackCoarse(const uint8_t* code) {
    float minimum;
    float step;
    memcpy(&minimum, code, sizeof(float));
    memcpy(&step, code + sizeof(float), sizeof(float));
    return std::abs(minimum) <= 1e30F && step <= 1e30F / 17.0F;
}

const uint8_t*
PackCoarse(const uint8_t* fine, uint64_t dim, std::vector<uint8_t>& packed) {
    const uint64_t code_size = 2 * sizeof(float) + (dim + 1) / 2;
    if (packed.size() < code_size) {
        packed.resize(code_size);
    }
    memcpy(packed.data(), fine, sizeof(float));
    float step;
    memcpy(&step, fine + sizeof(float), sizeof(float));
    step *= 17.0F;
    memcpy(packed.data() + sizeof(float), &step, sizeof(float));
    fine += 2 * sizeof(float);
    auto* coarse = packed.data() + 2 * sizeof(float);
    uint64_t coordinate = 0;
    for (; coordinate + 1 < dim; coordinate += 2) {
        coarse[coordinate / 2] = static_cast<uint8_t>(((fine[coordinate] + 8) / 17) |
                                                      (((fine[coordinate + 1] + 8) / 17) << 4));
    }
    if (coordinate < dim) {
        coarse[coordinate / 2] = static_cast<uint8_t>((fine[coordinate] + 8) / 17);
    }
    return packed.data();
}

float
DecodeCoarse(uint8_t fine_code, float fine_step, float minimum) {
    const auto coarse_code = static_cast<uint8_t>((fine_code + 8) / 17);
    const double coarse_step = static_cast<double>(fine_step) * 17.0;
    const double value = static_cast<double>(coarse_code) * coarse_step + minimum;
    const double limit = std::numeric_limits<float>::max();
    if (value > limit) {
        return std::numeric_limits<float>::max();
    }
    if (value < -limit) {
        return -std::numeric_limits<float>::max();
    }
    if (coarse_step > limit) {
        return static_cast<float>(value);
    }
    return std::fma(static_cast<float>(coarse_code), static_cast<float>(coarse_step), minimum);
}

float
Distance(const float* query,
         const uint8_t* first,
         const uint8_t* second,
         uint64_t dim,
         MetricType metric) {
    if (CanPackCoarse(second) && (query != nullptr || CanPackCoarse(first))) {
        const auto shared_compute = GetSQ4LocalSharedCompute();
        const bool l2 = metric == MetricType::METRIC_TYPE_L2SQR;
        if (shared_compute != nullptr) {
            const float result = shared_compute(query, first, second, dim, l2);
            uint32_t result_bits;
            memcpy(&result_bits, &result, sizeof(result_bits));
            if ((result_bits & 0x7f800000U) != 0x7f800000U) {
                return result;
            }
        } else {
            thread_local std::vector<uint8_t> first_scratch;
            thread_local std::vector<uint8_t> second_scratch;
            const auto* packed_second = PackCoarse(second, dim, second_scratch);
            if (query != nullptr) {
                return l2 ? SQ4LocalComputeL2Sqr(query, packed_second, dim)
                          : 1.0F - SQ4LocalComputeIP(query, packed_second, dim);
            }
            const auto* packed_first = PackCoarse(first, dim, first_scratch);
            return l2 ? SQ4LocalComputeCodesL2Sqr(packed_first, packed_second, dim)
                      : 1.0F - SQ4LocalComputeCodesIP(packed_first, packed_second, dim);
        }
    }
    float minimum_first = 0.0F;
    float step_first = 0.0F;
    if (query == nullptr) {
        memcpy(&minimum_first, first, sizeof(float));
        memcpy(&step_first, first + sizeof(float), sizeof(float));
        first += 2 * sizeof(float);
    }
    float minimum_second;
    float step_second;
    memcpy(&minimum_second, second, sizeof(float));
    memcpy(&step_second, second + sizeof(float), sizeof(float));
    second += 2 * sizeof(float);
    float first_levels[16];
    float second_levels[16];
    for (uint8_t level = 0; level < 16; ++level) {
        if (query == nullptr) {
            first_levels[level] = DecodeCoarse(level * 17, step_first, minimum_first);
        }
        second_levels[level] = DecodeCoarse(level * 17, step_second, minimum_second);
    }
    double total = 0.0;
    const bool l2 = metric == MetricType::METRIC_TYPE_L2SQR;
    for (uint64_t coordinate = 0; coordinate < dim; ++coordinate) {
        const float first_value =
            query == nullptr ? first_levels[(first[coordinate] + 8) / 17] : query[coordinate];
        const float second_value = second_levels[(second[coordinate] + 8) / 17];
        const double delta = static_cast<double>(first_value) - second_value;
        total += l2 ? delta * delta : static_cast<double>(first_value) * second_value;
    }
    if (l2 && total > std::numeric_limits<float>::max()) {
        return std::numeric_limits<float>::infinity();
    }
    const double result = l2 ? total : 1.0 - total;
    if (result > std::numeric_limits<float>::max()) {
        return std::numeric_limits<float>::infinity();
    }
    if (result < -std::numeric_limits<float>::max()) {
        return -std::numeric_limits<float>::infinity();
    }
    return static_cast<float>(result);
}
}  // namespace

SQ4LocalSharedView::SQ4LocalSharedView(FlattenInterfacePtr precise,
                                       const IndexCommonParam& common_param)
    : precise_(std::move(precise)), common_param_(common_param) {
    CHECK_ARGUMENT(
        precise_ != nullptr && precise_->GetQuantizerName() == QUANTIZATION_TYPE_VALUE_SQ8_LOCAL,
        "shared SQ4 local requires SQ8 local precise codes");
    this->code_size_ = precise_->code_size_;
    this->max_capacity_ = precise_->max_capacity_;
}

void
SQ4LocalSharedView::Query(float* result_dists,
                          const ComputerInterfacePtr& computer,
                          const InnerIdType* idx,
                          InnerIdType id_count,
                          QueryContext*) {
    const auto* sq8_computer = static_cast<const Computer<SQ8LocalQuantizer>*>(computer.get());
    const auto* query = reinterpret_cast<const float*>(sq8_computer->buf_);
    const auto shared_compute = GetSQ4LocalSharedCompute();
    const auto shared_compute_batch4 = GetSQ4LocalSharedComputeBatch4();
    const bool l2 = common_param_.metric_ == MetricType::METRIC_TYPE_L2SQR;
    const auto query_distance = [&](const uint8_t* code) {
        if (shared_compute != nullptr && CanPackCoarse(code)) {
            const float result = shared_compute(query, nullptr, code, common_param_.dim_, l2);
            if (std::isfinite(result)) {
                return result;
            }
        }
        return Distance(query, nullptr, code, common_param_.dim_, common_param_.metric_);
    };
    constexpr InnerIdType batch_size = 4;
    for (InnerIdType offset = 0; offset < std::min(id_count, batch_size); ++offset) {
        precise_->Prefetch(idx[offset]);
    }
    InnerIdType offset = 0;
    for (; offset + batch_size <= id_count; offset += batch_size) {
        for (InnerIdType ahead = offset + batch_size;
             ahead < std::min(id_count, offset + 2 * batch_size);
             ++ahead) {
            precise_->Prefetch(idx[ahead]);
        }
        auto first = precise_->AcquireCodesById(idx[offset]);
        auto second = precise_->AcquireCodesById(idx[offset + 1]);
        auto third = precise_->AcquireCodesById(idx[offset + 2]);
        auto fourth = precise_->AcquireCodesById(idx[offset + 3]);
        const uint8_t* codes[] = {first.Data(), second.Data(), third.Data(), fourth.Data()};
        if (shared_compute_batch4 != nullptr && CanPackCoarse(codes[0]) &&
            CanPackCoarse(codes[1]) && CanPackCoarse(codes[2]) && CanPackCoarse(codes[3])) {
            shared_compute_batch4(query, codes, common_param_.dim_, l2, result_dists + offset);
            for (InnerIdType candidate = 0; candidate < batch_size; ++candidate) {
                if (!std::isfinite(result_dists[offset + candidate])) {
                    result_dists[offset + candidate] = query_distance(codes[candidate]);
                }
            }
        } else {
            for (InnerIdType candidate = 0; candidate < batch_size; ++candidate) {
                result_dists[offset + candidate] = query_distance(codes[candidate]);
            }
        }
    }
    for (; offset < id_count; ++offset) {
        auto code = precise_->AcquireCodesById(idx[offset]);
        result_dists[offset] = query_distance(code.Data());
    }
}

ComputerInterfacePtr
SQ4LocalSharedView::FactoryComputer(const void* query) {
    return precise_->FactoryComputer(query);
}

void
SQ4LocalSharedView::Train(const void*, uint64_t) {
}

void
SQ4LocalSharedView::InsertVector(const void*, InnerIdType) {
}

bool
SQ4LocalSharedView::UpdateVector(const void*, InnerIdType idx) {
    return idx < precise_->TotalCount();
}

void
SQ4LocalSharedView::BatchInsertVector(const void*, InnerIdType, InnerIdType*) {
}

float
SQ4LocalSharedView::ComputePairVectors(InnerIdType id1, InnerIdType id2) {
    auto first = precise_->AcquireCodesById(id1);
    auto second = precise_->AcquireCodesById(id2);
    return Distance(
        nullptr, first.Data(), second.Data(), common_param_.dim_, common_param_.metric_);
}

void
SQ4LocalSharedView::Prefetch(InnerIdType id) {
    precise_->Prefetch(id);
}

std::string
SQ4LocalSharedView::GetQuantizerName() {
    return QUANTIZATION_TYPE_VALUE_SQ4_LOCAL_SHARED;
}

MetricType
SQ4LocalSharedView::GetMetricType() {
    return precise_->GetMetricType();
}

void
SQ4LocalSharedView::Resize(InnerIdType capacity) {
    this->max_capacity_ = std::max(this->max_capacity_, capacity);
}

void
SQ4LocalSharedView::ExportModel(const FlattenInterfacePtr& other) const {
    CHECK_ARGUMENT(std::dynamic_pointer_cast<SQ4LocalSharedView>(other) != nullptr,
                   "shared SQ4 local model requires matching view");
}

bool
SQ4LocalSharedView::Decode(const uint8_t* codes, float* vector) {
    return precise_->Decode(codes, vector);
}

bool
SQ4LocalSharedView::Encode(const float* vector, uint8_t* codes) {
    return precise_->Encode(vector, codes);
}

const uint8_t*
SQ4LocalSharedView::GetCodesById(InnerIdType id, bool& need_release) const {
    return precise_->GetCodesById(id, need_release);
}

void
SQ4LocalSharedView::Release(const uint8_t* data) const {
    precise_->Release(data);
}

bool
SQ4LocalSharedView::GetCodesById(InnerIdType id, uint8_t* codes) const {
    return precise_->GetCodesById(id, codes);
}

InnerIdType
SQ4LocalSharedView::TotalCount() const {
    return precise_->TotalCount();
}

void
SQ4LocalSharedView::Serialize(StreamWriter& writer) {
    StreamWriter::WriteObj(writer, kSharedViewMagic);
    StreamWriter::WriteObj(writer, this->code_size_);
}

void
SQ4LocalSharedView::Deserialize(LvalueOrRvalue<StreamReader> reader) {
    uint32_t magic;
    uint32_t code_size;
    StreamReader::ReadObj(reader, magic);
    StreamReader::ReadObj(reader, code_size);
    CHECK_ARGUMENT(magic == kSharedViewMagic && code_size == precise_->code_size_,
                   "invalid shared SQ4 local view metadata");
}

bool
SQ4LocalSharedView::InMemory() const {
    return precise_->InMemory();
}

void
SQ4LocalSharedView::MergeOther(const FlattenInterfacePtr& other, InnerIdType) {
    CHECK_ARGUMENT(std::dynamic_pointer_cast<SQ4LocalSharedView>(other) != nullptr,
                   "shared SQ4 local merge requires matching view");
}

void
SQ4LocalSharedView::Move(InnerIdType, InnerIdType) {
}

void
SQ4LocalSharedView::ShrinkToFit(InnerIdType capacity) {
    this->max_capacity_ = capacity;
}

IndexCommonParam
SQ4LocalSharedView::ExportCommonParam() {
    return common_param_;
}
}  // namespace vsag
