
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

#include "impl/reorder/pqr_reorder.h"

#include "simd/simd.h"
#include "storage/stream_writer.h"
namespace vsag {

static const int64_t TRAIN_MAX_SIZE = 65535;

DistHeapPtr
PqrReorder::Reorder(const DistHeapPtr& input,
                    const float* query,
                    int64_t topk,
                    Allocator* allocator) const {
    auto reorder_heap = DistanceHeap::MakeInstanceBySize<true, true>(allocator, topk);
    const auto* float_vector = reinterpret_cast<const float*>(query);
    Vector<float> normalize_vector(dim_, allocator_);
    if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        Normalize(float_vector, normalize_vector.data(), dim_);
        float_vector = normalize_vector.data();
    }
    auto computer = reorder_code_->FactoryComputer(float_vector);
    size_t candidate_size = input->Size();
    const auto* candidate_result = input->GetData();
    Vector<InnerIdType> ids(candidate_size, allocator);
    Vector<float> dists(candidate_size, allocator);
    for (int i = 0; i < candidate_size; ++i) {
        ids[i] = candidate_result[i].second;
    }
    reorder_code_->Query(dists.data(), computer, ids.data(), candidate_size);
    for (int i = 0; i < candidate_size; ++i) {
        float final_dist = candidate_result[i].first;
        dists[i] = 1 - dists[i];
        if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
            final_dist += bias_[ids[i]];
            dists[i] *= 2;
        }
        reorder_heap->Push(final_dist - dists[i], ids[i]);
    }
    return reorder_heap;
}

void
PqrReorder::InsertVector(const void* vector, vsag::InnerIdType id) {
    const auto* float_vector = reinterpret_cast<const float*>(vector);
    Vector<float> normalize_vector(dim_, allocator_);
    if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        Normalize(float_vector, normalize_vector.data(), dim_);
        float_vector = normalize_vector.data();
    }
    bool need_release = false;
    const auto* codes = flatten_->GetCodesById(id, need_release);
    Vector<float> code_vector(dim_, allocator_);
    Vector<float> residual_vector(dim_, allocator_);
    flatten_->Decode(codes, code_vector.data());
    FP32Sub(float_vector, code_vector.data(), residual_vector.data(), dim_);
    reorder_code_->InsertVector(residual_vector.data(), id);
    // get decoded_residual_vector
    Vector<float> decode_residual_vector(dim_, allocator_);
    bool residual_need_release = false;
    const auto* residual_codes = flatten_->GetCodesById(id, need_release);
    reorder_code_->Decode(residual_codes, decode_residual_vector.data());
    if (metric_ == MetricType::METRIC_TYPE_L2SQR) {
        bias_[id] =
            2 * FP32ComputeIP(decode_residual_vector.data(), code_vector.data(), dim_) +
            FP32ComputeIP(decode_residual_vector.data(), decode_residual_vector.data(), dim_);
    }
    if (residual_need_release) {
        allocator_->Deallocate((void*)residual_codes);
    }
    if (need_release) {
        allocator_->Deallocate((void*)codes);
    }
}

void
PqrReorder::Train(const void* vector, uint64_t count) {
    int64_t train_size = count > TRAIN_MAX_SIZE ? TRAIN_MAX_SIZE : static_cast<int64_t>(count);
    const auto* float_vector = reinterpret_cast<const float*>(vector);
    Vector<float> train_vectors(train_size * dim_, allocator_);
    Vector<float> residual_vectors(train_size * dim_, allocator_);
    if (metric_ == MetricType::METRIC_TYPE_COSINE) {
        for (int64_t i = 0; i < train_size; ++i) {
            Normalize(float_vector + i * dim_, train_vectors.data() + i * dim_, dim_);
        }
    } else {
        std::memcpy(train_vectors.data(), float_vector, sizeof(float) * train_size * dim_);
    }
    this->flatten_->CalResidual(train_vectors.data(), residual_vectors.data(), train_size);
    reorder_code_->Train(residual_vectors.data(), train_size);
}

void
PqrReorder::Resize(uint64_t new_size) {
    if (new_size <= this->size_) {
        return;
    }
    reorder_code_->Resize(new_size);
    bias_.resize(new_size);
}

void
PqrReorder::Serialize(StreamWriter& writer) const {
    reorder_code_->Serialize(writer);
    StreamWriter::WriteVector(writer, bias_);
}
void
PqrReorder::Deserialize(StreamReader& reader) {
    reorder_code_->Deserialize(reader);
    StreamReader::ReadVector(reader, bias_);
}

}  // namespace vsag
