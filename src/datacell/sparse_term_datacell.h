
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

#include "algorithm/sindi/sindi_parameter.h"
#include "impl/searcher/basic_searcher.h"
#include "quantization/sparse_quantization//sparse_term_computer.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "utils/pointer_define.h"
#include "vsag/dataset.h"

namespace vsag {

struct QuantizationParams {
    std::string type = "fp32";
    float min_val = 0.0f;
    float max_val = 0.0f;
    float diff = 1.0f;
};

DEFINE_POINTER(SparseTermDataCell);
class SparseTermDataCell {
public:
    SparseTermDataCell() = default;

    SparseTermDataCell(float doc_retain_ratio,
                       uint32_t term_id_limit,
                       Allocator* allocator,
                       std::shared_ptr<QuantizationParams> quantization_params)
        : doc_retain_ratio_(doc_retain_ratio),
          term_id_limit_(term_id_limit),
          allocator_(allocator),
          term_ids_(allocator),
          term_datas_(allocator),
          term_sizes_(allocator),
          quantization_params_(std::move(quantization_params)) {
    }

    void
    Query(float* global_dists, const SparseTermComputerPtr& computer) const;

    template <InnerSearchMode mode = InnerSearchMode::KNN_SEARCH,
              InnerSearchType type = InnerSearchType::PURE>
    void
    InsertHeap(float* dists,
               const SparseTermComputerPtr& computer,
               MaxHeap& heap,
               const InnerSearchParam& param,
               uint32_t offset_id) const;

    void
    DocPrune(Vector<std::pair<uint32_t, float>>& sorted_base) const;

    void
    InsertVector(const SparseVector& sparse_base, uint16_t base_id);

    void
    ResizeTermList(InnerIdType new_term_capacity);

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    float
    CalcDistanceByInnerId(const SparseTermComputerPtr& computer, uint16_t base_id);

    void
    GetSparseVector(uint16_t base_id, SparseVector* data);

private:
    void
    Encode(float val, uint8_t* dst) const;

    void
    Encodesq8(float val, uint8_t* dst) const;

    static void
    Encodefp16(float val, uint8_t* dst);

    void
    Decode(const uint8_t* src, size_t size, float* dst) const;

    void
    Decodesq8(const uint8_t* src, size_t size, float* dst) const;

    static void
    Decodefp16(const uint8_t* src, size_t size, float* dst);

public:
    uint32_t term_id_limit_{0};

    float doc_retain_ratio_{0};

    uint32_t term_capacity_{0};

    Vector<std::unique_ptr<Vector<uint16_t>>> term_ids_;

    Vector<std::unique_ptr<Vector<uint8_t>>> term_datas_;

    Vector<uint32_t> term_sizes_;

    Allocator* const allocator_{nullptr};

    std::shared_ptr<QuantizationParams> quantization_params_;
};
}  // namespace vsag
