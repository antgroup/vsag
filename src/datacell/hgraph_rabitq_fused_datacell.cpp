// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hgraph_rabitq_fused_datacell.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>

#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

namespace vsag {

namespace {
constexpr uint64_t K_CACHE_LINE_SIZE = 64;
constexpr uint64_t K_COUNT_OFFSET = 0;
constexpr uint64_t K_HEADER_SIZE = sizeof(uint32_t);
constexpr uint32_t K_SERIALIZATION_VERSION = 2;

struct FusedWireLayout {
    uint64_t record_size{0};
    uint64_t neighbors_offset{0};
    uint64_t cluster_id_offset{0};
    uint64_t label_offset{0};
    uint64_t one_bit_offset{0};
    uint64_t supplement_offset{0};
    uint64_t one_bit_code_size{0};
    uint64_t supplement_code_size{0};
};

uint64_t
fused_codec_model_size(int64_t dim) {
    CHECK_ARGUMENT(dim > 0, "invalid fused RaBitQ dimension");
    constexpr uint64_t fixed_size = sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t);
    constexpr uint64_t bytes_per_dimension =
        static_cast<uint64_t>(K_FUSED_CLUSTER_COUNT) * sizeof(float);
    const auto unsigned_dim = static_cast<uint64_t>(dim);
    CHECK_ARGUMENT(
        unsigned_dim <= (std::numeric_limits<uint64_t>::max() - fixed_size) / bytes_per_dimension,
        "fused RaBitQ codec size overflow");
    return fixed_size + unsigned_dim * bytes_per_dimension;
}

uint64_t
remaining_bytes(StreamReader& reader) {
    const auto length = reader.Length();
    const auto cursor = reader.GetCursor();
    CHECK_ARGUMENT(cursor <= length, "invalid fused graph reader cursor");
    return length - cursor;
}
}  // namespace

uint64_t
HGraphRaBitQFusedDataCell::AlignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

HGraphRaBitQFusedDataCell::HGraphRaBitQFusedDataCell(const GraphDataCellParamPtr& graph_param,
                                                     uint64_t one_bit_code_size,
                                                     uint64_t supplement_code_size,
                                                     const IndexCommonParam& common_param)
    : storage_(common_param.allocator_.get()),
      one_bit_code_size_(one_bit_code_size),
      supplement_code_size_(supplement_code_size),
      dim_(common_param.dim_) {
    CHECK_ARGUMENT(graph_param != nullptr, "fused graph parameter must not be null");
    CHECK_ARGUMENT(graph_param->max_degree_ <= std::numeric_limits<uint32_t>::max(),
                   "fused graph maximum degree exceeds uint32 range");
    CHECK_ARGUMENT(graph_param->init_max_capacity_ <= std::numeric_limits<InnerIdType>::max(),
                   "fused graph initial capacity exceeds id range");
    allocator_ = common_param.allocator_.get();
    maximum_degree_ = static_cast<uint32_t>(graph_param->max_degree_);
    max_capacity_ = static_cast<InnerIdType>(graph_param->init_max_capacity_);
    CHECK_ARGUMENT(not graph_param->support_remove_, "fused RaBitQ graph does not support removal");
    CHECK_ARGUMENT(not graph_param->use_reverse_edges_,
                   "fused RaBitQ graph does not support reverse edges");

    neighbors_offset_ = K_HEADER_SIZE;
    cluster_id_offset_ =
        neighbors_offset_ + static_cast<uint64_t>(maximum_degree_) * sizeof(InnerIdType);
    label_offset_ = AlignUp(cluster_id_offset_ + sizeof(uint32_t), alignof(LabelType));
    one_bit_offset_ = label_offset_ + sizeof(LabelType);
    supplement_offset_ = one_bit_offset_ + one_bit_code_size_;
    record_size_ = AlignUp(supplement_offset_ + supplement_code_size_, K_CACHE_LINE_SIZE);
    CHECK_ARGUMENT(
        max_capacity_ <=
            (std::numeric_limits<uint64_t>::max() - (K_CACHE_LINE_SIZE - 1)) / record_size_,
        "fused graph initial capacity and record stride overflow");

    if (graph_param->support_duplicate_) {
        InitDuplicateTracker();
    }
    Reallocate(max_capacity_);
}

void
HGraphRaBitQFusedDataCell::Reallocate(InnerIdType new_capacity) {
    Vector<uint8_t> replacement(
        static_cast<uint64_t>(new_capacity) * record_size_ + K_CACHE_LINE_SIZE - 1, 0, allocator_);
    const auto replacement_address = reinterpret_cast<uintptr_t>(replacement.data());
    const uint64_t replacement_offset =
        (K_CACHE_LINE_SIZE - replacement_address % K_CACHE_LINE_SIZE) % K_CACHE_LINE_SIZE;
    if (not storage_.empty()) {
        const auto copy_count = std::min(max_capacity_, new_capacity);
        std::memcpy(replacement.data() + replacement_offset,
                    storage_.data() + aligned_offset_,
                    static_cast<uint64_t>(copy_count) * record_size_);
    }
    storage_ = std::move(replacement);
    aligned_offset_ = replacement_offset;
    max_capacity_ = new_capacity;
    if (duplicate_tracker_ != nullptr) {
        duplicate_tracker_->Resize(new_capacity);
    }
}

uint8_t*
HGraphRaBitQFusedDataCell::MutableNodeRecord(InnerIdType id) {
    return storage_.data() + aligned_offset_ + static_cast<uint64_t>(id) * record_size_;
}

const InnerIdType*
HGraphRaBitQFusedDataCell::GetNeighborData(const uint8_t* record) const {
    return reinterpret_cast<const InnerIdType*>(record + neighbors_offset_);
}

const uint8_t*
HGraphRaBitQFusedDataCell::GetOneBitCode(const uint8_t* record) const {
    return record + one_bit_offset_;
}

const uint8_t*
HGraphRaBitQFusedDataCell::GetSupplementCode(const uint8_t* record) const {
    return record + supplement_offset_;
}

LabelType
HGraphRaBitQFusedDataCell::GetLabel(const uint8_t* record) const {
    LabelType label = 0;
    std::memcpy(&label, record + label_offset_, sizeof(label));
    return label;
}

uint32_t
HGraphRaBitQFusedDataCell::GetClusterId(const uint8_t* record) const {
    uint32_t cluster_id = 0;
    std::memcpy(&cluster_id, record + cluster_id_offset_, sizeof(cluster_id));
    return cluster_id;
}

void
HGraphRaBitQFusedDataCell::InsertNeighborsById(InnerIdType id,
                                               const Vector<InnerIdType>& neighbor_ids) {
    CHECK_ARGUMENT(neighbor_ids.size() <= maximum_degree_,
                   "fused node neighbor count exceeds maximum degree");
    if (id >= max_capacity_) {
        Resize(id + 1);
    }

    auto* record = MutableNodeRecord(id);
    const auto count = static_cast<uint32_t>(neighbor_ids.size());
    std::memcpy(record + K_COUNT_OFFSET, &count, sizeof(count));
    auto* output = reinterpret_cast<InnerIdType*>(record + neighbors_offset_);
    for (uint64_t i = 0; i < neighbor_ids.size(); ++i) {
        output[i] = neighbor_ids[i];
    }
    auto current = total_count_.load(std::memory_order_relaxed);
    while (current < id + 1 and
           not total_count_.compare_exchange_weak(
               current, id + 1, std::memory_order_release, std::memory_order_relaxed)) {
    }
}

void
HGraphRaBitQFusedDataCell::SetNodeCodes(InnerIdType id,
                                        LabelType label,
                                        uint32_t cluster_id,
                                        const uint8_t* one_bit_code,
                                        const uint8_t* supplement_code) {
    SetFusedCodes(id, cluster_id, one_bit_code, supplement_code);
    std::memcpy(MutableNodeRecord(id) + label_offset_, &label, sizeof(label));
}

void
HGraphRaBitQFusedDataCell::SetLabel(InnerIdType id, LabelType label) {
    CHECK_ARGUMENT(id < max_capacity_, "fused graph label id does not exist");
    std::memcpy(MutableNodeRecord(id) + label_offset_, &label, sizeof(label));
}

bool
HGraphRaBitQFusedDataCell::GetFusedCodeView(InnerIdType id, RaBitQFusedCodeView& view) const {
    // Logical duplicate aliases own encoded records but no graph edges, so their IDs may be
    // greater than the graph-node high-water mark. Callers validate logical IDs through the
    // label table; this layer only enforces the allocated slab boundary.
    if (id >= max_capacity_) {
        view = {};
        return false;
    }
    const auto* record = GetNodeRecord(id);
    view.one_bit_code = record + one_bit_offset_;
    view.supplement_code = record + supplement_offset_;
    std::memcpy(&view.cluster_id, record + cluster_id_offset_, sizeof(view.cluster_id));
    return true;
}

void
HGraphRaBitQFusedDataCell::SetFusedCodes(InnerIdType id,
                                         uint32_t cluster_id,
                                         const uint8_t* one_bit_code,
                                         const uint8_t* supplement_code) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        one_bit_code != nullptr and supplement_code != nullptr,
        "fused RaBitQ codes must not be null");
    if (id >= max_capacity_) {
        Resize(id + 1);
    }
    auto* record = MutableNodeRecord(id);
    std::memcpy(record + cluster_id_offset_, &cluster_id, sizeof(cluster_id));
    std::memcpy(record + one_bit_offset_, one_bit_code, one_bit_code_size_);
    std::memcpy(record + supplement_offset_, supplement_code, supplement_code_size_);
}

void
HGraphRaBitQFusedDataCell::PrefetchFusedCodes(InnerIdType id, bool include_supplement) const {
    const auto* record = GetNodeRecord(id);
    const auto prefetch_l1 = [](const uint8_t* begin, uint64_t size) {
        const auto begin_address = reinterpret_cast<uintptr_t>(begin);
        const auto begin_offset = begin_address % K_CACHE_LINE_SIZE;
        const auto* first = begin - begin_offset;
        const auto span = begin_offset + size;
        for (uint64_t offset = 0; offset < span; offset += K_CACHE_LINE_SIZE) {
            __builtin_prefetch(first + offset, 0, 3);
        }
    };
    prefetch_l1(record + one_bit_offset_, one_bit_code_size_);
    if (include_supplement) {
        prefetch_l1(record + supplement_offset_, supplement_code_size_);
    }
}

uint32_t
HGraphRaBitQFusedDataCell::GetNeighborSize(InnerIdType id) const {
    uint32_t count = 0;
    std::memcpy(&count, GetNodeRecord(id) + K_COUNT_OFFSET, sizeof(count));
    return count;
}

void
HGraphRaBitQFusedDataCell::GetNeighbors(InnerIdType id, Vector<InnerIdType>& neighbor_ids) const {
    const auto* record = GetNodeRecord(id);
    const auto count = GetNeighborSize(id);
    if (count > maximum_degree_) {
        neighbor_ids.clear();
        return;
    }
    const auto* input = GetNeighborData(record);
    neighbor_ids.clear();
    neighbor_ids.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (input[i] < total_count_) {
            neighbor_ids.push_back(input[i]);
        }
    }
}

bool
HGraphRaBitQFusedDataCell::CheckIdExists(InnerIdType id) const {
    return id < total_count_ and id < max_capacity_;
}

void
HGraphRaBitQFusedDataCell::Resize(InnerIdType new_size) {
    std::unique_lock lock(storage_mutex_);
    if (new_size <= max_capacity_) {
        return;
    }
    Reallocate(new_size);
}

void
HGraphRaBitQFusedDataCell::Prefetch(InnerIdType id, uint32_t neighbor_i) {
    const auto* record = GetNodeRecord(id);
    __builtin_prefetch(
        record + neighbors_offset_ + static_cast<uint64_t>(neighbor_i) * sizeof(InnerIdType), 0, 3);
    __builtin_prefetch(record + one_bit_offset_, 0, 3);
}

void
HGraphRaBitQFusedDataCell::Serialize(StreamWriter& writer) {
    GraphInterface::Serialize(writer);
    StreamWriter::WriteObj(writer, K_SERIALIZATION_VERSION);
    StreamWriter::WriteObj(writer, record_size_);
    StreamWriter::WriteObj(writer, neighbors_offset_);
    StreamWriter::WriteObj(writer, cluster_id_offset_);
    StreamWriter::WriteObj(writer, label_offset_);
    StreamWriter::WriteObj(writer, one_bit_offset_);
    StreamWriter::WriteObj(writer, supplement_offset_);
    StreamWriter::WriteObj(writer, one_bit_code_size_);
    StreamWriter::WriteObj(writer, supplement_code_size_);
    StreamWriter::WriteString(writer, codec_model_);
    const uint64_t bytes = static_cast<uint64_t>(max_capacity_) * record_size_;
    StreamWriter::WriteObj(writer, bytes);
    writer.Write(reinterpret_cast<const char*>(storage_.data() + aligned_offset_), bytes);
}

void
HGraphRaBitQFusedDataCell::Deserialize(StreamReader& reader) {
    const auto expected_maximum_degree = maximum_degree_;
    const FusedWireLayout expected_layout{record_size_,
                                          neighbors_offset_,
                                          cluster_id_offset_,
                                          label_offset_,
                                          one_bit_offset_,
                                          supplement_offset_,
                                          one_bit_code_size_,
                                          supplement_code_size_};

    GraphInterface::Deserialize(reader);
    uint32_t version = 0;
    StreamReader::ReadObj(reader, version);
    CHECK_ARGUMENT(version == K_SERIALIZATION_VERSION,
                   "unsupported fused graph serialization version");

    FusedWireLayout layout;
    StreamReader::ReadObj(reader, layout.record_size);
    StreamReader::ReadObj(reader, layout.neighbors_offset);
    StreamReader::ReadObj(reader, layout.cluster_id_offset);
    StreamReader::ReadObj(reader, layout.label_offset);
    StreamReader::ReadObj(reader, layout.one_bit_offset);
    StreamReader::ReadObj(reader, layout.supplement_offset);
    StreamReader::ReadObj(reader, layout.one_bit_code_size);
    StreamReader::ReadObj(reader, layout.supplement_code_size);
    CHECK_ARGUMENT(total_count_ <= max_capacity_, "invalid fused graph count and capacity");
    CHECK_ARGUMENT(maximum_degree_ == expected_maximum_degree,
                   "fused graph maximum degree does not match construction parameters");

    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.record_size > 0 and layout.record_size % K_CACHE_LINE_SIZE == 0,
        "invalid fused graph record stride");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.neighbors_offset == K_HEADER_SIZE and
            layout.neighbors_offset % alignof(InnerIdType) == 0,
        "invalid fused graph neighbor offset");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.cluster_id_offset >= layout.neighbors_offset and
            static_cast<uint64_t>(maximum_degree_) <=
                (layout.cluster_id_offset - layout.neighbors_offset) / sizeof(InnerIdType),
        "invalid fused graph cluster offset");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.cluster_id_offset % alignof(uint32_t) == 0 and
            layout.label_offset >= layout.cluster_id_offset and
            sizeof(uint32_t) <= layout.label_offset - layout.cluster_id_offset,
        "invalid fused graph label offset");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.label_offset % alignof(LabelType) == 0 and
            layout.one_bit_offset >= layout.label_offset and
            sizeof(LabelType) <= layout.one_bit_offset - layout.label_offset,
        "invalid fused graph one-bit offset");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.supplement_offset >= layout.one_bit_offset and
            layout.one_bit_code_size <= layout.supplement_offset - layout.one_bit_offset,
        "invalid fused graph supplement offset");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.record_size >= layout.supplement_offset and
            layout.supplement_code_size <= layout.record_size - layout.supplement_offset,
        "invalid fused graph code bounds");

    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        max_capacity_ <=
            (std::numeric_limits<uint64_t>::max() - (K_CACHE_LINE_SIZE - 1)) / layout.record_size,
        "fused graph capacity and record stride overflow");

    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.record_size == expected_layout.record_size and
            layout.neighbors_offset == expected_layout.neighbors_offset and
            layout.cluster_id_offset == expected_layout.cluster_id_offset and
            layout.label_offset == expected_layout.label_offset and
            layout.one_bit_offset == expected_layout.one_bit_offset and
            layout.supplement_offset == expected_layout.supplement_offset,
        "fused graph record layout does not match construction parameters");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.one_bit_code_size == expected_layout.one_bit_code_size and
            layout.supplement_code_size == expected_layout.supplement_code_size,
        "fused graph code sizes do not match construction parameters");
    uint64_t codec_model_size = 0;
    StreamReader::ReadObj(reader, codec_model_size);
    if (codec_model_size != 0) {
        CHECK_ARGUMENT(codec_model_size == fused_codec_model_size(dim_),
                       "invalid fused RaBitQ codec payload size");
    }
    auto available_bytes = remaining_bytes(reader);
    CHECK_ARGUMENT(available_bytes >= sizeof(uint64_t), "truncated fused RaBitQ codec payload");
    available_bytes -= sizeof(uint64_t);
    CHECK_ARGUMENT(codec_model_size <= available_bytes, "truncated fused RaBitQ codec payload");
    std::string codec_model(codec_model_size, '\0');
    reader.Read(codec_model.data(), codec_model_size);

    uint64_t bytes = 0;
    StreamReader::ReadObj(reader, bytes);
    const uint64_t expected_bytes = static_cast<uint64_t>(max_capacity_) * layout.record_size;
    CHECK_ARGUMENT(bytes == expected_bytes, "invalid fused graph payload size");
    CHECK_ARGUMENT(bytes <= remaining_bytes(reader), "truncated fused graph node payload");

    Vector<uint8_t> replacement(bytes + K_CACHE_LINE_SIZE - 1, 0, allocator_);
    const auto replacement_address = reinterpret_cast<uintptr_t>(replacement.data());
    const uint64_t replacement_offset =
        (K_CACHE_LINE_SIZE - replacement_address % K_CACHE_LINE_SIZE) % K_CACHE_LINE_SIZE;
    if (bytes > 0) {
        reader.Read(reinterpret_cast<char*>(replacement.data() + replacement_offset), bytes);
    }

    codec_model_ = std::move(codec_model);
    storage_ = std::move(replacement);
    aligned_offset_ = replacement_offset;
    if (duplicate_tracker_ != nullptr) {
        duplicate_tracker_->Resize(max_capacity_);
    }
    CHECK_ARGUMENT(
        reinterpret_cast<uintptr_t>(storage_.data() + aligned_offset_) % K_CACHE_LINE_SIZE == 0,
        "fused graph node slab is not cache-line aligned");
}

uint64_t
HGraphRaBitQFusedDataCell::GetMemoryUsage() const {
    return sizeof(*this) + storage_.capacity() + codec_model_.capacity();
}

DuplicateTrackerPtr
HGraphRaBitQFusedDataCell::CreateDuplicateTracker() {
    return std::make_shared<DenseDuplicateTracker>(allocator_);
}

}  // namespace vsag
