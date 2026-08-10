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

#include "rabitq_split_datacell.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

namespace vsag {

namespace {
constexpr uint64_t K_CACHE_LINE_SIZE = 64;
constexpr uint64_t K_COUNT_OFFSET = 0;
constexpr uint64_t K_VERSION_OFFSET = sizeof(uint32_t);
constexpr uint64_t K_HEADER_SIZE = 2 * sizeof(uint32_t);
constexpr uint32_t K_SERIALIZATION_VERSION = 1;
constexpr uint64_t K_FUSED_CLUSTER_COUNT = 16;

struct fused_wire_layout {
    uint64_t record_size{0};
    uint64_t neighbors_offset{0};
    uint64_t cluster_id_offset{0};
    uint64_t label_offset{0};
    uint64_t one_bit_offset{0};
    uint64_t supplement_offset{0};
    uint64_t one_bit_code_size{0};
    uint64_t supplement_code_size{0};
    bool support_remove{false};
    uint32_t remove_flag_bit{0};
};

uint64_t
fused_codec_model_size(int64_t dim) {
    CHECK_ARGUMENT(dim > 0, "invalid fused RaBitQ dimension");
    constexpr uint64_t fixed_size = sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint32_t);
    constexpr uint64_t bytes_per_dimension = K_FUSED_CLUSTER_COUNT * sizeof(float);
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
    support_remove_ = graph_param->support_remove_;
    remove_flag_bit_ = graph_param->remove_flag_bit_;
    constexpr uint32_t id_width = sizeof(InnerIdType) * 8;
    CHECK_ARGUMENT(remove_flag_bit_ < id_width, "invalid fused graph remove flag bits");
    if (support_remove_) {
        CHECK_ARGUMENT(remove_flag_bit_ > 0, "fused graph removal requires version bits");
    }
    id_bit_ = id_width - remove_flag_bit_;
    remove_flag_mask_ = id_bit_ == id_width ? std::numeric_limits<InnerIdType>::max()
                                            : (InnerIdType{1} << id_bit_) - 1U;
    if (support_remove_) {
        CHECK_ARGUMENT(max_capacity_ <= remove_flag_mask_,
                       "fused graph initial capacity exceeds remove-id bits");
    }

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

    if (graph_param->use_reverse_edges_) {
        reverse_edges_ = std::make_unique<ReverseEdge>(allocator_);
    }
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

uint32_t
HGraphRaBitQFusedDataCell::NodeVersion(const uint8_t* record) {
    uint32_t version = 0;
    std::memcpy(&version, record + K_VERSION_OFFSET, sizeof(version));
    return version;
}

void
HGraphRaBitQFusedDataCell::SetNodeVersion(uint8_t* record, uint32_t version) {
    std::memcpy(record + K_VERSION_OFFSET, &version, sizeof(version));
}

void
HGraphRaBitQFusedDataCell::InsertNeighborsById(InnerIdType id,
                                               const Vector<InnerIdType>& neighbor_ids) {
    CHECK_ARGUMENT(neighbor_ids.size() <= maximum_degree_,
                   "fused node neighbor count exceeds maximum degree");
    if (id >= max_capacity_) {
        Resize(id + 1);
    }

    Vector<InnerIdType> old_neighbors(allocator_);
    if (reverse_edges_ != nullptr and id < total_count_) {
        GetNeighbors(id, old_neighbors);
    }
    UpdateReverseEdges(id, old_neighbors, neighbor_ids);

    auto* record = MutableNodeRecord(id);
    const auto count = static_cast<uint32_t>(neighbor_ids.size());
    std::memcpy(record + K_COUNT_OFFSET, &count, sizeof(count));
    auto* output = reinterpret_cast<InnerIdType*>(record + neighbors_offset_);
    for (uint64_t i = 0; i < neighbor_ids.size(); ++i) {
        auto neighbor = neighbor_ids[i];
        if (support_remove_) {
            const auto version = NodeVersion(GetNodeRecord(neighbor));
            neighbor |= static_cast<InnerIdType>(version << id_bit_);
        }
        output[i] = neighbor;
    }
    auto current = total_count_.load();
    while (current < id + 1 and not total_count_.compare_exchange_weak(current, id + 1)) {
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

void
HGraphRaBitQFusedDataCell::SetLabel(InnerIdType id, LabelType label) {
    CHECK_ARGUMENT(id < max_capacity_, "fused graph label id does not exist");
    std::memcpy(MutableNodeRecord(id) + label_offset_, &label, sizeof(label));
}

bool
HGraphRaBitQFusedDataCell::SyncNodeCodes(InnerIdType id,
                                         LabelType label,
                                         uint32_t cluster_id,
                                         const RaBitQSplitDataCellInterface& split_codes) {
    if (id >= max_capacity_) {
        Resize(id + 1);
    }
    auto* record = MutableNodeRecord(id);
    if (not split_codes.CopySplitCodes(id, record + one_bit_offset_, record + supplement_offset_)) {
        return false;
    }
    std::memcpy(record + cluster_id_offset_, &cluster_id, sizeof(cluster_id));
    std::memcpy(record + label_offset_, &label, sizeof(label));
    return true;
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
        InnerIdType neighbor = 0;
        if (not ResolveNeighbor(input[i], neighbor)) {
            continue;
        }
        neighbor_ids.push_back(neighbor);
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
    if (support_remove_ and new_size > remove_flag_mask_) {
        throw VsagException(ErrorType::INTERNAL_ERROR, "fused graph id capacity exceeded");
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
HGraphRaBitQFusedDataCell::DeleteNeighborsById(InnerIdType id) {
    CHECK_ARGUMENT(support_remove_, "remove is disabled for fused graph");
    CHECK_ARGUMENT(id < max_capacity_, "fused graph remove id does not exist");
    auto* record = MutableNodeRecord(id);
    const auto version = NodeVersion(record);
    CHECK_ARGUMENT(version < (1U << remove_flag_bit_) - 1U, "fused graph node version exhausted");
    SetNodeVersion(record, version + 1);
}

void
HGraphRaBitQFusedDataCell::RecoverDeleteNeighborsById(InnerIdType id) {
    CHECK_ARGUMENT(support_remove_, "remove is disabled for fused graph");
    CHECK_ARGUMENT(id < max_capacity_, "fused graph recover id does not exist");
    auto* record = MutableNodeRecord(id);
    const auto version = NodeVersion(record);
    CHECK_ARGUMENT(version > 0, "fused graph node has not been removed");
    SetNodeVersion(record, version - 1);
}

void
HGraphRaBitQFusedDataCell::Move(InnerIdType from, InnerIdType to) {
    if (from == to) {
        return;
    }
    if (to >= max_capacity_) {
        Resize(to + 1);
    }

    Vector<uint8_t> source_record(record_size_, allocator_);
    std::memcpy(source_record.data(), GetNodeRecord(from), record_size_);

    // Incoming edges encode the target's remove version, so restore it before rebuilding them.
    auto* target_record = MutableNodeRecord(to);
    SetNodeVersion(target_record, NodeVersion(source_record.data()));

    Vector<InnerIdType> reverse_neighbors(allocator_);
    GetIncomingNeighbors(from, reverse_neighbors);
    Vector<InnerIdType> neighbors(allocator_);
    // Move runs under HGraph's external mutation lock, so this temporary empty adjacency is not
    // observable by concurrent readers.
    InsertNeighborsById(to, neighbors);
    for (const auto reverse_neighbor : reverse_neighbors) {
        GetNeighbors(reverse_neighbor, neighbors);
        Vector<InnerIdType> replacement(allocator_);
        bool contains_to = false;
        for (const auto neighbor : neighbors) {
            if (neighbor != from) {
                replacement.push_back(neighbor);
            }
            contains_to = contains_to or neighbor == to;
        }
        if (not contains_to) {
            replacement.push_back(to);
        }
        InsertNeighborsById(reverse_neighbor, replacement);
    }

    Vector<InnerIdType> from_neighbors(allocator_);
    GetNeighbors(from, from_neighbors);
    InsertNeighborsById(to, from_neighbors);
    from_neighbors.clear();
    InsertNeighborsById(from, from_neighbors);

    std::memcpy(target_record + cluster_id_offset_,
                source_record.data() + cluster_id_offset_,
                record_size_ - cluster_id_offset_);
}

void
HGraphRaBitQFusedDataCell::ShrinkToFit(InnerIdType capacity) {
    std::unique_lock lock(storage_mutex_);
    if (capacity < total_count_) {
        capacity = total_count_;
    }
    Reallocate(capacity);
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
    StreamWriter::WriteObj(writer, support_remove_);
    StreamWriter::WriteObj(writer, remove_flag_bit_);
    StreamWriter::WriteString(writer, codec_model_);
    const uint64_t bytes = static_cast<uint64_t>(max_capacity_) * record_size_;
    StreamWriter::WriteObj(writer, bytes);
    writer.Write(reinterpret_cast<const char*>(storage_.data() + aligned_offset_), bytes);
}

void
HGraphRaBitQFusedDataCell::Deserialize(StreamReader& reader) {
    const auto expected_maximum_degree = maximum_degree_;
    const fused_wire_layout expected_layout{record_size_,
                                            neighbors_offset_,
                                            cluster_id_offset_,
                                            label_offset_,
                                            one_bit_offset_,
                                            supplement_offset_,
                                            one_bit_code_size_,
                                            supplement_code_size_,
                                            support_remove_,
                                            remove_flag_bit_};

    GraphInterface::Deserialize(reader);
    uint32_t version = 0;
    StreamReader::ReadObj(reader, version);
    CHECK_ARGUMENT(version == K_SERIALIZATION_VERSION,
                   "unsupported fused graph serialization version");

    fused_wire_layout layout;
    StreamReader::ReadObj(reader, layout.record_size);
    StreamReader::ReadObj(reader, layout.neighbors_offset);
    StreamReader::ReadObj(reader, layout.cluster_id_offset);
    StreamReader::ReadObj(reader, layout.label_offset);
    StreamReader::ReadObj(reader, layout.one_bit_offset);
    StreamReader::ReadObj(reader, layout.supplement_offset);
    StreamReader::ReadObj(reader, layout.one_bit_code_size);
    StreamReader::ReadObj(reader, layout.supplement_code_size);
    uint8_t support_remove = 0;
    static_assert(sizeof(support_remove) == sizeof(bool));
    StreamReader::ReadObj(reader, support_remove);
    CHECK_ARGUMENT(support_remove <= 1, "invalid fused graph remove flag");
    layout.support_remove = support_remove != 0;
    StreamReader::ReadObj(reader, layout.remove_flag_bit);

    constexpr uint32_t id_width = sizeof(InnerIdType) * 8;
    CHECK_ARGUMENT(layout.remove_flag_bit < id_width, "invalid fused graph remove flag bits");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        not layout.support_remove or layout.remove_flag_bit > 0,
        "fused graph removal requires version bits");
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

    const uint32_t wire_id_bit = id_width - layout.remove_flag_bit;
    const InnerIdType wire_remove_flag_mask = wire_id_bit == id_width
                                                  ? std::numeric_limits<InnerIdType>::max()
                                                  : (InnerIdType{1} << wire_id_bit) - 1U;
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        not layout.support_remove or max_capacity_ <= wire_remove_flag_mask,
        "fused graph capacity exceeds remove-id bits");
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
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        layout.support_remove == expected_layout.support_remove and
            layout.remove_flag_bit == expected_layout.remove_flag_bit,
        "fused graph remove parameters do not match construction parameters");

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

    Vector<uint8_t> node_payload(bytes, allocator_);
    if (bytes > 0) {
        reader.Read(reinterpret_cast<char*>(node_payload.data()), bytes);
    }

    codec_model_ = std::move(codec_model);
    id_bit_ = wire_id_bit;
    remove_flag_mask_ = wire_remove_flag_mask;
    storage_.clear();
    aligned_offset_ = 0;
    Reallocate(max_capacity_);
    CHECK_ARGUMENT(
        reinterpret_cast<uintptr_t>(storage_.data() + aligned_offset_) % K_CACHE_LINE_SIZE == 0,
        "fused graph node slab is not cache-line aligned");
    if (bytes > 0) {
        std::memcpy(storage_.data() + aligned_offset_, node_payload.data(), bytes);
    }
}

uint64_t
HGraphRaBitQFusedDataCell::GetMemoryUsage() const {
    uint64_t result = sizeof(*this) + storage_.capacity() + codec_model_.capacity();
    if (reverse_edges_ != nullptr) {
        result += reverse_edges_->GetMemoryUsage();
    }
    return result;
}

DuplicateTrackerPtr
HGraphRaBitQFusedDataCell::CreateDuplicateTracker() {
    return std::make_shared<DenseDuplicateTracker>(allocator_);
}

}  // namespace vsag
