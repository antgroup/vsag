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

#include "sindi_datacell_utils.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <type_traits>

#include "simd/fp16_simd.h"
#include "vsag_exception.h"

// NOLINTNEXTLINE(modernize-concat-nested-namespaces)
namespace vsag {
namespace sindi_datacell_utils {
namespace {

uint64_t
get_term_payload_size(uint32_t non_empty_window_count,
                      uint32_t posting_count,
                      uint32_t value_code_size) {
    return sizeof(uint32_t) +
           static_cast<uint64_t>(non_empty_window_count) * sizeof(TermWindowMeta) +
           static_cast<uint64_t>(posting_count) * sizeof(uint16_t) + GetIdsPadding(posting_count) +
           static_cast<uint64_t>(posting_count) * value_code_size;
}

void
validate_term_posting_record(const TermPostingRecord& record,
                             uint32_t term_dict_count,
                             uint32_t window_count) {
    CHECK_ARGUMENT(record.term_id < term_dict_count, "SINDI posting term exceeds term dictionary");
    CHECK_ARGUMENT(record.window_id < window_count, "SINDI posting window is out of range");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        record.posting.ids != nullptr && record.posting.values != nullptr,
        "SINDI non-empty posting has null payload");
}

}  // namespace

uint32_t
GetValueCodeSize(SparseValueQuantizationType type) {
    switch (type) {
        case SparseValueQuantizationType::FP32:
            return sizeof(float);
        case SparseValueQuantizationType::SQ8:
            return sizeof(uint8_t);
        case SparseValueQuantizationType::FP16:
            return sizeof(uint16_t);
        default:
            CHECK_ARGUMENT(false, "unknown sparse value quantization type");
    }
    return sizeof(float);
}

void
EncodeValue(float value,
            SparseValueQuantizationType type,
            const QuantizationParams* quantization_params,
            uint8_t* destination) {
    CHECK_ARGUMENT(destination != nullptr, "SINDI encoded value destination is null");
    if (type == SparseValueQuantizationType::SQ8) {
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            quantization_params != nullptr && quantization_params->diff != 0.0F,
            "SINDI SQ8 quantization parameters are invalid");
        const auto normalized =
            (value - quantization_params->min_val) / quantization_params->diff * 255.0F;
        *destination = static_cast<uint8_t>(std::clamp(normalized, 0.0F, 255.0F));
    } else if (type == SparseValueQuantizationType::FP16) {
        const auto fp16 = generic::FloatToFP16(value);
        std::memcpy(destination, &fp16, sizeof(fp16));
    } else {
        std::memcpy(destination, &value, sizeof(value));
    }
}

float
DecodeValue(const uint8_t* source,
            SparseValueQuantizationType type,
            const QuantizationParams* quantization_params) {
    CHECK_ARGUMENT(source != nullptr, "SINDI encoded value source is null");
    if (type == SparseValueQuantizationType::SQ8) {
        CHECK_ARGUMENT(quantization_params != nullptr,
                       "SINDI SQ8 quantization parameters are missing");
        return static_cast<float>(*source) / 255.0F * quantization_params->diff +
               quantization_params->min_val;
    }
    if (type == SparseValueQuantizationType::FP16) {
        uint16_t fp16 = 0;
        std::memcpy(&fp16, source, sizeof(fp16));
        return generic::FP16ToFloat(fp16);
    }
    float value = 0.0F;
    std::memcpy(&value, source, sizeof(value));
    return value;
}

void
SortPostingListByValue(uint16_t* ids,
                       uint8_t* data,
                       uint32_t posting_count,
                       SparseValueQuantizationType quantization_type,
                       Vector<uint32_t>& order,
                       Vector<uint16_t>& sorted_ids,
                       Vector<uint8_t>& sorted_data) {
    if (posting_count == 0) {
        return;
    }

    order.resize(posting_count);
    std::iota(order.begin(), order.end(), 0);
    if (posting_count == 1) {
        return;
    }

    const auto sort_by_code = [&order, ids, posting_count, &sorted_ids, &sorted_data](auto* codes) {
        using CodeType = std::remove_pointer_t<decltype(codes)>;
        const auto compare = [codes, ids](uint32_t left, uint32_t right) {
            if (codes[left] != codes[right]) {
                return codes[left] > codes[right];
            }
            return ids[left] < ids[right];
        };
        if (std::is_sorted(order.begin(), order.end(), compare)) {
            return;
        }
        std::sort(order.begin(), order.end(), compare);

        sorted_ids.resize(posting_count);
        sorted_data.resize(static_cast<uint64_t>(posting_count) * sizeof(CodeType));
        auto* sorted_codes = reinterpret_cast<CodeType*>(sorted_data.data());
        for (uint32_t i = 0; i < posting_count; ++i) {
            const auto source = order[i];
            sorted_ids[i] = ids[source];
            sorted_codes[i] = codes[source];
        }
        std::copy(sorted_ids.begin(), sorted_ids.end(), ids);
        std::copy(sorted_codes, sorted_codes + posting_count, codes);
    };
    switch (quantization_type) {
        case SparseValueQuantizationType::SQ8:
            sort_by_code(data);
            break;
        case SparseValueQuantizationType::FP16:
            sort_by_code(reinterpret_cast<uint16_t*>(data));
            break;
        case SparseValueQuantizationType::FP32:
            sort_by_code(reinterpret_cast<float*>(data));
            break;
        default:
            CHECK_ARGUMENT(false, "unknown sparse value quantization type");
    }
}

uint64_t
GetIdsPadding(uint64_t posting_count) {
    const auto ids_size = posting_count * sizeof(uint16_t);
    return (sizeof(uint32_t) - ids_size % sizeof(uint32_t)) % sizeof(uint32_t);
}

TermLayout
BuildTermLayout(uint32_t term_dict_count,
                uint32_t window_count,
                uint32_t value_code_size,
                std::vector<TermPostingRecord> postings) {
    postings.erase(std::remove_if(postings.begin(),
                                  postings.end(),
                                  [](const auto& record) { return record.posting.count == 0; }),
                   postings.end());
    for (const auto& record : postings) {
        validate_term_posting_record(record, term_dict_count, window_count);
    }
    std::sort(postings.begin(), postings.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.term_id != rhs.term_id) {
            return lhs.term_id < rhs.term_id;
        }
        return lhs.window_id < rhs.window_id;
    });

    TermLayout layout;
    layout.term_dict.resize(term_dict_count);
    layout.postings = std::move(postings);
    uint64_t payload_offset = 0;
    uint64_t record_index = 0;
    while (record_index < layout.postings.size()) {
        const auto term_id = layout.postings[record_index].term_id;
        uint32_t posting_count = 0;
        uint32_t non_empty_window_count = 0;
        uint32_t previous_window = std::numeric_limits<uint32_t>::max();
        while (record_index + non_empty_window_count < layout.postings.size()) {
            const auto& record = layout.postings[record_index + non_empty_window_count];
            if (record.term_id != term_id) {
                break;
            }
            CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
                previous_window == std::numeric_limits<uint32_t>::max() ||
                    record.window_id > previous_window,
                "SINDI term windows must be strictly increasing");
            CHECK_ARGUMENT(
                posting_count <= std::numeric_limits<uint32_t>::max() - record.posting.count,
                "SINDI term posting count exceeds uint32_t");
            posting_count += record.posting.count;
            previous_window = record.window_id;
            ++non_empty_window_count;
        }
        const auto payload_size =
            get_term_payload_size(non_empty_window_count, posting_count, value_code_size);
        CHECK_ARGUMENT(payload_size <= std::numeric_limits<uint32_t>::max(),
                       "SINDI term payload exceeds uint32_t");
        layout.term_dict[term_id] = {
            payload_offset, static_cast<uint32_t>(payload_size), posting_count};
        payload_offset += payload_size;
        record_index += non_empty_window_count;
    }
    layout.payload_size = payload_offset;
    return layout;
}

void
SerializeTermLayout(StreamWriter& writer, const TermLayout& layout, uint32_t value_code_size) {
    StreamWriter::WriteVector(writer, layout.term_dict);
    StreamWriter::WriteObj(writer, layout.payload_size);

    constexpr char padding[sizeof(uint32_t)] = {0};
    uint64_t record_index = 0;
    while (record_index < layout.postings.size()) {
        const auto term_id = layout.postings[record_index].term_id;
        uint32_t non_empty_window_count = 0;
        while (record_index + non_empty_window_count < layout.postings.size() &&
               layout.postings[record_index + non_empty_window_count].term_id == term_id) {
            ++non_empty_window_count;
        }
        StreamWriter::WriteObj(writer, non_empty_window_count);
        for (uint32_t index = 0; index < non_empty_window_count; ++index) {
            const auto& record = layout.postings[record_index + index];
            const TermWindowMeta meta{record.window_id, record.posting.count};
            StreamWriter::WriteObj(writer, meta);
        }
        uint32_t posting_count = 0;
        for (uint32_t index = 0; index < non_empty_window_count; ++index) {
            const auto& posting = layout.postings[record_index + index].posting;
            writer.Write(reinterpret_cast<const char*>(posting.ids),
                         static_cast<uint64_t>(posting.count) * sizeof(uint16_t));
            posting_count += posting.count;
        }
        const auto ids_padding = GetIdsPadding(posting_count);
        if (ids_padding != 0) {
            writer.Write(padding, ids_padding);
        }
        for (uint32_t index = 0; index < non_empty_window_count; ++index) {
            const auto& posting = layout.postings[record_index + index].posting;
            writer.Write(reinterpret_cast<const char*>(posting.values),
                         static_cast<uint64_t>(posting.count) * value_code_size);
        }
        record_index += non_empty_window_count;
    }
}

std::vector<DiskTermEntry>
DeserializeTermDictionary(StreamReader& reader, uint32_t term_id_limit) {
    uint64_t term_dict_count = 0;
    StreamReader::ReadObj(reader, term_dict_count);
    CHECK_ARGUMENT(term_dict_count <= static_cast<uint64_t>(term_id_limit) + 1,
                   "SINDI_V2 term dict exceeds term_id_limit");
    CHECK_ARGUMENT(reader.GetCursor() <= reader.Length(),
                   "SINDI_V2 term dictionary starts outside stream");
    const auto remaining = reader.Length() - reader.GetCursor();
    CHECK_ARGUMENT(term_dict_count <= remaining / sizeof(DiskTermEntry),
                   "SINDI_V2 term dictionary exceeds stream length");
    std::vector<DiskTermEntry> term_dict(term_dict_count);
    reader.Read(reinterpret_cast<char*>(term_dict.data()), term_dict_count * sizeof(DiskTermEntry));
    return term_dict;
}

void
ValidateTermDict(const std::vector<DiskTermEntry>& term_dict, uint64_t payload_size) {
    uint64_t previous_payload_end = 0;
    for (const auto& entry : term_dict) {
        if (entry.posting_count == 0) {
            CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
                entry.posting_payload_offset == 0 && entry.posting_payload_size == 0,
                "empty SINDI_V2 term has a payload descriptor");
            continue;
        }
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            entry.posting_payload_size > 0 &&
                entry.posting_payload_offset == previous_payload_end &&
                entry.posting_payload_offset <= payload_size &&
                entry.posting_payload_size <= payload_size - entry.posting_payload_offset,
            "invalid SINDI_V2 term dictionary layout");
        previous_payload_end = entry.posting_payload_offset + entry.posting_payload_size;
    }
    CHECK_ARGUMENT(previous_payload_end == payload_size,
                   "SINDI_V2 posting payload size does not match term dictionary");
}

namespace {

SindiTermBuffer
parse_term_payload_impl(const uint8_t* payload,
                        uint64_t payload_size,
                        const DiskTermEntry& entry,
                        uint32_t window_count,
                        uint32_t window_size,
                        uint64_t total_count,
                        uint32_t value_code_size,
                        Allocator* allocator,
                        bool view_payload) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        entry.posting_count > 0 && entry.posting_payload_size == payload_size,
        "invalid SINDI_V2 term payload descriptor");
    CHECK_ARGUMENT(payload_size >= sizeof(uint32_t), "SINDI_V2 term payload is truncated");

    uint64_t cursor = 0;
    uint32_t non_empty_window_count = 0;
    std::memcpy(&non_empty_window_count, payload + cursor, sizeof(non_empty_window_count));
    cursor += sizeof(non_empty_window_count);
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        non_empty_window_count > 0 && non_empty_window_count <= window_count &&
            non_empty_window_count <= (payload_size - cursor) / sizeof(TermWindowMeta),
        "invalid SINDI_V2 non-empty window count");

    const auto metadata_offset = cursor;
    cursor += static_cast<uint64_t>(non_empty_window_count) * sizeof(TermWindowMeta);
    const auto ids_size = static_cast<uint64_t>(entry.posting_count) * sizeof(uint16_t);
    const auto ids_padding = GetIdsPadding(entry.posting_count);
    const auto values_size = static_cast<uint64_t>(entry.posting_count) * value_code_size;
    CHECK_ARGUMENT(ids_size <= payload_size - cursor, "SINDI_V2 term ids are truncated");
    const auto ids_offset = cursor;
    cursor += ids_size;
    CHECK_ARGUMENT(ids_padding <= payload_size - cursor, "SINDI_V2 term id padding is truncated");
    cursor += ids_padding;
    CHECK_ARGUMENT(values_size == payload_size - cursor, "SINDI_V2 term values size is invalid");

    SindiTermBuffer buffer(allocator);
    buffer.window_offsets.resize(static_cast<uint64_t>(window_count) + 1, 0);
    if (view_payload) {
        buffer.external_ids = reinterpret_cast<const uint16_t*>(payload + ids_offset);
        buffer.external_values = payload + cursor;
        buffer.external_values_size = values_size;
    } else {
        buffer.ids.resize(entry.posting_count);
        buffer.values.resize(values_size);
        std::memcpy(buffer.ids.data(), payload + ids_offset, ids_size);
        std::memcpy(buffer.values.data(), payload + cursor, values_size);
    }
    const auto* ids = buffer.IdsData();

    uint32_t posting_count = 0;
    uint32_t previous_window = std::numeric_limits<uint32_t>::max();
    uint32_t next_offset = 0;
    uint64_t metadata_cursor = metadata_offset;
    for (uint32_t index = 0; index < non_empty_window_count; ++index) {
        TermWindowMeta meta;
        std::memcpy(&meta, payload + metadata_cursor, sizeof(meta));
        metadata_cursor += sizeof(meta);
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            meta.posting_count > 0 && meta.window_id < window_count &&
                (previous_window == std::numeric_limits<uint32_t>::max() ||
                 meta.window_id > previous_window),
            "invalid SINDI_V2 term window metadata");
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            posting_count <= entry.posting_count &&
                meta.posting_count <= entry.posting_count - posting_count,
            "SINDI_V2 term posting count does not match term dictionary");
        while (next_offset <= meta.window_id) {
            buffer.window_offsets[next_offset++] = posting_count;
        }
        const auto window_start = static_cast<uint64_t>(meta.window_id) * window_size;
        const auto window_document_count = static_cast<uint32_t>(std::min<uint64_t>(
            window_size, total_count > window_start ? total_count - window_start : 0));
        for (uint32_t posting = 0; posting < meta.posting_count; ++posting) {
            CHECK_ARGUMENT(ids[posting_count + posting] < window_document_count,
                           "SINDI_V2 posting id exceeds its window document count");
        }
        posting_count += meta.posting_count;
        previous_window = meta.window_id;
    }
    while (next_offset <= window_count) {
        buffer.window_offsets[next_offset++] = posting_count;
    }
    CHECK_ARGUMENT(posting_count == entry.posting_count,
                   "SINDI_V2 term posting count does not match term dictionary");
    return buffer;
}

}  // namespace

SindiTermBuffer
ParseTermPayload(const uint8_t* payload,
                 uint64_t payload_size,
                 const DiskTermEntry& entry,
                 uint32_t window_count,
                 uint32_t window_size,
                 uint64_t total_count,
                 uint32_t value_code_size,
                 Allocator* allocator) {
    return parse_term_payload_impl(payload,
                                   payload_size,
                                   entry,
                                   window_count,
                                   window_size,
                                   total_count,
                                   value_code_size,
                                   allocator,
                                   false);
}

SindiTermBuffer
ViewTermPayload(const uint8_t* payload,
                uint64_t payload_size,
                const DiskTermEntry& entry,
                uint32_t window_count,
                uint32_t window_size,
                uint64_t total_count,
                uint32_t value_code_size,
                Allocator* allocator) {
    return parse_term_payload_impl(payload,
                                   payload_size,
                                   entry,
                                   window_count,
                                   window_size,
                                   total_count,
                                   value_code_size,
                                   allocator,
                                   true);
}

SindiTermBuffer
ReadTermPayload(StreamReader& reader,
                uint64_t payload_start,
                uint64_t payload_size,
                const DiskTermEntry& entry,
                uint32_t window_count,
                uint32_t window_size,
                uint64_t total_count,
                uint32_t value_code_size,
                Allocator* allocator) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        entry.posting_payload_offset <= payload_size &&
            entry.posting_payload_size <= payload_size - entry.posting_payload_offset,
        "SINDI_V2 term payload is out of range");
    Vector<uint8_t> payload(entry.posting_payload_size, allocator);
    reader.PushSeek(payload_start + entry.posting_payload_offset);
    reader.Read(reinterpret_cast<char*>(payload.data()), payload.size());
    reader.PopSeek();
    return ParseTermPayload(payload.data(),
                            payload.size(),
                            entry,
                            window_count,
                            window_size,
                            total_count,
                            value_code_size,
                            allocator);
}

TermPayloadLayout
ReadTermPayloadMetadata(StreamReader& reader,
                        uint64_t payload_start,
                        uint64_t payload_size,
                        const DiskTermEntry& entry,
                        uint32_t window_count,
                        uint32_t value_code_size,
                        Vector<TermWindowMeta>& metadata) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        entry.posting_count > 0 && entry.posting_payload_offset <= payload_size &&
            entry.posting_payload_size <= payload_size - entry.posting_payload_offset,
        "SINDI_V2 term payload is out of range");
    CHECK_ARGUMENT(entry.posting_payload_size >= sizeof(uint32_t),
                   "SINDI_V2 term payload is truncated");
    CHECK_ARGUMENT(
        payload_start <= std::numeric_limits<uint64_t>::max() - entry.posting_payload_offset,
        "SINDI_V2 term payload offset overflows uint64_t");

    const auto term_payload_start = payload_start + entry.posting_payload_offset;
    uint32_t non_empty_window_count = 0;
    reader.PushSeek(term_payload_start);
    StreamReader::ReadObj(reader, non_empty_window_count);
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        non_empty_window_count > 0 && non_empty_window_count <= window_count,
        "invalid SINDI_V2 non-empty window count");
    const auto metadata_size =
        static_cast<uint64_t>(non_empty_window_count) * sizeof(TermWindowMeta);
    CHECK_ARGUMENT(metadata_size <= entry.posting_payload_size - sizeof(uint32_t),
                   "invalid SINDI_V2 non-empty window count");
    metadata.resize(non_empty_window_count);
    reader.Read(reinterpret_cast<char*>(metadata.data()), metadata_size);
    reader.PopSeek();

    const auto expected_payload_size =
        get_term_payload_size(non_empty_window_count, entry.posting_count, value_code_size);
    CHECK_ARGUMENT(expected_payload_size == entry.posting_payload_size,
                   "SINDI_V2 term values size is invalid");

    uint32_t posting_count = 0;
    uint32_t previous_window = std::numeric_limits<uint32_t>::max();
    for (const auto& meta : metadata) {
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            meta.posting_count > 0 && meta.window_id < window_count &&
                (previous_window == std::numeric_limits<uint32_t>::max() ||
                 meta.window_id > previous_window),
            "invalid SINDI_V2 term window metadata");
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            posting_count <= entry.posting_count &&
                meta.posting_count <= entry.posting_count - posting_count,
            "SINDI_V2 term posting count does not match term dictionary");
        posting_count += meta.posting_count;
        previous_window = meta.window_id;
    }
    CHECK_ARGUMENT(posting_count == entry.posting_count,
                   "SINDI_V2 term posting count does not match term dictionary");

    const auto ids_offset = term_payload_start + sizeof(uint32_t) + metadata_size;
    const auto values_offset = ids_offset +
                               static_cast<uint64_t>(entry.posting_count) * sizeof(uint16_t) +
                               GetIdsPadding(entry.posting_count);
    return {ids_offset, values_offset, entry.posting_count};
}

void
ReadTermPostingRange(StreamReader& reader,
                     const TermPayloadLayout& layout,
                     uint32_t posting_offset,
                     uint32_t posting_count,
                     uint32_t value_code_size,
                     uint16_t* ids,
                     uint8_t* values) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        posting_count > 0 && posting_offset <= layout.posting_count &&
            posting_count <= layout.posting_count - posting_offset,
        "SINDI_V2 term posting range is out of bounds");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        ids != nullptr && values != nullptr,
        "SINDI_V2 term posting destination is null");

    reader.PushSeek(layout.ids_offset + static_cast<uint64_t>(posting_offset) * sizeof(uint16_t));
    reader.Read(reinterpret_cast<char*>(ids),
                static_cast<uint64_t>(posting_count) * sizeof(uint16_t));
    reader.Seek(layout.values_offset + static_cast<uint64_t>(posting_offset) * value_code_size);
    reader.Read(reinterpret_cast<char*>(values),
                static_cast<uint64_t>(posting_count) * value_code_size);
    reader.PopSeek();
}

}  // namespace sindi_datacell_utils
}  // namespace vsag
