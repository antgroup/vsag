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

#pragma once

#include <vector>

#include "algorithm/sindi/sindi_parameter.h"
#include "datacell/sindi_term_datacell.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"

namespace vsag::sindi_datacell_utils {

struct TermPostingRecord {
    uint32_t term_id{0};
    uint32_t window_id{0};
    SindiTermPostingView posting;
};

struct TermLayout {
    std::vector<DiskTermEntry> term_dict;
    std::vector<TermPostingRecord> postings;
    uint64_t payload_size{0};
};

[[nodiscard]] uint32_t
GetValueCodeSize(SparseValueQuantizationType type);

void
EncodeValue(float value,
            SparseValueQuantizationType type,
            const QuantizationParams* quantization_params,
            uint8_t* destination);

[[nodiscard]] float
DecodeValue(const uint8_t* source,
            SparseValueQuantizationType type,
            const QuantizationParams* quantization_params);

[[nodiscard]] uint64_t
GetIdsPadding(uint64_t posting_count);

[[nodiscard]] TermLayout
BuildTermLayout(uint32_t term_dict_count,
                uint32_t window_count,
                uint32_t value_code_size,
                std::vector<TermPostingRecord> postings);

void
SerializeTermLayout(StreamWriter& writer, const TermLayout& layout, uint32_t value_code_size);

[[nodiscard]] std::vector<DiskTermEntry>
DeserializeTermDictionary(StreamReader& reader, uint32_t term_id_limit);

[[nodiscard]] SindiTermBuffer
ReadTermPayload(StreamReader& reader,
                uint64_t payload_start,
                uint64_t payload_size,
                const DiskTermEntry& entry,
                uint32_t window_count,
                uint32_t window_size,
                uint64_t total_count,
                uint32_t value_code_size,
                Allocator* allocator);

[[nodiscard]] SindiTermBuffer
ParseTermPayload(const uint8_t* payload,
                 uint64_t payload_size,
                 const DiskTermEntry& entry,
                 uint32_t window_count,
                 uint32_t window_size,
                 uint64_t total_count,
                 uint32_t value_code_size,
                 Allocator* allocator);

[[nodiscard]] SindiTermBuffer
ViewTermPayload(const uint8_t* payload,
                uint64_t payload_size,
                const DiskTermEntry& entry,
                uint32_t window_count,
                uint32_t window_size,
                uint64_t total_count,
                uint32_t value_code_size,
                Allocator* allocator);

void
ValidateTermDict(const std::vector<DiskTermEntry>& term_dict, uint64_t payload_size);

}  // namespace vsag::sindi_datacell_utils
