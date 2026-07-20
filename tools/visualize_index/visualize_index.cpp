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

#include <algorithm>
#include <argparse/argparse.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "storage/serialization_tags.h"
#include "vsag/constants.h"

namespace {

constexpr uint64_t STREAM_BLOCK_HEADER_SIZE = 4 + 4 + 8 + 8 + 4;
constexpr uint64_t STREAM_BLOCK_CRITICAL_FLAG = 1;
constexpr uint64_t STREAM_METADATA_CHECKSUM_SIZE = 4;
constexpr uint64_t STREAM_HEADER_MAX_METADATA_LEN = 16ULL * 1024ULL * 1024ULL;
constexpr uint64_t TERMINAL_BAR_WIDTH = 96;
constexpr uint64_t TERMINAL_VERTICAL_BAR_WIDTH = 18;
constexpr uint64_t TERMINAL_VERTICAL_SMALL_BLOCK_SIZE = 4ULL * 1024ULL;
constexpr uint64_t TERMINAL_VERTICAL_HUGE_ROWS = 4;
constexpr uint64_t TERMINAL_VERTICAL_MAX_ROWS = 10;
constexpr double EXPLODED_LARGE_SEGMENT_THRESHOLD = 0.60;
constexpr double EXPLODED_LARGE_SEGMENT_WIDTH = 0.18;

struct StreamBlockHeaderView {
    uint32_t tag{0};
    uint32_t block_version{0};
    uint64_t flags{0};
    uint64_t value_len{0};
    uint32_t payload_checksum{0};

    [[nodiscard]] bool
    IsCritical() const {
        return (flags & STREAM_BLOCK_CRITICAL_FLAG) != 0;
    }

    [[nodiscard]] bool
    IsSectionEnd() const {
        return tag == 0;
    }
};

struct Segment {
    std::string name;
    std::string kind;
    uint64_t offset{0};
    uint64_t size{0};
    std::string detail;
    std::string color;
};

struct ParsedIndex {
    uint64_t file_size{0};
    uint16_t format_major{0};
    uint16_t format_minor{0};
    std::string metadata_json;
    std::vector<Segment> segments;
};

struct VisualSegment {
    const Segment* segment{nullptr};
    double width_ratio{0.0};
    bool folded{false};
};

struct LogicalBlock {
    std::string name;
    std::string kind;
    uint64_t offset{0};
    uint64_t size{0};
    std::string color;
    std::vector<const Segment*> segments;
};

struct VisualBlock {
    const LogicalBlock* block{nullptr};
    double width_ratio{0.0};
    bool folded{false};
};

std::string
escape_html(const std::string& input) {
    std::string output;
    output.reserve(input.size());
    for (char ch : input) {
        switch (ch) {
            case '&':
                output += "&amp;";
                break;
            case '<':
                output += "&lt;";
                break;
            case '>':
                output += "&gt;";
                break;
            case '"':
                output += "&quot;";
                break;
            case '\'':
                output += "&#39;";
                break;
            default:
                output += ch;
                break;
        }
    }
    return output;
}

std::string
format_bytes(uint64_t bytes) {
    constexpr double kib = 1024.0;
    constexpr double mib = 1024.0 * 1024.0;
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;

    std::ostringstream out;
    if (bytes >= static_cast<uint64_t>(gib)) {
        out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / gib << " GiB";
    } else if (bytes >= static_cast<uint64_t>(mib)) {
        out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / mib << " MiB";
    } else if (bytes >= static_cast<uint64_t>(kib)) {
        out << std::fixed << std::setprecision(2) << static_cast<double>(bytes) / kib << " KiB";
    } else {
        out << bytes << " B";
    }
    return out.str();
}

std::string
format_percent(uint64_t size, uint64_t total) {
    std::ostringstream out;
    const double percent = total == 0 ? 0.0 : static_cast<double>(size) * 100.0 / total;
    out << std::fixed << std::setprecision(2) << percent << "%";
    return out.str();
}

uint64_t
file_size(std::ifstream& input) {
    input.seekg(0, std::ios::end);
    const auto end = input.tellg();
    if (end < 0) {
        throw std::runtime_error("failed to get input file size");
    }
    input.seekg(0, std::ios::beg);
    return static_cast<uint64_t>(end);
}

uint64_t
current_offset(std::istream& input) {
    const auto cursor = input.tellg();
    if (cursor < 0) {
        throw std::runtime_error("failed to get current input offset");
    }
    return static_cast<uint64_t>(cursor);
}

void
read_exact(std::istream& input, char* data, uint64_t size) {
    if (size > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("read size is too large");
    }
    input.read(data, static_cast<std::streamsize>(size));
    if (!input) {
        throw std::runtime_error("unexpected end of index file");
    }
}

template <typename T>
T
read_obj(std::istream& input) {
    T value{};
    read_exact(input, reinterpret_cast<char*>(&value), sizeof(value));
    return value;
}

uint32_t
calculate_checksum(std::string_view bytes) {
    const uint32_t polynomial = 0xEDB88320;
    uint32_t crc = 0xFFFFFFFF;
    for (const char& byte : bytes) {
        crc ^= static_cast<uint8_t>(byte);
        for (uint64_t j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ ((crc & 1) == 1 ? polynomial : 0);
        }
    }
    return crc ^ 0xFFFFFFFF;
}

StreamBlockHeaderView
read_block_header(std::istream& input) {
    StreamBlockHeaderView header;
    header.tag = read_obj<uint32_t>(input);
    header.block_version = read_obj<uint32_t>(input);
    header.flags = read_obj<uint64_t>(input);
    header.value_len = read_obj<uint64_t>(input);
    header.payload_checksum = read_obj<uint32_t>(input);
    if (header.IsSectionEnd() && (header.block_version != 0 || header.flags != 0 ||
                                  header.value_len != 0 || header.payload_checksum != 0)) {
        throw std::runtime_error("invalid streaming serialization section end block");
    }
    return header;
}

void
skip_forward(std::istream& input, uint64_t size) {
    constexpr uint64_t buffer_size = 8192;
    std::array<char, buffer_size> buffer{};
    uint64_t remaining = size;
    while (remaining > 0) {
        const uint64_t read_size = std::min<uint64_t>(remaining, buffer_size);
        read_exact(input, buffer.data(), read_size);
        remaining -= read_size;
    }
}

void
add_segment(std::vector<Segment>& segments,
            std::string name,
            std::string kind,
            uint64_t offset,
            uint64_t size,
            std::string detail,
            std::string color) {
    segments.push_back(Segment{
        std::move(name), std::move(kind), offset, size, std::move(detail), std::move(color)});
}

std::string
streaming_tag_name(uint32_t tag) {
    std::string name = vsag::StreamSerializationTagName(tag);
    if (name == "unknown") {
        name += "_" + std::to_string(tag);
    }
    return name;
}

ParsedIndex
parse_streaming_index(const std::string& index_path) {
    std::ifstream input(index_path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("failed to open index file: " + index_path);
    }

    ParsedIndex parsed;
    parsed.file_size = file_size(input);
    const uint64_t magic_offset = current_offset(input);
    std::array<char, 9> magic{};
    read_exact(input, magic.data(), 8);
    if (std::string_view(magic.data(), 8) != vsag::SERIAL_STREAM_MAGIC) {
        throw std::runtime_error("unsupported index format: only streaming magic " +
                                 std::string(vsag::SERIAL_STREAM_MAGIC) + " is supported");
    }
    add_segment(parsed.segments,
                "stream_magic",
                "header",
                magic_offset,
                8,
                std::string(magic.data(), 8),
                "#5465ff");

    const uint64_t version_offset = current_offset(input);
    parsed.format_major = read_obj<uint16_t>(input);
    parsed.format_minor = read_obj<uint16_t>(input);
    add_segment(parsed.segments,
                "format_version",
                "header",
                version_offset,
                4,
                std::to_string(parsed.format_major) + "." + std::to_string(parsed.format_minor),
                "#6a8dff");

    const uint64_t metadata_len_offset = current_offset(input);
    const uint64_t metadata_len = read_obj<uint64_t>(input);
    add_segment(parsed.segments,
                "metadata_length",
                "header",
                metadata_len_offset,
                8,
                std::to_string(metadata_len),
                "#8aa4ff");

    const uint64_t metadata_offset = current_offset(input);
    if (metadata_len > STREAM_HEADER_MAX_METADATA_LEN) {
        throw std::runtime_error("streaming serialization metadata is too large");
    }
    parsed.metadata_json.resize(metadata_len);
    if (metadata_len > 0) {
        read_exact(input, parsed.metadata_json.data(), metadata_len);
    }
    add_segment(parsed.segments,
                "metadata_json",
                "metadata",
                metadata_offset,
                metadata_len,
                "StreamHeader metadata",
                "#33a02c");

    const uint64_t metadata_checksum_offset = current_offset(input);
    const uint32_t metadata_checksum = read_obj<uint32_t>(input);
    const uint32_t calculated_checksum = calculate_checksum(parsed.metadata_json);
    add_segment(parsed.segments,
                "metadata_checksum",
                "header",
                metadata_checksum_offset,
                STREAM_METADATA_CHECKSUM_SIZE,
                metadata_checksum == calculated_checksum ? "crc32 ok" : "crc32 mismatch",
                metadata_checksum == calculated_checksum ? "#74c476" : "#e31a1c");

    while (true) {
        const uint64_t block_header_offset = current_offset(input);
        auto block_header = read_block_header(input);
        const std::string tag_name = streaming_tag_name(block_header.tag);

        if (block_header.IsSectionEnd()) {
            add_segment(parsed.segments,
                        "section_end",
                        "block_header",
                        block_header_offset,
                        STREAM_BLOCK_HEADER_SIZE,
                        "tag=0",
                        "#6c757d");
            break;
        }

        std::ostringstream block_detail;
        block_detail << "tag=" << block_header.tag << " (" << tag_name
                     << "), version=" << block_header.block_version
                     << ", critical=" << (block_header.IsCritical() ? "true" : "false")
                     << ", payload=" << block_header.value_len << " bytes";
        add_segment(parsed.segments,
                    tag_name + "_header",
                    "block_header",
                    block_header_offset,
                    STREAM_BLOCK_HEADER_SIZE,
                    block_detail.str(),
                    "#fdbf6f");

        if (block_header.value_len == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error(
                "chunked streaming block payload is not supported by visualize_index yet");
        }

        const uint64_t payload_offset = current_offset(input);
        add_segment(parsed.segments,
                    tag_name + "_payload",
                    block_header.IsCritical() ? "critical_payload" : "optional_payload",
                    payload_offset,
                    block_header.value_len,
                    block_detail.str(),
                    block_header.IsCritical() ? "#1f78b4" : "#a6cee3");
        skip_forward(input, block_header.value_len);
    }

    const uint64_t consumed = current_offset(input);
    if (consumed < parsed.file_size) {
        add_segment(parsed.segments,
                    "trailing_bytes",
                    "trailing",
                    consumed,
                    parsed.file_size - consumed,
                    "bytes after SECTION_END",
                    "#b15928");
    }

    return parsed;
}

double
segment_ratio(const Segment& segment, uint64_t total) {
    if (total == 0) {
        return 0.0;
    }
    return static_cast<double>(segment.size) / static_cast<double>(total);
}

std::vector<VisualSegment>
build_raw_view(const ParsedIndex& parsed) {
    std::vector<VisualSegment> view;
    view.reserve(parsed.segments.size());
    for (const auto& segment : parsed.segments) {
        view.push_back(VisualSegment{&segment, segment_ratio(segment, parsed.file_size), false});
    }
    return view;
}

std::vector<uint64_t>
allocate_terminal_units(const std::vector<VisualSegment>& view, uint64_t total_units) {
    std::vector<uint64_t> units(view.size(), 0);
    if (view.empty() || total_units == 0) {
        return units;
    }

    uint64_t used_units = 0;
    for (uint64_t i = 0; i < view.size(); ++i) {
        const auto index = static_cast<size_t>(i);
        const auto& segment = *view[index].segment;
        units[index] = static_cast<uint64_t>(view[index].width_ratio * total_units);
        if (segment.size > 0 && units[index] == 0) {
            units[index] = 1;
        }
        used_units += units[index];
    }

    while (used_units > total_units) {
        auto largest = std::max_element(units.begin(), units.end());
        if (largest == units.end() || *largest <= 1) {
            break;
        }
        --(*largest);
        --used_units;
    }

    while (used_units < total_units) {
        auto largest =
            std::max_element(view.begin(), view.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.width_ratio < rhs.width_ratio;
            });
        if (largest == view.end()) {
            break;
        }
        ++units[static_cast<size_t>(std::distance(view.begin(), largest))];
        ++used_units;
    }
    return units;
}

std::string
ansi_color_for(const Segment& segment) {
    if (segment.name == "stream_magic") {
        return "\033[45m";
    }
    if (segment.name == "format_version") {
        return "\033[46m";
    }
    if (segment.name == "metadata_length") {
        return "\033[44m";
    }
    if (segment.name == "metadata_checksum") {
        return segment.detail == "crc32 ok" ? "\033[42m" : "\033[41m";
    }
    if (segment.kind == "metadata") {
        return "\033[42m";
    }
    if (segment.kind == "critical_payload") {
        return "\033[44m";
    }
    if (segment.kind == "optional_payload") {
        return "\033[46m";
    }
    if (segment.kind == "block_header") {
        return "\033[43m";
    }
    if (segment.kind == "trailing") {
        return "\033[41m";
    }
    if (segment.kind == "header") {
        return "\033[45m";
    }
    return "\033[47m";
}

void
print_terminal_horizontal_bar(const std::string& title, const std::vector<VisualSegment>& view) {
    std::cout << title << std::endl;
    auto widths = allocate_terminal_units(view, TERMINAL_BAR_WIDTH);
    for (uint64_t i = 0; i < view.size(); ++i) {
        const auto index = static_cast<size_t>(i);
        const auto& visual = view[index];
        const auto& segment = *visual.segment;
        const uint64_t width = widths[index];
        const char fill = visual.folded ? '/' : ' ';
        std::cout << ansi_color_for(segment) << std::string(static_cast<size_t>(width), fill)
                  << "\033[0m";
    }
    std::cout << std::endl << std::endl;
}

std::vector<LogicalBlock>
build_logical_blocks(const ParsedIndex& parsed);

void
print_terminal_logical_block_bars(const std::string& title,
                                  const std::vector<LogicalBlock>& blocks,
                                  uint64_t total_size);

void
print_summary(const ParsedIndex& parsed) {
    std::cout << "Streaming index layout" << std::endl;
    std::cout << "Format: " << parsed.format_major << "." << parsed.format_minor << std::endl;
    std::cout << "File size: " << parsed.file_size << " bytes (" << format_bytes(parsed.file_size)
              << ")" << std::endl
              << std::endl;

    auto raw_view = build_raw_view(parsed);
    print_terminal_horizontal_bar(
        "Raw horizontal layout by real byte proportion (rounded to 96 terminal columns):",
        raw_view);
    const auto logical_blocks = build_logical_blocks(parsed);
    print_terminal_logical_block_bars(
        "Exploded vertical layout by logical block size:", logical_blocks, parsed.file_size);

    std::cout << std::left << std::setw(4) << "#" << std::setw(26) << "name" << std::setw(18)
              << "kind" << std::right << std::setw(14) << "offset" << std::setw(14) << "size"
              << std::setw(10) << "percent"
              << "  detail" << std::endl;

    for (uint64_t i = 0; i < parsed.segments.size(); ++i) {
        const auto& segment = parsed.segments[static_cast<size_t>(i)];
        std::cout << std::left << std::setw(4) << i << std::setw(26) << segment.name
                  << std::setw(18) << segment.kind << std::right << std::setw(14) << segment.offset
                  << std::setw(14) << segment.size << std::setw(10)
                  << format_percent(segment.size, parsed.file_size) << "  " << segment.detail
                  << std::endl;
    }
}

std::string
base_block_name(const std::string& name) {
    constexpr std::string_view header_suffix = "_header";
    constexpr std::string_view payload_suffix = "_payload";
    if (name.size() > header_suffix.size() &&
        name.compare(name.size() - header_suffix.size(), header_suffix.size(), header_suffix) ==
            0) {
        return name.substr(0, name.size() - header_suffix.size());
    }
    if (name.size() > payload_suffix.size() &&
        name.compare(name.size() - payload_suffix.size(), payload_suffix.size(), payload_suffix) ==
            0) {
        return name.substr(0, name.size() - payload_suffix.size());
    }
    return name;
}

std::string
logical_block_detail(const LogicalBlock& block, uint64_t total_size) {
    std::ostringstream detail;
    detail << block.name << " - " << block.size << " bytes - "
           << format_percent(block.size, total_size);
    for (const auto* segment : block.segments) {
        detail << "\n" << segment->name << ": " << segment->size << " bytes @ " << segment->offset;
    }
    return detail.str();
}

std::string
logical_block_color(uint64_t index) {
    static const std::array<const char*, 10> colors{"#4c78a8",
                                                    "#f58518",
                                                    "#54a24b",
                                                    "#e45756",
                                                    "#72b7b2",
                                                    "#b279a2",
                                                    "#ff9da6",
                                                    "#9d755d",
                                                    "#bab0ac",
                                                    "#59a14f"};
    return colors[static_cast<size_t>(index % colors.size())];
}

std::vector<LogicalBlock>
build_logical_blocks(const ParsedIndex& parsed) {
    std::vector<LogicalBlock> blocks;
    LogicalBlock stream_header;
    stream_header.name = "stream_header";
    stream_header.kind = "header_group";
    stream_header.color = "#6f79d8";

    for (const auto& segment : parsed.segments) {
        if (segment.name == "section_end" || segment.name == "trailing_bytes") {
            LogicalBlock block;
            block.name = segment.name;
            block.kind = segment.kind;
            block.offset = segment.offset;
            block.size = segment.size;
            block.color = segment.color;
            block.segments.push_back(&segment);
            blocks.push_back(std::move(block));
            continue;
        }

        const bool is_tlv_part = segment.name.size() > std::string("_header").size() &&
                                 (segment.name.rfind("_header") == segment.name.size() - 7 ||
                                  segment.name.rfind("_payload") == segment.name.size() - 8);
        if (!is_tlv_part) {
            if (stream_header.segments.empty()) {
                stream_header.offset = segment.offset;
            }
            stream_header.size += segment.size;
            stream_header.segments.push_back(&segment);
            continue;
        }

        if (!stream_header.segments.empty()) {
            blocks.push_back(std::move(stream_header));
            stream_header = LogicalBlock{};
        }

        const auto block_name = base_block_name(segment.name);
        if (blocks.empty() || blocks.back().name != block_name) {
            LogicalBlock block;
            block.name = block_name;
            block.kind = "tlv_block";
            block.offset = segment.offset;
            block.color = logical_block_color(blocks.size());
            blocks.push_back(std::move(block));
        }
        auto& block = blocks.back();
        block.size += segment.size;
        block.segments.push_back(&segment);
        if (segment.kind == "block_header" && block.color.empty()) {
            block.color = segment.color;
        }
    }

    if (!stream_header.segments.empty()) {
        blocks.push_back(std::move(stream_header));
    }
    return blocks;
}

std::string
ansi_color_for_logical_block(const LogicalBlock& block, uint64_t index) {
    if (block.kind == "header_group") {
        return "\033[45m";
    }
    if (block.kind == "trailing") {
        return "\033[41m";
    }
    if (block.name == "section_end") {
        return "\033[43m";
    }
    static const std::array<const char*, 6> colors{
        "\033[44m", "\033[46m", "\033[42m", "\033[43m", "\033[45m", "\033[41m"};
    return colors[static_cast<size_t>(index % colors.size())];
}

void
print_terminal_logical_block_bars(const std::string& title,
                                  const std::vector<LogicalBlock>& blocks,
                                  uint64_t total_size) {
    std::cout << title << std::endl;
    constexpr uint64_t bar_width = 32;
    for (uint64_t i = 0; i < blocks.size(); ++i) {
        const auto& block = blocks[static_cast<size_t>(i)];
        const double ratio =
            total_size == 0 ? 0.0
                            : static_cast<double>(block.size) / static_cast<double>(total_size);
        uint64_t width = static_cast<uint64_t>(std::round(ratio * bar_width));
        if (block.size > 0 && width == 0) {
            width = 1;
        }
        width = std::min<uint64_t>(bar_width, width);
        const uint64_t padding = bar_width - width;
        std::cout << std::left << std::setw(18) << block.name << " ["
                  << ansi_color_for_logical_block(block, i)
                  << std::string(static_cast<size_t>(width), ' ') << "\033[0m"
                  << std::string(static_cast<size_t>(padding), ' ') << "] " << std::right
                  << std::setw(10) << format_bytes(block.size) << " " << std::setw(8)
                  << format_percent(block.size, total_size) << "  " << block.segments.size()
                  << " segment" << (block.segments.size() == 1 ? "" : "s") << std::endl;
    }
    std::cout << std::endl;
}

std::vector<VisualBlock>
build_exploded_block_view(const std::vector<LogicalBlock>& blocks, uint64_t total_size) {
    std::vector<VisualBlock> view;
    view.reserve(blocks.size());
    double reserved_width = 0.0;
    double remaining_source_ratio = 0.0;
    for (const auto& block : blocks) {
        const double ratio = total_size == 0 ? 0.0 : static_cast<double>(block.size) / total_size;
        const bool fold = ratio >= EXPLODED_LARGE_SEGMENT_THRESHOLD;
        if (fold) {
            reserved_width += EXPLODED_LARGE_SEGMENT_WIDTH;
        } else {
            remaining_source_ratio += ratio;
        }
        view.push_back(VisualBlock{&block, 0.0, fold});
    }

    const double remaining_width = std::max(0.0, 1.0 - reserved_width);
    for (auto& visual : view) {
        const double ratio =
            total_size == 0 ? 0.0 : static_cast<double>(visual.block->size) / total_size;
        if (visual.folded) {
            visual.width_ratio = EXPLODED_LARGE_SEGMENT_WIDTH;
        } else if (remaining_source_ratio > 0.0) {
            visual.width_ratio = ratio / remaining_source_ratio * remaining_width;
        }
    }
    return view;
}

std::string
json_type_name(const nlohmann::json& value) {
    if (value.is_object()) {
        return "object";
    }
    if (value.is_array()) {
        return "array";
    }
    if (value.is_string()) {
        return "string";
    }
    if (value.is_boolean()) {
        return "boolean";
    }
    if (value.is_number()) {
        return "number";
    }
    if (value.is_null()) {
        return "null";
    }
    return "unknown";
}

std::string
json_value_summary(const nlohmann::json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_object()) {
        return "object (" + std::to_string(value.size()) + " keys)";
    }
    if (value.is_array()) {
        return "array (" + std::to_string(value.size()) + " items)";
    }
    return value.dump();
}

bool
try_parse_json(const std::string& raw, nlohmann::json& output) {
    output = nlohmann::json::parse(raw, nullptr, false);
    return !output.is_discarded();
}

std::string
pretty_json_or_raw(const std::string& raw) {
    nlohmann::json parsed;
    if (try_parse_json(raw, parsed)) {
        return parsed.dump(2);
    }
    return raw;
}

void
write_json_table(std::ofstream& output, const nlohmann::json& object) {
    output << "<table><thead><tr><th>Key</th><th>Type</th><th>Value</th></tr></thead><tbody>\n";
    if (object.is_object()) {
        for (const auto& item : object.items()) {
            output << "<tr><td>" << escape_html(item.key()) << "</td><td>"
                   << escape_html(json_type_name(item.value())) << "</td><td>"
                   << escape_html(json_value_summary(item.value())) << "</td></tr>\n";
        }
    }
    output << "</tbody></table>\n";
}

void
write_metadata_overview(std::ofstream& output, const nlohmann::json& metadata) {
    output << "<div class=\"meta-grid\">\n";
    const std::array<std::string, 4> keys{"index_name", "format", "_update_time", "_empty"};
    for (const auto& key : keys) {
        if (!metadata.contains(key)) {
            continue;
        }
        output << "<div class=\"meta-card\"><div class=\"meta-label\">" << escape_html(key)
               << "</div><div class=\"meta-value\">"
               << escape_html(json_value_summary(metadata.at(key))) << "</div></div>\n";
    }
    output << "</div>\n";
}

void
write_block_manifest(std::ofstream& output, const nlohmann::json& metadata) {
    if (!metadata.contains("block_manifest") || !metadata.at("block_manifest").is_array()) {
        return;
    }
    output << "<h3>Block Manifest</h3>\n";
    output << "<table class=\"manifest-table\"><colgroup><col class=\"manifest-name\"><col "
              "class=\"manifest-center\"><col class=\"manifest-center\"><col "
              "class=\"manifest-critical\"></colgroup><thead><tr><th>Name</th><th "
              "class=\"manifest-center\">Tag</th><th class=\"manifest-center\">Version</th><th "
              "class=\"manifest-critical\">Critical</th></tr>"
              "</thead><tbody>\n";
    for (const auto& block : metadata.at("block_manifest")) {
        output << "<tr><td>" << escape_html(json_value_summary(block.value("name", "")))
               << "</td><td class=\"manifest-center\">" << block.value("tag", 0)
               << "</td><td class=\"manifest-center\">" << block.value("version", 0)
               << "</td><td class=\"manifest-critical\">"
               << (block.value("critical", false) ? "<span class=\"badge yes\">yes</span>"
                                                  : "<span class=\"badge no\">no</span>")
               << "</td></tr>\n";
    }
    output << "</tbody></table>\n";
}

void
write_json_block(std::ofstream& output,
                 const std::string& title,
                 const std::string& content,
                 bool enable_copy) {
    output << R"(<div class="json-block"><div class="json-title"><h3>)" << escape_html(title)
           << "</h3>";
    if (enable_copy) {
        output << R"HTML(<button type="button" class="copy-json" )HTML"
               << R"HTML(onclick="copyJsonBlock(this)">Copy</button>)HTML";
    }
    output << R"(</div><pre class="json-pre">)" << escape_html(content) << "</pre></div>\n";
}

void
write_pretty_json_section(std::ofstream& output,
                          const std::string& title,
                          const std::string& raw,
                          bool enable_copy = false) {
    if (raw.empty()) {
        return;
    }
    write_json_block(output, title, pretty_json_or_raw(raw), enable_copy);
}

void
write_metadata_panel(std::ofstream& output, const ParsedIndex& parsed) {
    nlohmann::json metadata;
    if (!try_parse_json(parsed.metadata_json, metadata) || !metadata.is_object()) {
        output
            << "<section class=\"panel metadata-panel\"><h2>Metadata</h2><pre class=\"json-pre\">"
            << escape_html(parsed.metadata_json) << "</pre></section>\n";
        return;
    }

    output << "<section class=\"panel metadata-panel\"><h2>Metadata</h2>\n";
    write_metadata_overview(output, metadata);
    write_block_manifest(output, metadata);

    if (metadata.contains("basic_info") && metadata.at("basic_info").is_object()) {
        output << "<h3>Basic Info</h3>\n";
        write_json_table(output, metadata.at("basic_info"));
        const auto& basic_info = metadata.at("basic_info");
        if (basic_info.contains("index_param")) {
            if (basic_info.at("index_param").is_string()) {
                write_pretty_json_section(output,
                                          "Basic Info / Index Param",
                                          basic_info.at("index_param").get<std::string>(),
                                          true);
            } else {
                write_json_block(
                    output, "Basic Info / Index Param", basic_info.at("index_param").dump(2), true);
            }
        }
    }

    if (metadata.contains("build_param_snapshot")) {
        if (metadata.at("build_param_snapshot").is_string()) {
            write_pretty_json_section(output,
                                      "Build Param Snapshot",
                                      metadata.at("build_param_snapshot").get<std::string>());
        } else {
            output << "<h3>Build Param Snapshot</h3><pre class=\"json-pre\">"
                   << escape_html(metadata.at("build_param_snapshot").dump(2)) << "</pre>\n";
        }
    }

    write_json_block(output, "Full Metadata", metadata.dump(2), true);
    output << "</section>\n";
}

void
write_html(const ParsedIndex& parsed, const std::string& output_path) {
    std::ofstream output(output_path, std::ios::binary);
    if (!output.is_open()) {
        throw std::runtime_error("failed to open html output path: " + output_path);
    }

    const auto logical_blocks = build_logical_blocks(parsed);
    const auto exploded_blocks = build_exploded_block_view(logical_blocks, parsed.file_size);

    output
        << "<!doctype html>\n<html><head><meta charset=\"utf-8\">\n"
        << "<title>VSAG streaming index layout</title>\n"
        << "<style>\n"
        << "body{font-family:Inter,Arial,sans-serif;margin:32px;background:#f7f8fa;color:#17202a;}"
        << "h1{font-size:24px;margin:0 0 8px;}h2{font-size:18px;margin:28px 0 8px;}"
        << ".muted{color:#5f6b7a}.note{color:#5f6b7a;font-size:13px;line-height:1.5}"
        << ".summary{display:flex;gap:18px;flex-wrap:wrap;margin:12px 0 18px}.summary "
           "span{background:#fff;border:1px solid #d7dde5;padding:8px 10px;font-size:13px}"
        << ".panel{background:#fff;border:1px solid #d7dde5;padding:16px;margin:18px 0;}"
        << ".bar{display:flex;align-items:stretch;width:100%;height:84px;border:1px solid "
           "#cfd6df;background:#fff;}"
        << ".block{position:relative;min-width:12px;max-width:52%;border-right:1px solid "
           "rgba(255,255,255,.85);box-sizing:border-box;}"
        << ".block.folded{background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.55) "
           "0,rgba(255,255,255,.55) 8px,rgba(0,0,0,.10) 8px,rgba(0,0,0,.10) 16px);}"
        << ".block:hover::after{content:attr(data-tip);position:absolute;left:0;top:calc(100% + "
           "6px);white-space:pre;background:#111827;color:#fff;padding:7px "
           "9px;border-radius:4px;font-size:12px;z-index:10;max-width:680px;}"
        << ".legend{display:flex;gap:12px;flex-wrap:wrap;margin:12px 0 0}.legend "
           "span{display:inline-flex;align-items:center;gap:6px;font-size:12px}.sw{width:12px;"
           "height:12px;display:inline-block;border:1px solid #c8ced8}"
        << "table{border-collapse:collapse;width:100%;background:#fff;margin-top:16px}th,td{border-"
           "bottom:1px solid #e5e7eb;padding:8px "
           "10px;text-align:left;font-size:13px}th{background:#edf1f5}td.num{text-align:right;font-"
           "variant-numeric:tabular-nums}"
        << "pre{background:#111827;color:#e5e7eb;padding:16px;overflow:auto;max-height:360px;}"
        << "h3{font-size:15px;margin:18px 0 8px}.metadata-panel{margin-top:24px}"
        << ".meta-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:"
           "10px;margin:12px 0}"
        << ".meta-card{border:1px solid #d7dde5;background:#f8fafc;padding:10px}"
        << ".meta-label{font-size:11px;text-transform:uppercase;color:#5f6b7a;margin-bottom:4px}"
        << ".meta-value{font-size:13px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
           "word-break:break-word}"
        << ".manifest-table{width:100%;table-layout:fixed}"
        << ".manifest-table th,.manifest-table td{padding:8px 12px;white-space:nowrap}"
        << ".manifest-table td:first-child{overflow:hidden;text-overflow:ellipsis}"
        << ".manifest-name{width:auto}.manifest-center{width:96px;text-align:center}.manifest-"
           "critical{width:120px;text-align:center}"
        << ".badge{display:inline-block;min-width:34px;text-align:center;padding:2px "
           "7px;border-radius:999px;font-size:11px;font-weight:600}"
        << ".badge.yes{background:#dcfce7;color:#166534}.badge.no{background:#fee2e2;color:#991b1b}"
        << ".json-block{margin-top:14px}.json-title{display:flex;align-items:center;gap:10px;"
           "justify-content:space-between}.json-title h3{margin:0}.copy-json{border:1px solid "
           "#cbd5e1;background:#f8fafc;color:#17202a;padding:4px 10px;border-radius:4px;"
           "font-size:12px;cursor:pointer}.copy-json:hover{background:#edf1f5}.copy-json.copied{"
           "border-color:#86efac;background:#dcfce7;color:#166534}.copy-json.failed{border-color:"
           "#fecaca;background:#fee2e2;color:#991b1b}"
        << ".json-pre{font-size:12px;line-height:1.45;border-radius:4px;max-height:520px;margin-"
           "top:8px}"
        << "</style></head><body>\n";

    output << "<h1>VSAG streaming index layout</h1>\n";
    output << "<div class=\"muted\">Grouped exploded layout for streaming serialization "
              "files.</div>\n";
    output << "<div class=\"summary\"><span>Format " << parsed.format_major << "."
           << parsed.format_minor << "</span><span>Total " << parsed.file_size << " bytes ("
           << format_bytes(parsed.file_size) << ")</span><span>Logical blocks "
           << logical_blocks.size() << "</span></div>\n";

    auto write_tip = [&](const LogicalBlock& block) {
        return escape_html(logical_block_detail(block, parsed.file_size));
    };

    output
        << "<section class=\"panel\"><h2>Grouped Exploded Layout</h2>\n"
        << "<div class=\"note\">Related small segments are merged into logical blocks. "
        << "Very large blocks are visually folded with diagonal stripes. Hover a block to inspect "
        << "its internal segments; tables below keep exact byte sizes and real percentages.</div>\n"
        << "<div class=\"bar\">\n";
    for (const auto& visual : exploded_blocks) {
        const auto& block = *visual.block;
        output << "<div class=\"block" << (visual.folded ? " folded" : "")
               << "\" style=\"flex:" << std::fixed << std::setprecision(8) << visual.width_ratio
               << " 1 0;background-color:" << block.color << "\" data-tip=\"" << write_tip(block)
               << "\"></div>\n";
    }
    output << "</div><div class=\"legend\">\n";
    for (const auto& block : logical_blocks) {
        output << "<span><i class=\"sw\" style=\"background-color:" << block.color << "\"></i>"
               << escape_html(block.name) << "</span>\n";
    }
    output << "</div></section>\n";

    output << "<h2>Logical Block Summary</h2>\n";
    output << "<table><thead><tr><th>Name</th><th>Offset</th><th>Size</th><th>Percent</"
              "th><th>Segments</th></tr></thead><tbody>\n";
    for (const auto& block : logical_blocks) {
        output << "<tr><td>" << escape_html(block.name) << "</td><td class=\"num\">" << block.offset
               << "</td><td class=\"num\">" << block.size << "</td><td class=\"num\">"
               << format_percent(block.size, parsed.file_size) << "</td><td>";
        for (uint64_t i = 0; i < block.segments.size(); ++i) {
            if (i > 0) {
                output << ", ";
            }
            output << escape_html(block.segments[static_cast<size_t>(i)]->name);
        }
        output << "</td></tr>\n";
    }
    output << "</tbody></table>\n";

    output << "<h2>Exact Segment Table</h2>\n";
    output << "<table><thead><tr><th>#</th><th>Name</th><th>Kind</th><th>Offset</th>"
           << "<th>Size</th><th>Percent</th><th>Detail</th></tr></thead><tbody>\n";
    for (uint64_t i = 0; i < parsed.segments.size(); ++i) {
        const auto& segment = parsed.segments[static_cast<size_t>(i)];
        output << "<tr><td>" << i << "</td><td>" << escape_html(segment.name) << "</td><td>"
               << escape_html(segment.kind) << "</td><td class=\"num\">" << segment.offset
               << "</td><td class=\"num\">" << segment.size << "</td><td class=\"num\">"
               << format_percent(segment.size, parsed.file_size) << "</td><td>"
               << escape_html(segment.detail) << "</td></tr>\n";
    }
    output << "</tbody></table>\n";
    write_metadata_panel(output, parsed);
    output << R"(<script>
function copyJsonBlock(button) {
  var block = button.closest('.json-block');
  var pre = block ? block.querySelector('pre') : null;
  if (!pre) {
    return;
  }
  var text = pre.textContent;
  var finish = function(ok) {
    button.classList.remove('copied', 'failed');
    button.classList.add(ok ? 'copied' : 'failed');
    button.textContent = ok ? 'Copied' : 'Failed';
    window.setTimeout(function() {
      button.classList.remove('copied', 'failed');
      button.textContent = 'Copy';
    }, 1200);
  };
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(
      function() { finish(true); },
      function() { fallbackCopy(text, finish); });
  } else {
    fallbackCopy(text, finish);
  }
}
function fallbackCopy(text, finish) {
  var area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.left = '-9999px';
  document.body.appendChild(area);
  area.select();
  var ok = false;
  try {
    ok = document.execCommand('copy');
  } catch (err) {
    ok = false;
  }
  document.body.removeChild(area);
  finish(ok);
}
</script></body></html>
)";
}

void
parse_args(argparse::ArgumentParser& parser, int argc, char** argv) {
    parser.add_argument<std::string>("--index_path", "-i")
        .required()
        .help("Streaming serialized index file path");
    parser.add_argument<std::string>("--html")
        .default_value(std::string())
        .help("Optional output path for a self-contained HTML visualization");

    try {
        parser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << parser;
        std::exit(1);
    }
}

}  // namespace

int
main(int argc, char** argv) {
    argparse::ArgumentParser parser("visualize_index");
    parse_args(parser, argc, argv);

    try {
        const auto index_path = parser.get<std::string>("--index_path");
        const auto html_path = parser.get<std::string>("--html");
        auto parsed = parse_streaming_index(index_path);
        print_summary(parsed);
        if (!html_path.empty()) {
            write_html(parsed, html_path);
            std::cout << std::endl << "HTML visualization written to: " << html_path << std::endl;
        }
    } catch (const std::exception& err) {
        std::cerr << "visualize_index failed: " << err.what() << std::endl;
        return 1;
    }
    return 0;
}
