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

#include "chunked_manifest.h"

#include <fmt/format.h>

#include <algorithm>
#include <nlohmann/json.hpp>
#include <unordered_set>

#include "vsag_exception.h"

namespace vsag {

JsonType
ChunkedManifest::ToJson() const {
    nlohmann::json json;
    json["version"] = VERSION;
    json["codec"] = codec_;
    json["chunk_size"] = chunk_size_;
    json["components"] = nlohmann::json::array();
    for (const auto& comp : components_) {
        nlohmann::json comp_json;
        comp_json["name"] = comp.name;
        if (comp.granularity == ComponentGranularity::Byte) {
            comp_json["type"] = "chunked";
            comp_json["head"] = {{"offset", comp.head_offset}, {"size", comp.head_size}};
            comp_json["io_size"] = comp.io_size;
            auto chunks_json = nlohmann::json::array();
            for (const auto& chunk : comp.chunks) {
                chunks_json.push_back({{"offset", chunk.offset}, {"csize", chunk.compressed_size}});
            }
            comp_json["chunks"] = std::move(chunks_json);
            comp_json["tail"] = {{"offset", comp.tail_offset}, {"size", comp.tail_size}};
        } else if (comp.granularity == ComponentGranularity::Whole) {
            comp_json["type"] = "whole";
            comp_json["offset"] = comp.offset;
            comp_json["csize"] = comp.compressed_size;
            comp_json["lsize"] = comp.logical_size;
        } else {
            // a granularity missing here would otherwise be written out
            // under the wrong tag and corrupt the file silently
            throw VsagException(ErrorType::INTERNAL_ERROR,
                                "unsupported component granularity in layout serialization");
        }
        json["components"].push_back(std::move(comp_json));
    }
    JsonType result;
    *result.GetInnerJson() = std::move(json);
    return result;
}

ChunkedManifest
ChunkedManifest::FromJson(const JsonType& json_wrapper) {
    const auto& json = *json_wrapper.GetInnerJson();
    if (not json.contains("version") or not json["version"].is_number_integer() or
        json["version"].get<int64_t>() != VERSION) {
        throw VsagException(ErrorType::INVALID_BINARY, "unsupported chunked_layout version");
    }
    // a missing key or a type mismatch below throws a raw json exception;
    // map it to the same error class as any other malformed binary input
    try {
        ChunkedManifest manifest;
        manifest.codec_ = json.at("codec").get<std::string>();
        manifest.chunk_size_ = json.at("chunk_size").get<uint64_t>();
        for (const auto& comp_json : json.at("components")) {
            ComponentManifestEntry comp;
            comp.name = comp_json.at("name").get<std::string>();
            const auto type = comp_json.at("type").get<std::string>();
            if (type == "chunked") {
                comp.granularity = ComponentGranularity::Byte;
                comp.head_offset = comp_json.at("head").at("offset").get<uint64_t>();
                comp.head_size = comp_json.at("head").at("size").get<uint64_t>();
                comp.io_size = comp_json.at("io_size").get<uint64_t>();
                for (const auto& chunk_json : comp_json.at("chunks")) {
                    ChunkRecord chunk;
                    chunk.offset = chunk_json.at("offset").get<uint64_t>();
                    chunk.compressed_size = chunk_json.at("csize").get<uint64_t>();
                    comp.chunks.push_back(chunk);
                }
                comp.tail_offset = comp_json.at("tail").at("offset").get<uint64_t>();
                comp.tail_size = comp_json.at("tail").at("size").get<uint64_t>();
            } else if (type == "whole") {
                comp.granularity = ComponentGranularity::Whole;
                comp.offset = comp_json.at("offset").get<uint64_t>();
                comp.compressed_size = comp_json.at("csize").get<uint64_t>();
                comp.logical_size = comp_json.at("lsize").get<uint64_t>();
            } else {
                // granularities newer than this reader must fail here, not
                // get silently parsed under the wrong branch
                throw VsagException(ErrorType::INVALID_BINARY,
                                    "unsupported component type: " + type);
            }
            manifest.components_.push_back(std::move(comp));
        }
        return manifest;
    } catch (const VsagException&) {
        throw;
    } catch (const std::exception& e) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            std::string("malformed chunked_layout: ") + e.what());
    }
}

void
ChunkedManifest::Validate(uint64_t body_end) const {
    // collected non-empty physical extents; checked for mutual overlap below
    std::vector<std::pair<uint64_t, uint64_t>> frames;
    frames.reserve(components_.size());

    auto add_frame = [&](uint64_t offset, uint64_t size, const std::string& label) {
        // a zero-size frame occupies no bytes, but a bogus offset past the
        // body is still rejected
        if (offset > body_end) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format(
                    "chunked layout {} offset {} exceeds body end {}", label, offset, body_end));
        }
        if (size == 0) {
            return;
        }
        // offset + size must not overflow and must stay within the body;
        // check size first so body_end - size cannot underflow
        if (size > body_end or offset > body_end - size) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                fmt::format("chunked layout {} extent [{}, {}+{}) exceeds body "
                                            "end {}",
                                            label,
                                            offset,
                                            offset,
                                            size,
                                            body_end));
        }
        frames.emplace_back(offset, offset + size);
    };

    const bool uncompressed = (codec_ == "none");
    std::unordered_set<std::string> seen_names;

    for (const auto& comp : components_) {
        if (not seen_names.insert(comp.name).second) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format("chunked layout has duplicate component {}", comp.name));
        }

        if (comp.granularity == ComponentGranularity::Whole) {
            add_frame(
                comp.offset, comp.compressed_size, fmt::format("component {} frame", comp.name));
            if (uncompressed and comp.compressed_size != comp.logical_size) {
                throw VsagException(
                    ErrorType::INVALID_BINARY,
                    fmt::format("uncompressed component {} physical size {} != logical size {}",
                                comp.name,
                                comp.compressed_size,
                                comp.logical_size));
            }
        } else {
            add_frame(
                comp.head_offset, comp.head_size, fmt::format("component {} head", comp.name));
            add_frame(
                comp.tail_offset, comp.tail_size, fmt::format("component {} tail", comp.name));

            // must stay ahead of the round-up guard below: it is what keeps a
            // zeroed chunk_size from carrying an arbitrary io_size past the
            // overflow check, which only runs when chunk_size_ > 0
            if (comp.io_size > 0 and chunk_size_ == 0) {
                throw VsagException(ErrorType::INVALID_BINARY,
                                    fmt::format("component {} has io size {} but zero chunk size",
                                                comp.name,
                                                comp.io_size));
            }
            // the round-up below would wrap for an io size close to the
            // unsigned maximum, which would yield a bogus expected count
            if (chunk_size_ > 0 and
                comp.io_size > std::numeric_limits<uint64_t>::max() - (chunk_size_ - 1)) {
                throw VsagException(
                    ErrorType::INVALID_BINARY,
                    fmt::format("component {} io size {} is too large", comp.name, comp.io_size));
            }
            const uint64_t expected_chunks =
                chunk_size_ == 0 ? 0 : (comp.io_size + chunk_size_ - 1) / chunk_size_;
            if (comp.chunks.size() != expected_chunks) {
                throw VsagException(
                    ErrorType::INVALID_BINARY,
                    fmt::format("component {} has {} chunks but io size {} needs {}",
                                comp.name,
                                comp.chunks.size(),
                                comp.io_size,
                                expected_chunks));
            }
            uint64_t logical_offset = 0;
            for (size_t i = 0; i < comp.chunks.size(); ++i) {
                const auto& chunk = comp.chunks[i];
                add_frame(chunk.offset,
                          chunk.compressed_size,
                          fmt::format("component {} chunk {}", comp.name, i));
                if (uncompressed) {
                    const uint64_t logical = std::min(chunk_size_, comp.io_size - logical_offset);
                    if (chunk.compressed_size != logical) {
                        throw VsagException(
                            ErrorType::INVALID_BINARY,
                            fmt::format("uncompressed component {} chunk {} physical size {} != "
                                        "logical size {}",
                                        comp.name,
                                        i,
                                        chunk.compressed_size,
                                        logical));
                    }
                    logical_offset += logical;
                }
            }
        }
    }

    // the writer advances its physical cursor by exactly the bytes it emits and
    // never pads, so the recorded frames must tile [0, body_end) exactly: a gap
    // would mean the footer omits or misplaces a frame, which can hide a
    // truncated or corrupted body
    std::sort(frames.begin(), frames.end());
    for (size_t i = 0; i < frames.size(); ++i) {
        const uint64_t expected_start = (i == 0) ? 0 : frames[i - 1].second;
        if (frames[i].first == expected_start) {
            continue;
        }
        if (i > 0 and frames[i].first < frames[i - 1].second) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format("chunked layout has overlapping frames: [{}, {}) and [{}, {})",
                            frames[i - 1].first,
                            frames[i - 1].second,
                            frames[i].first,
                            frames[i].second));
        }
        throw VsagException(
            ErrorType::INVALID_BINARY,
            fmt::format(
                "chunked layout leaves [{}, {}) unaccounted for", expected_start, frames[i].first));
    }
    if (frames.empty()) {
        if (body_end != 0) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format("chunked layout records no frames but the body ends at {}", body_end));
        }
    } else if (frames.back().second != body_end) {
        throw VsagException(ErrorType::INVALID_BINARY,
                            fmt::format("chunked layout ends at {} but the body ends at {}",
                                        frames.back().second,
                                        body_end));
    }
}

const ComponentManifestEntry*
ChunkedManifest::FindComponent(const std::string& name) const {
    for (const auto& comp : components_) {
        if (comp.name == name) {
            return &comp;
        }
    }
    return nullptr;
}

}  // namespace vsag
