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

#include <fmt/format.h>

#include <algorithm>
#include <cstring>
#include <functional>
#include <mutex>
#include <optional>
#include <vector>

#include "datacell/clique_datacell.h"
#include "hgraph.h"
#include "hgraph_component_names.h"  // IWYU pragma: keep
#include "impl/thread_pool/safe_thread_pool.h"
#include "storage/chunked_manifest.h"
#include "storage/parallel_deserialize_utils.h"
#include "storage/serialization.h"
#include "storage/stream_reader.h"
#include "vsag/deserialize_reader.h"
#include "vsag/serialize_writer.h"
#include "vsag/thread_pool.h"

namespace vsag {

namespace {

// how much of one io extent a single probe-path fill task moves. The probe
// path has no recorded layout, so this is pure work partitioning and is
// deliberately independent of DEFAULT_SERIALIZE_CHUNK_SIZE: changing the
// serialization default must not change how an existing file is loaded.
constexpr uint64_t PROBE_FILL_SPLIT_SIZE = 128ULL * 1024 * 1024;

// components the chunked path can restore frame by frame; every other known
// component is always written as a single whole frame (serialize_whole in
// hgraph_serialize.cpp), so a Byte granularity for them means the layout was
// tampered with
bool
supports_chunked_granularity(const std::string& name) {
    return name == COMPONENT_BASE_CODES or name == COMPONENT_BOTTOM_GRAPH or
           name == COMPONENT_PRECISE_CODES or name == COMPONENT_RAW_VECTOR;
}

// adding a component means touching three places here: is_known_component (so
// the layout pre-validation accepts the name), deserialize_whole_component (the
// whole dispatch), and, only if it is chunk-capable,
// supports_chunked_granularity plus resolve_chunked_target. Forgetting any of
// them fails loudly on load rather than corrupting the index: an unknown name
// is rejected by the pre-validation, and both dispatches end in a throw.
bool
is_known_component(const std::string& name) {
    return supports_chunked_granularity(name) or name == COMPONENT_LABEL_TABLE or
           name == COMPONENT_CODE_SLOT_MAP or name == COMPONENT_ROUTE_GRAPHS or
           name == COMPONENT_EXTRA_INFOS or name == COMPONENT_ATTR_FILTER or
           name == COMPONENT_MCI_CLIQUES or name == COMPONENT_CONJUGATE_GRAPH;
}

// components whose Deserialize seeks inside its own payload rather than reading
// it front to back: the conjugate graph declares its total size in a header,
// jumps to a trailing footer to check its magic and version before trusting the
// body, then rewinds to stream that body, and it stops FOOTER_SIZE bytes short
// of the end. Such a component needs a seekable reader over its frame, and its
// own cursor says nothing about how much of the frame is real.
//
// Nothing about the data itself requires this: the map is written front to back
// and rebuilt with insertions in any order. It is the footer-last layout read
// footer-first that does, so moving that validation after the body would make
// the component streamable and let this whole case go away - at the cost of
// parsing attacker-controlled counts before the magic check.
bool
requires_seekable_payload(const std::string& name) {
    return name == COMPONENT_CONJUGATE_GRAPH;
}

// a known component whose enabling option is off in this index is a
// configuration mismatch, not an unknown name; shared by the chunked and the
// whole dispatch so the two cannot drift
void
require_component_enabled(const std::string& name, bool enabled) {
    if (not enabled) {
        throw VsagException(ErrorType::INVALID_ARGUMENT,
                            fmt::format("file contains component {} but the index configuration "
                                        "does not enable it",
                                        name));
    }
}

// three-part hooks of one chunked component, type-erased over
// FlattenInterface / GraphInterface
struct ChunkedTarget {
    std::function<uint64_t(StreamReader&)> reserve_io;
    WriteRawFunc write_raw;
    std::function<void(StreamReader&)> deserialize_tail;
};

template <typename ComponentPtr>
ChunkedTarget
make_chunked_target(const ComponentPtr& component) {
    return ChunkedTarget{
        [component](StreamReader& reader) { return component->ReserveIO(reader); },
        [component](const uint8_t* data, uint64_t size, uint64_t offset) {
            component->WriteRaw(data, size, offset);
        },
        [component](StreamReader& reader) { component->DeserializeTail(reader); },
    };
}

// targets entries exist only for Byte components. Every caller has already
// checked the granularity, so a missing entry is a slip in this file rather
// than bad input; say so instead of letting an empty std::function surface as
// bad_function_call somewhere further down.
ChunkedTarget&
require_target(std::optional<ChunkedTarget>& target, const std::string& name) {
    if (not target.has_value()) {
        throw VsagException(ErrorType::INTERNAL_ERROR,
                            fmt::format("no chunked target resolved for component {}", name));
    }
    return *target;
}

}  // namespace

void
HGraph::ParallelDeserialize(DeserializeReader& reader) {
    // Engine-created indexes carry the Resource-managed pool in thread_pool_.
    // Keep an owned fallback only for direct construction without a Resource;
    // it must outlive every task dispatched below.
    auto pool_owner = this->thread_pool_;
    if (pool_owner == nullptr) {
        pool_owner = SafeThreadPool::FactoryDefaultThreadPool();
    }
    ThreadPool& pool = *pool_owner;

    auto read_func = [&reader](uint64_t offset, uint64_t len, void* dest) {
        reader.Read(offset, len, dest);
    };
    ReadFuncStreamReader footer_reader(read_func, 0, reader.Size());
    auto footer = Footer::Parse(footer_reader);
    if (footer == nullptr) {
        throw VsagException(ErrorType::INVALID_BINARY, "failed to parse index footer");
    }
    auto metadata = footer->GetMetadata();

    const auto serialized_total_count = this->apply_footer_metadata(metadata);

    auto layout_json = metadata->Get(CHUNKED_LAYOUT_KEY);
    if (layout_json.IsObject()) {
        auto chunked_manifest = ChunkedManifest::FromJson(layout_json);
        // FromJson only checks JSON shape; validate the physical layout
        // against the body extent before dispatching concurrent writers, so
        // a tampered footer cannot alias frames or escape the body bounds
        const uint64_t physical_body = reader.Size() - footer->Length();
        chunked_manifest.Validate(physical_body);
        this->parallel_deserialize_manifest(reader, pool, chunked_manifest);
    } else {
        // no recorded layout: probe an uncompressed sequential body
        const uint64_t physical_body = reader.Size() - footer->Length();
        this->parallel_deserialize_probe(reader, pool, physical_body);
    }

    this->validate_and_publish_dedup_state(serialized_total_count);
    this->publish_physical_code_capacity();
    this->initialize_deserialized_runtime_state();
    if (not this->using_dedup_storage()) {
        this->total_count_ = this->basic_flatten_codes_->TotalCount();
    }
    if (this->raw_vector_ != nullptr) {
        this->has_raw_vector_ = true;
    }
    this->finish_deserialize();
}

void
HGraph::parallel_deserialize_manifest(DeserializeReader& reader,
                                      ThreadPool& pool,
                                      const ChunkedManifest& chunked_manifest) {
    // Validate() proves the physical invariants but does not know which
    // components can be restored frame by frame; reject a tampered
    // granularity here, before any extent is reserved or task dispatched
    for (const auto& comp : chunked_manifest.components_) {
        if (not is_known_component(comp.name)) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                fmt::format("unknown component in chunked layout: {}", comp.name));
        }
        if (comp.granularity == ComponentGranularity::Byte and
            not supports_chunked_granularity(comp.name)) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format("component {} cannot be deserialized in chunked form", comp.name));
        }
    }

    const bool compressed = (chunked_manifest.codec_ != "none");
    auto read_func = [&reader](uint64_t offset, uint64_t len, void* dest) {
        reader.Read(offset, len, dest);
    };

    auto resolve_chunked_target = [this](const std::string& name) -> ChunkedTarget {
        if (name == COMPONENT_BASE_CODES) {
            return make_chunked_target(this->basic_flatten_codes_);
        }
        if (name == COMPONENT_BOTTOM_GRAPH) {
            return make_chunked_target(this->bottom_graph_);
        }
        if (name == COMPONENT_PRECISE_CODES) {
            require_component_enabled(name, this->high_precise_codes_ != nullptr);
            return make_chunked_target(this->high_precise_codes_);
        }
        if (name == COMPONENT_RAW_VECTOR) {
            require_component_enabled(name, this->raw_vector_ != nullptr);
            return make_chunked_target(this->raw_vector_);
        }
        // unreachable under the current pre-validation: parallel_deserialize_manifest
        // already rejects an unknown name or a non-chunkable component with Byte
        // granularity, and the caller only invokes this for Byte components, so
        // every name reaching here is handled by a branch above. Kept as
        // defense-in-depth: if that guard is ever weakened, or a new
        // chunk-capable component is added without a branch here, fail loudly
        // instead of falling through to undefined behaviour.
        throw VsagException(
            ErrorType::INVALID_BINARY,
            fmt::format("component {} cannot be deserialized in chunked form", name));
    };

    // pass 1 (this thread): read the heads and pre-allocate the io extents,
    // so the chunk tasks can fill disjoint ranges without reallocation
    //
    // Staying on the calling thread is a deliberate simplification, not an
    // inherent constraint: the layout already records every head offset and io
    // size, so each head read plus ReserveIO is independent and could be
    // dispatched to the pool. At most four components are chunked and their
    // heads are a few hundred bytes each, so even with ReserveIO reserving
    // blocks through fallocate for mmap-backed io the one-off cost stays well
    // below the scheduling overhead, and running in order keeps the in-place
    // fallback single-writer by construction. If the component count grows or
    // ReserveIO becomes expensive, this phase can be parallelised at the cost
    // of routing failures through a TaskBatch. The probe path differs: there
    // the boundaries are only discovered while reading, so its sequential phase
    // is unavoidable.
    // only Byte components get an entry; std::nullopt says "no chunked hooks
    // here" at the type level, so a future loop that forgets the granularity
    // check trips on the optional instead of on an empty std::function.
    // Read through require_target so that slip names the component instead of
    // surfacing as bad_function_call
    std::vector<std::optional<ChunkedTarget>> targets(chunked_manifest.components_.size());
    // components consumed in place during pass 1 (io without the parallel
    // hooks, e.g. reader_io); they get no chunk tasks and no tail pass
    std::vector<bool> consumed(chunked_manifest.components_.size(), false);
    for (size_t i = 0; i < chunked_manifest.components_.size(); ++i) {
        const auto& comp = chunked_manifest.components_[i];
        if (comp.granularity != ComponentGranularity::Byte) {
            continue;
        }
        // the chunk list is already proven to cover the whole io extent by
        // ChunkedManifest::Validate(), which runs before any extent is reserved
        targets[i] = resolve_chunked_target(comp.name);
        ReadFuncStreamReader head_reader(
            read_func, comp.head_offset, comp.head_offset + comp.head_size);
        try {
            const auto io_size = require_target(targets[i], comp.name).reserve_io(head_reader);
            if (io_size != comp.io_size) {
                throw VsagException(ErrorType::INVALID_BINARY,
                                    fmt::format("component {} io size mismatch: {} != {}",
                                                comp.name,
                                                io_size,
                                                comp.io_size));
            }
            if (head_reader.GetCursor() != comp.head_offset + comp.head_size) {
                throw VsagException(
                    ErrorType::INVALID_BINARY,
                    fmt::format("component {} head does not end at its recorded size", comp.name));
            }
        } catch (const VsagException& e) {
            if (e.error_.type != ErrorType::UNSUPPORTED_INDEX_OPERATION) {
                throw;
            }
            // io without the hooks keeps its bytes out of memory (reader_io
            // seeks over the data body and reads it back on demand), which
            // only works when the on-disk bytes are the logical bytes
            if (compressed) {
                throw VsagException(
                    ErrorType::UNSUPPORTED_INDEX_OPERATION,
                    fmt::format("component {} io does not hold deserialized bytes; a "
                                "compressed chunked index requires an io that does",
                                comp.name));
            }
            // uncompressed chunks are plain contiguous bytes, so the whole
            // head + data + tail span equals the sequential form; consume it
            // in place exactly like the sequential Deserialize would
            ReadFuncStreamReader whole_reader(
                read_func, comp.head_offset, comp.tail_offset + comp.tail_size);
            this->deserialize_whole_component(comp.name, whole_reader);
            consumed[i] = true;
        }
    }

    // pass 2: one task per chunk / whole component
    size_t task_count = 0;
    for (size_t i = 0; i < chunked_manifest.components_.size(); ++i) {
        const auto& comp = chunked_manifest.components_[i];
        task_count += comp.granularity == ComponentGranularity::Byte
                          ? (consumed[i] ? 0 : comp.chunks.size())
                          : 1;
    }
    {
        // every task below captures by reference into this frame: reader and
        // pool are the caller's, targets / consumed / chunked_manifest live until the
        // closing brace. TaskBatch joins in Run() and again in its destructor,
        // so no task can outlive any of them.
        TaskBatch batch(pool, task_count);
        for (size_t i = 0; i < chunked_manifest.components_.size(); ++i) {
            const auto& comp = chunked_manifest.components_[i];
            if (comp.granularity == ComponentGranularity::Byte) {
                if (consumed[i]) {
                    continue;
                }
                const auto& write_raw = require_target(targets[i], comp.name).write_raw;
                uint64_t logical_offset = 0;
                for (const auto& chunk : comp.chunks) {
                    const auto logical =
                        std::min(chunked_manifest.chunk_size_, comp.io_size - logical_offset);
                    if (compressed) {
                        batch.Submit([&reader,
                                      &write_raw,
                                      offset = chunk.offset,
                                      csize = chunk.compressed_size,
                                      logical_offset,
                                      logical]() {
                            reader.ReadDecompressed(offset, csize, [&](std::istream& is) {
                                consume_chunk_stream(is, write_raw, logical_offset, logical);
                            });
                        });
                    } else {
                        batch.Submit([&reader,
                                      &write_raw,
                                      offset = chunk.offset,
                                      logical_offset,
                                      logical]() {
                            fill_extent_from_reader(
                                reader, write_raw, offset, logical_offset, logical);
                        });
                    }
                    logical_offset += logical;
                }
            } else {
                // whole tasks run concurrently and must only touch index
                // members no other component task touches; keep that
                // invariant when adding a component handler
                batch.Submit([this, &reader, &read_func, &comp, compressed]() {
                    if (requires_seekable_payload(comp.name)) {
                        this->deserialize_seekable_whole_component(reader, comp, compressed);
                        return;
                    }
                    if (compressed) {
                        reader.ReadDecompressed(
                            comp.offset, comp.compressed_size, [&](std::istream& is) {
                                ForwardStreamReader forward_reader(is);
                                BoundedForwardReader bounded(&forward_reader, comp.logical_size);
                                this->deserialize_whole_component(comp.name, bounded);
                                if (bounded.GetCursor() != comp.logical_size) {
                                    throw VsagException(
                                        ErrorType::INVALID_BINARY,
                                        fmt::format("component {} consumed {} bytes but logical "
                                                    "size is {}",
                                                    comp.name,
                                                    bounded.GetCursor(),
                                                    comp.logical_size));
                                }
                                // the bound stops at logical_size, so
                                // residual bytes only show up on the
                                // decompressed stream itself
                                if (is.peek() != std::istream::traits_type::eof()) {
                                    throw VsagException(
                                        ErrorType::INVALID_BINARY,
                                        fmt::format("component {} frame holds more than "
                                                    "its logical size {}",
                                                    comp.name,
                                                    comp.logical_size));
                                }
                            });
                    } else {
                        ReadFuncStreamReader plain_reader(
                            read_func, comp.offset, comp.offset + comp.logical_size);
                        this->deserialize_whole_component(comp.name, plain_reader);
                        if (plain_reader.GetCursor() != comp.offset + comp.logical_size) {
                            throw VsagException(
                                ErrorType::INVALID_BINARY,
                                fmt::format("component {} consumed up to {} but logical "
                                            "end is {}",
                                            comp.name,
                                            plain_reader.GetCursor(),
                                            comp.offset + comp.logical_size));
                        }
                    }
                });
            }
        }
        batch.Run();
    }

    // read the tails after all io data landed
    for (size_t i = 0; i < chunked_manifest.components_.size(); ++i) {
        const auto& comp = chunked_manifest.components_[i];
        if (comp.granularity != ComponentGranularity::Byte or consumed[i]) {
            continue;
        }
        ReadFuncStreamReader tail_reader(
            read_func, comp.tail_offset, comp.tail_offset + comp.tail_size);
        require_target(targets[i], comp.name).deserialize_tail(tail_reader);
        if (tail_reader.GetCursor() != comp.tail_offset + comp.tail_size) {
            throw VsagException(
                ErrorType::INVALID_BINARY,
                fmt::format("component {} tail does not end at its recorded size", comp.name));
        }
    }
}

void
HGraph::deserialize_seekable_whole_component(DeserializeReader& reader,
                                             const ComponentManifestEntry& comp,
                                             bool compressed) {
    // Whatever the source, the component gets a reader it can seek in and its
    // own cursor is not used as the consumption check: a component that seeks
    // does not leave the cursor at the end of its frame.
    if (not compressed) {
        // DeserializeReader::Read is positional, so an uncompressed frame is
        // already random access. Read straight through it, no staging copy.
        auto frame_read = [&reader, base = comp.offset, size = comp.logical_size](
                              uint64_t offset, uint64_t len, void* dest) {
            if (offset > size or len > size - offset) {
                throw VsagException(ErrorType::INVALID_BINARY,
                                    "seekable component reader exceeds its frame");
            }
            if (len > 0) {
                reader.Read(base + offset, len, dest);
            }
        };
        ReadFuncStreamReader frame_reader(frame_read, 0, comp.logical_size);
        this->deserialize_whole_component(comp.name, frame_reader);
        return;
    }

    // Compressed frames leave no choice: decompression only runs forward, so
    // reaching the trailing footer this component validates before its body
    // means materializing the frame. That costs the component's decompressed
    // size on this task's thread. Splitting such a component into separate
    // header / body / footer frames would let all three be streamed, at the
    // price of encoding its internal layout into this index format.
    std::vector<char> payload(comp.logical_size);
    {
        reader.ReadDecompressed(comp.offset, comp.compressed_size, [&](std::istream& is) {
            uint64_t filled = 0;
            while (filled < comp.logical_size) {
                is.read(payload.data() + filled,
                        static_cast<std::streamsize>(comp.logical_size - filled));
                const auto got = static_cast<uint64_t>(is.gcount());
                if (got == 0) {
                    throw VsagException(ErrorType::INVALID_BINARY,
                                        fmt::format("component {} frame ends early: {} of {} bytes",
                                                    comp.name,
                                                    filled,
                                                    comp.logical_size));
                }
                filled += got;
            }
            if (is.peek() != std::istream::traits_type::eof()) {
                throw VsagException(
                    ErrorType::INVALID_BINARY,
                    fmt::format("component {} frame holds more than its logical size {}",
                                comp.name,
                                comp.logical_size));
            }
        });
    }

    auto payload_read = [&payload](uint64_t offset, uint64_t len, void* dest) {
        if (offset > payload.size() or len > payload.size() - offset) {
            throw VsagException(ErrorType::INVALID_BINARY,
                                "seekable component reader exceeds its frame");
        }
        if (len > 0) {
            std::memcpy(dest, payload.data() + offset, len);
        }
    };
    ReadFuncStreamReader payload_reader(payload_read, 0, comp.logical_size);
    this->deserialize_whole_component(comp.name, payload_reader);
}

void
HGraph::deserialize_whole_component(const std::string& name, StreamReader& reader) {
    auto require_enabled = [&name](bool enabled) { require_component_enabled(name, enabled); };
    if (name == COMPONENT_LABEL_TABLE) {
        this->deserialize_label_info(reader);
    } else if (name == COMPONENT_CODE_SLOT_MAP) {
        require_enabled(this->using_dedup_storage());
        this->code_slot_map_->Deserialize(reader);
    } else if (name == COMPONENT_BASE_CODES) {
        this->basic_flatten_codes_->Deserialize(reader);
    } else if (name == COMPONENT_BOTTOM_GRAPH) {
        this->bottom_graph_->Deserialize(reader);
    } else if (name == COMPONENT_PRECISE_CODES) {
        require_enabled(this->high_precise_codes_ != nullptr);
        this->high_precise_codes_->Deserialize(reader);
    } else if (name == COMPONENT_ROUTE_GRAPHS) {
        for (auto& route_graph : this->route_graphs_) {
            route_graph->Deserialize(reader);
        }
    } else if (name == COMPONENT_EXTRA_INFOS) {
        require_enabled(this->extra_infos_ != nullptr);
        this->extra_infos_->Deserialize(reader);
    } else if (name == COMPONENT_ATTR_FILTER) {
        require_enabled(this->attr_filter_index_ != nullptr);
        this->attr_filter_index_->Deserialize(reader);
    } else if (name == COMPONENT_RAW_VECTOR) {
        require_enabled(this->raw_vector_ != nullptr);
        this->raw_vector_->Deserialize(reader);
    } else if (name == COMPONENT_MCI_CLIQUES) {
        require_enabled(this->mci_parameters_.enabled);
        // single-writer: each whole component is dispatched as exactly one task
        // (see parallel_deserialize_manifest), so this lazy shared_ptr assignment
        // has no concurrent writer. Preserve that invariant if mci_cliques is
        // ever moved to chunked dispatch or duplicated across tasks.
        if (this->mci_cliques_ == nullptr) {
            this->mci_cliques_ = std::make_shared<CliqueDataCell>(this->allocator_);
        }
        this->mci_cliques_->Deserialize(reader);
    } else if (name == COMPONENT_CONJUGATE_GRAPH) {
        require_enabled(this->use_conjugate_graph_);
        // the lock keeps this consistent with the search path rather than with
        // the other restore tasks: no other component task touches the
        // conjugate graph, so there is no contention here
        std::unique_lock graph_lock(this->conjugate_graph_mutex_);
        auto result = this->conjugate_graph_->Deserialize(reader);
        if (not result) {
            throw VsagException(result.error().type, result.error().message);
        }
    } else {
        throw VsagException(ErrorType::INVALID_BINARY,
                            fmt::format("unknown component in chunked layout: {}", name));
    }
}

void
HGraph::parallel_deserialize_probe(DeserializeReader& reader, ThreadPool& pool, uint64_t body_end) {
    // without a recorded layout the component boundaries are only discovered
    // while reading, so the body is walked as one sequential stream in
    // Serialize(StreamWriter&) order and every head / tail stays on this thread;
    // parallelism is limited to filling the io extents afterwards. A recorded
    // chunked layout locates every frame up front instead, which is what makes
    // the fully concurrent path possible.
    //
    // Cost of that walk: the reader is asked for every byte of the body in
    // offset order before any fill task starts, so on a high-latency
    // DeserializeReader this phase can dominate the load. It is not a buffered
    // bulk read (ReadFuncStreamReader holds no buffer, so peak memory does not
    // grow with the body), but it is serial. Callers who want the parallel load
    // to pay off should write the file through Serialize(SerializeWriter&) so
    // that a layout is recorded; this path exists to keep files written before
    // that loadable.
    auto read_func = [&reader](uint64_t offset, uint64_t len, void* dest) {
        reader.Read(offset, len, dest);
    };
    ReadFuncStreamReader body(read_func, 0, body_end);

    struct IoExtent {
        WriteRawFunc write_raw;
        uint64_t file_offset;
        uint64_t io_size;
    };
    std::vector<IoExtent> extents;

    // probe one potentially three-part component: implementations without the
    // hooks throw UNSUPPORTED before consuming any byte (so no extent has
    // been recorded yet when the fallback rewinds), then the component is
    // deserialized in place
    auto probe_component = [&](const auto& component) {
        const auto start = body.GetCursor();
        try {
            ChunkedTarget target = make_chunked_target(component);
            const auto io_size = target.reserve_io(body);
            extents.push_back(IoExtent{target.write_raw, body.GetCursor(), io_size});
            body.Seek(body.GetCursor() + io_size);
            target.deserialize_tail(body);
        } catch (const VsagException& e) {
            if (e.error_.type != ErrorType::UNSUPPORTED_INDEX_OPERATION) {
                throw;
            }
            body.Seek(start);
            component->Deserialize(body);
        }
    };

    // mirror the component order and conditions of Serialize(StreamWriter&)
    this->deserialize_label_info(body);
    if (this->using_dedup_storage()) {
        this->code_slot_map_->Deserialize(body);
    }
    probe_component(this->basic_flatten_codes_);
    probe_component(this->bottom_graph_);
    if (this->has_precise_reorder()) {
        probe_component(this->high_precise_codes_);
    }
    for (auto& route_graph : this->route_graphs_) {
        route_graph->Deserialize(body);
    }
    if (this->extra_info_size_ > 0 and this->extra_infos_ != nullptr) {
        this->extra_infos_->Deserialize(body);
    }
    if (this->use_attribute_filter_ and this->attr_filter_index_ != nullptr) {
        this->attr_filter_index_->Deserialize(body);
    }
    if (this->create_new_raw_vector_) {
        probe_component(this->raw_vector_);
    }
    if (this->mci_parameters_.enabled) {
        // single-writer, same as in deserialize_whole_component: the probe path
        // walks the body on the calling thread and only dispatches io-extent
        // fills to the pool, so this lazy shared_ptr assignment has no
        // concurrent writer. Preserve that if this section is ever dispatched.
        if (this->mci_cliques_ == nullptr) {
            this->mci_cliques_ = std::make_shared<CliqueDataCell>(this->allocator_);
        }
        this->mci_cliques_->Deserialize(body);
    }
    if (this->use_conjugate_graph_) {
        // the conjugate graph declares its total size in its first field and
        // stops FOOTER_SIZE bytes short of the end, so land the walk on the
        // declared boundary instead of wherever its cursor happens to stop.
        // Without this the walk would only stay in sync while this component
        // is the last one in the body.
        const auto start = body.GetCursor();
        uint32_t declared_size = 0;
        StreamReader::ReadObj(body, declared_size);
        body.Seek(start);
        {
            std::unique_lock graph_lock(this->conjugate_graph_mutex_);
            auto result = this->conjugate_graph_->Deserialize(body);
            if (not result) {
                throw VsagException(result.error().type, result.error().message);
            }
        }
        body.Seek(start + declared_size);
    }

    // fill the recorded io extents in parallel
    size_t task_count = 0;
    for (const auto& extent : extents) {
        task_count += (extent.io_size + PROBE_FILL_SPLIT_SIZE - 1) / PROBE_FILL_SPLIT_SIZE;
    }
    TaskBatch batch(pool, task_count);
    for (const auto& extent : extents) {
        uint64_t pos = 0;
        while (pos < extent.io_size) {
            const auto step = std::min(PROBE_FILL_SPLIT_SIZE, extent.io_size - pos);
            batch.Submit([&reader, &extent, pos, step]() {
                fill_extent_from_reader(
                    reader, extent.write_raw, extent.file_offset + pos, pos, step);
            });
            pos += step;
        }
    }
    batch.Run();
}

}  // namespace vsag
