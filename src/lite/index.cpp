// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "vsag/lite/index.h"

#include <cmath>
#include <cstring>
#include <istream>
#include <limits>
#include <new>
#include <ostream>
#include <stdexcept>

#include "lite/backend.h"
#include "lite/fp16_codec.h"

namespace vsag::lite {
namespace {

auto
failure(ErrorType type, const char* message) {
    return tl::unexpected(Error(type, message));
}

void
write(std::ostream& out, uint64_t value, uint64_t bytes = 8) {
    for (uint64_t i = 0; i < bytes; ++i) {
        out.put(static_cast<char>((value >> (8 * i)) & 255));
    }
}

void
read_bytes(std::istream& in, void* destination, uint64_t bytes) {
    if (bytes == 0) {
        return;
    }
    if (bytes > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::length_error("snapshot section exceeds stream capacity");
    }
    in.read(static_cast<char*>(destination), static_cast<std::streamsize>(bytes));
    if (not in) {
        throw std::ios_base::failure("truncated snapshot");
    }
}

uint64_t
read(std::istream& in, uint64_t bytes = 8) {
    char encoded[8]{};
    read_bytes(in, encoded, bytes);
    uint64_t value = 0;
    for (uint64_t i = 0; i < bytes; ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(encoded[i])) << (8 * i);
    }
    return value;
}

bool
native_little_endian() {
    const uint16_t value = 1;
    unsigned char first = 0;
    std::memcpy(&first, &value, 1);
    return first == 1;
}

uint64_t
byte_swap(uint64_t value) {
    value = ((value & 0x00ff00ff00ff00ffULL) << 8U) | ((value & 0xff00ff00ff00ff00ULL) >> 8U);
    value = ((value & 0x0000ffff0000ffffULL) << 16U) | ((value & 0xffff0000ffff0000ULL) >> 16U);
    return (value << 32U) | (value >> 32U);
}

uint16_t
byte_swap(uint16_t value) {
    return static_cast<uint16_t>((value << 8U) | (value >> 8U));
}

uint32_t
byte_swap(uint32_t value) {
    value = ((value & 0x00ff00ffU) << 8U) | ((value & 0xff00ff00U) >> 8U);
    return (value << 16U) | (value >> 16U);
}

constexpr uint64_t K_HEADER_BYTES = 48;
constexpr char K_MAGIC[] = "VSAGLT01";
static_assert(sizeof(float) == 4 and std::numeric_limits<float>::is_iec559);

}  // namespace

struct Index::Impl {
    explicit Impl(std::unique_ptr<detail::Backend> index_backend)
        : backend(std::move(index_backend)) {
    }
    std::unique_ptr<detail::Backend> backend;
};

Index::Index(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}
Index::~Index() = default;

tl::expected<std::unique_ptr<Index>, Error>
Index::Create(uint64_t dim) {
    auto backend = detail::make_brute_force_backend(dim);
    if (not backend) {
        return tl::unexpected(backend.error());
    }
    try {
        auto impl = std::make_unique<Impl>(std::move(*backend));
        return std::unique_ptr<Index>(new Index(std::move(impl)));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "create allocation failed");
    }
}

tl::expected<void, Error>
Index::BuildGraph(uint64_t max_degree, uint64_t ef_search) {
    return BuildGraph(VectorStorage::FP32, max_degree, ef_search);
}

tl::expected<void, Error>
Index::BuildGraph(VectorStorage storage, uint64_t max_degree, uint64_t ef_search) {
    if (impl_->backend->Kind() != BackendKind::BRUTE_FORCE) {
        return failure(ErrorType::INVALID_ARGUMENT, "index is already a graph");
    }
    auto graph = storage == VectorStorage::FP16
                     ? detail::make_fp16_graph_backend(*impl_->backend, max_degree, ef_search)
                     : detail::make_graph_backend(*impl_->backend, max_degree, ef_search);
    if (not graph) {
        return tl::unexpected(graph.error());
    }
    impl_->backend = std::move(*graph);
    return {};
}

BackendKind
Index::ActiveBackend() const {
    return impl_->backend->Kind();
}

VectorStorage
Index::ActiveVectorStorage() const {
    return impl_->backend->Storage();
}

uint64_t
Index::Size() const {
    return impl_->backend->Size();
}

uint64_t
Index::Dim() const {
    return impl_->backend->Dim();
}

tl::expected<void, Error>
Index::Add(int64_t id, const float* vector, uint64_t dim) {
    return impl_->backend->Add(id, vector, dim);
}

tl::expected<void, Error>
Index::Update(int64_t id, const float* vector, uint64_t dim) {
    return impl_->backend->Update(id, vector, dim);
}

bool
Index::Remove(int64_t id) {
    return impl_->backend->Remove(id);
}

tl::expected<std::vector<Neighbor>, Error>
Index::Search(const float* query, uint64_t dim, uint64_t k) const {
    return impl_->backend->Search(query, dim, k);
}

tl::expected<std::vector<Neighbor>, Error>
Index::Search(const float* query, uint64_t dim, uint64_t k, const IdFilter& filter) const {
    return filter ? impl_->backend->Search(query, dim, k, filter)
                  : impl_->backend->Search(query, dim, k);
}

tl::expected<void, Error>
Index::Save(std::ostream& output) const {
    const bool graph = ActiveBackend() == BackendKind::GRAPH;
    const bool fp16 = graph and ActiveVectorStorage() == VectorStorage::FP16;
    const uint64_t vector_bytes = fp16 ? 2 : 4;
    if (Dim() > (UINT64_MAX - 8) / vector_bytes or
        Size() > (UINT64_MAX - K_HEADER_BYTES) / (8 + vector_bytes * Dim())) {
        return failure(ErrorType::INVALID_ARGUMENT, "snapshot size overflow");
    }
    uint64_t payload = Size() * (8 + vector_bytes * Dim());
    if (graph) {
        if (payload > UINT64_MAX - K_HEADER_BYTES - 16 or
            Size() > (UINT64_MAX - K_HEADER_BYTES - 16 - payload) / 8) {
            return failure(ErrorType::INVALID_ARGUMENT, "graph snapshot size overflow");
        }
        payload += 16 + Size() * 8;
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            const auto links = impl_->backend->LinkCountAt(slot);
            if (links > impl_->backend->MaxDegree() or
                links > (UINT64_MAX - K_HEADER_BYTES - payload) / 8) {
                return failure(ErrorType::INVALID_ARGUMENT, "graph snapshot size overflow");
            }
            payload += links * 8;
        }
    }
    try {
        output.write(K_MAGIC, 8);
        write(output, fp16 ? 3 : (graph ? 2 : 1));
        write(output, Dim());
        write(output, Size());
        write(output, payload);
        write(output, fp16 ? 3 : (graph ? 2 : 1));
        if (graph) {
            write(output, impl_->backend->MaxDegree());
            write(output, impl_->backend->EfSearch());
        }
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            write(output, static_cast<uint64_t>(impl_->backend->IdAt(slot)));
        }
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            const auto* vector = impl_->backend->VectorAt(slot);
            for (uint64_t i = 0; i < Dim(); ++i) {
                if (fp16) {
                    write(output, detail::encode_fp16(vector[i]), 2);
                } else {
                    uint32_t bits;
                    std::memcpy(&bits, vector + i, 4);
                    write(output, bits, 4);
                }
            }
        }
        if (graph) {
            for (uint64_t slot = 0; slot < Size(); ++slot) {
                const auto links = impl_->backend->LinkCountAt(slot);
                write(output, links);
                for (uint64_t i = 0; i < links; ++i) {
                    write(output, impl_->backend->LinkAt(slot, i));
                }
            }
        }
        if (not output) {
            return failure(ErrorType::READ_ERROR, "snapshot write failed");
        }
        return {};
    } catch (const std::ios_base::failure&) {
        return failure(ErrorType::READ_ERROR, "snapshot write failed");
    }
}

tl::expected<std::unique_ptr<Index>, Error>
Index::Load(std::istream& input) {
    try {
        const auto start = input.tellg();
        input.seekg(0, std::ios::end);
        const auto end = input.tellg();
        if (start == std::streampos(-1) or end == std::streampos(-1) or end < start) {
            return failure(ErrorType::INVALID_BINARY, "a seekable input is required");
        }
        input.seekg(start);
        const auto available = static_cast<uint64_t>(end - start);
        char magic[8];
        input.read(magic, 8);
        if (not input or std::memcmp(magic, K_MAGIC, 8) != 0) {
            return failure(ErrorType::INVALID_BINARY, "invalid snapshot magic");
        }
        const auto version = read(input);
        if (version != 1 and version != 2 and version != 3) {
            return failure(ErrorType::INVALID_BINARY, "unsupported snapshot version");
        }
        const auto dim = read(input);
        const auto count = read(input);
        const auto payload = read(input);
        const auto representation = read(input);
        const uint64_t vector_bytes = version == 3 ? 2 : 4;
        if (dim == 0 or dim > (UINT64_MAX - 16) / vector_bytes or representation != version or
            available < K_HEADER_BYTES or payload != available - K_HEADER_BYTES) {
            return failure(ErrorType::INVALID_BINARY, "invalid snapshot layout");
        }
        if (version == 1) {
            if (count > (UINT64_MAX - K_HEADER_BYTES) / (8 + 4 * dim) or
                payload != count * (8 + 4 * dim)) {
                return failure(ErrorType::INVALID_BINARY, "invalid flat snapshot layout");
            }
        } else {
            if (count > (UINT64_MAX - K_HEADER_BYTES - 16) / (16 + vector_bytes * dim) or
                payload < 16 + count * (16 + vector_bytes * dim)) {
                return failure(ErrorType::INVALID_BINARY, "invalid graph snapshot layout");
            }
        }
        uint64_t degree = 0;
        uint64_t ef_search = 0;
        if (version >= 2) {
            degree = read(input);
            ef_search = read(input);
            if (degree < 2 or degree > 64 or ef_search < degree) {
                return failure(ErrorType::INVALID_BINARY, "invalid graph options");
            }
        }
        // Validate the layout before allocating and keep one owned copy of each payload.
        const bool little_endian = native_little_endian();
        std::vector<int64_t> ids(count);
        read_bytes(input, ids.data(), count * sizeof(int64_t));
        if (not little_endian) {
            for (auto& id : ids) {
                uint64_t bits = 0;
                std::memcpy(&bits, &id, sizeof(bits));
                bits = byte_swap(bits);
                std::memcpy(&id, &bits, sizeof(id));
            }
        }
        std::vector<float> vectors;
        std::vector<uint16_t> fp16_vectors;
        if (version == 3) {
            fp16_vectors.resize(count * dim);
            read_bytes(input, fp16_vectors.data(), fp16_vectors.size() * sizeof(uint16_t));
            for (auto& value : fp16_vectors) {
                if (not little_endian) {
                    value = byte_swap(value);
                }
                if (not detail::is_finite_fp16(value)) {
                    return failure(ErrorType::INVALID_BINARY, "non-finite snapshot vector");
                }
            }
        } else {
            vectors.resize(count * dim);
            read_bytes(input, vectors.data(), vectors.size() * sizeof(float));
            for (auto& value : vectors) {
                if (not little_endian) {
                    uint32_t bits = 0;
                    std::memcpy(&bits, &value, sizeof(bits));
                    bits = byte_swap(bits);
                    std::memcpy(&value, &bits, sizeof(value));
                }
                if (not std::isfinite(value)) {
                    return failure(ErrorType::INVALID_BINARY, "non-finite snapshot vector");
                }
            }
        }
        std::vector<std::vector<uint64_t>> links;
        if (version >= 2) {
            uint64_t link_bytes = payload - 16 - count * (16 + vector_bytes * dim);
            links.reserve(count);
            for (uint64_t slot = 0; slot < count; ++slot) {
                const auto link_count = read(input);
                if (link_count > degree or link_count > link_bytes / 8) {
                    return failure(ErrorType::INVALID_BINARY, "invalid graph link count");
                }
                std::vector<uint64_t> row(link_count);
                read_bytes(input, row.data(), link_count * sizeof(uint64_t));
                if (not little_endian) {
                    for (auto& neighbor : row) {
                        neighbor = byte_swap(neighbor);
                    }
                }
                links.push_back(std::move(row));
                link_bytes -= link_count * 8;
            }
            if (link_bytes != 0 or input.tellg() != end) {
                return failure(ErrorType::INVALID_BINARY, "graph snapshot trailing bytes");
            }
        }
        auto backend =
            version == 1
                ? detail::restore_brute_force_backend(dim, std::move(ids), std::move(vectors))
            : version == 3
                ? detail::restore_fp16_graph_backend(dim,
                                                     degree,
                                                     ef_search,
                                                     std::move(ids),
                                                     std::move(fp16_vectors),
                                                     std::move(links))
                : detail::restore_graph_backend(
                      dim, degree, ef_search, std::move(ids), std::move(vectors), std::move(links));
        if (not backend) {
            return tl::unexpected(backend.error());
        }
        auto impl = std::make_unique<Impl>(std::move(*backend));
        return std::unique_ptr<Index>(new Index(std::move(impl)));
    } catch (const std::ios_base::failure&) {
        return failure(ErrorType::INVALID_BINARY, "snapshot read failed");
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "load allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::INVALID_BINARY, "snapshot exceeds container capacity");
    }
}

}  // namespace vsag::lite
