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

uint64_t
read(std::istream& in, uint64_t bytes = 8) {
    uint64_t value = 0;
    for (uint64_t i = 0; i < bytes; ++i) {
        auto byte = in.get();
        if (byte == std::char_traits<char>::eof()) {
            throw std::ios_base::failure("truncated snapshot");
        }
        value |= static_cast<uint64_t>(static_cast<unsigned char>(byte)) << (8 * i);
    }
    return value;
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
    return impl_->backend->Search(query, dim, k, filter ? &filter : nullptr);
}

tl::expected<void, Error>
Index::Save(std::ostream& output) const {
    if (Dim() > (UINT64_MAX - 8) / 4 or Size() > (UINT64_MAX - K_HEADER_BYTES) / (8 + 4 * Dim())) {
        return failure(ErrorType::INVALID_ARGUMENT, "snapshot size overflow");
    }
    try {
        output.write(K_MAGIC, 8);
        write(output, 1);
        write(output, Dim());
        write(output, Size());
        write(output, Size() * (8 + 4 * Dim()));
        write(output, 1);
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            write(output, static_cast<uint64_t>(impl_->backend->IdAt(slot)));
        }
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            const auto* vector = impl_->backend->VectorAt(slot);
            for (uint64_t i = 0; i < Dim(); ++i) {
                uint32_t bits;
                std::memcpy(&bits, vector + i, 4);
                write(output, bits, 4);
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
        if (not input or std::memcmp(magic, K_MAGIC, 8) != 0 or read(input) != 1) {
            return failure(ErrorType::INVALID_BINARY, "invalid magic or version");
        }
        const auto dim = read(input);
        const auto count = read(input);
        const auto payload = read(input);
        const auto representation = read(input);
        if (dim == 0 or dim > (UINT64_MAX - 8) / 4 or representation != 1 or
            count > (UINT64_MAX - K_HEADER_BYTES) / (8 + 4 * dim) or
            payload != count * (8 + 4 * dim) or available != K_HEADER_BYTES + payload) {
            return failure(ErrorType::INVALID_BINARY, "invalid snapshot layout");
        }
        // Validate the layout before allocating and keep one owned copy of each payload.
        std::vector<int64_t> ids;
        ids.reserve(count);
        for (uint64_t slot = 0; slot < count; ++slot) {
            const uint64_t bits = read(input);
            int64_t id;
            std::memcpy(&id, &bits, sizeof(id));
            ids.push_back(id);
        }
        std::vector<float> vectors;
        vectors.reserve(count * dim);
        for (uint64_t i = 0; i < count * dim; ++i) {
            auto bits = static_cast<uint32_t>(read(input, 4));
            float value;
            std::memcpy(&value, &bits, 4);
            if (not std::isfinite(value)) {
                return failure(ErrorType::INVALID_BINARY, "non-finite snapshot vector");
            }
            vectors.push_back(value);
        }
        auto backend = detail::restore_brute_force_backend(dim, std::move(ids), std::move(vectors));
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
