// Copyright 2024-present the vsag project
// SPDX-License-Identifier: Apache-2.0
#include "vsag/lite/index.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <istream>
#include <limits>
#include <new>
#include <ostream>
#include <queue>
#include <stdexcept>
#include <unordered_map>

#include "simd/kernels/compute_l2.h"
#include "simd/traits/simd_traits_generic.h"

namespace vsag::lite {
namespace {

auto
failure(ErrorType type, const char* message) {
    return tl::unexpected(Error(type, message));
}

tl::expected<void, Error>
validate(const float* data, uint64_t actual, uint64_t expected) {
    if (actual != expected) {
        return failure(ErrorType::DIMENSION_NOT_EQUAL, "dimension mismatch");
    }
    if (data == nullptr) {
        return failure(ErrorType::INVALID_ARGUMENT, "null vector");
    }
    for (uint64_t i = 0; i < actual; ++i) {
        if (not std::isfinite(data[i])) {
            return failure(ErrorType::INVALID_ARGUMENT, "non-finite vector");
        }
    }
    return {};
}

bool
better(const Neighbor& a, const Neighbor& b) {
    return a.distance < b.distance or (a.distance == b.distance and a.id < b.id);
}

struct NeighborWorseFirst {
    bool
    operator()(const Neighbor& a, const Neighbor& b) const {
        return better(a, b);
    }
};

// Explicit byte encoding: never serialize native containers, pointers, or struct padding.
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
    explicit Impl(uint64_t dimension) : dim(dimension) {
    }
    uint64_t dim;
    std::vector<float> vectors;
    std::vector<int64_t> ids;
    std::unordered_map<int64_t, uint64_t> slots;
};

Index::Index(uint64_t dim) : impl_(std::make_unique<Impl>(dim)) {
}
Index::~Index() = default;

tl::expected<std::unique_ptr<Index>, Error>
Index::Create(uint64_t dim) {
    if (dim == 0 or dim > std::vector<float>().max_size()) {
        return failure(ErrorType::INVALID_ARGUMENT, "invalid dimension");
    }
    try {
        return std::unique_ptr<Index>(new Index(dim));
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "create allocation failed");
    }
}

uint64_t
Index::Size() const {
    return impl_->ids.size();
}

uint64_t
Index::Dim() const {
    return impl_->dim;
}

tl::expected<void, Error>
Index::Add(int64_t id, const float* vector, uint64_t dim) {
    auto valid = validate(vector, dim, Dim());
    if (not valid) {
        return tl::unexpected(valid.error());
    }
    if (impl_->slots.count(id) != 0) {
        return failure(ErrorType::INVALID_ARGUMENT, "duplicate ID");
    }
    if (Size() >= impl_->vectors.max_size() / dim or Size() >= impl_->ids.max_size()) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "index capacity exceeded");
    }
    try {
        // Reserve geometrically, before publishing any logical record. reserve(n+1) each time
        // would turn a sequence of single-vector inserts into quadratic copying.
        auto grow = [](auto& values, uint64_t required) {
            if (required > values.capacity()) {
                const uint64_t maximum = values.max_size();
                const uint64_t capacity = values.capacity();
                const uint64_t doubled =
                    capacity == 0 ? 1 : (capacity > maximum / 2 ? maximum : capacity * 2);
                values.reserve(std::max(required, doubled));
            }
        };
        const uint64_t slot = Size();
        grow(impl_->vectors, (slot + 1) * dim);
        grow(impl_->ids, slot + 1);
        // Publish the map entry only after every operation that can reallocate a vector has
        // succeeded. The following scalar inserts cannot throw after the reserves above.
        impl_->slots.emplace(id, slot);
        impl_->vectors.insert(impl_->vectors.end(), vector, vector + dim);
        impl_->ids.push_back(id);
        return {};
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "add allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "add capacity exceeded");
    }
}

tl::expected<void, Error>
Index::Update(int64_t id, const float* vector, uint64_t dim) {
    auto valid = validate(vector, dim, Dim());
    if (not valid) {
        return tl::unexpected(valid.error());
    }
    auto slot = impl_->slots.find(id);
    if (slot == impl_->slots.end()) {
        return failure(ErrorType::INVALID_ARGUMENT, "missing ID");
    }
    std::copy_n(vector, dim, impl_->vectors.data() + slot->second * dim);
    return {};
}

bool
Index::Remove(int64_t id) {
    auto found = impl_->slots.find(id);
    if (found == impl_->slots.end()) {
        return false;
    }
    const uint64_t slot = found->second;
    const uint64_t last = Size() - 1;
    // Same dense-slot principle as Full BruteForce::Remove(FORCE_REMOVE).
    // This is deliberately not a graph-compatible stable-slot contract.
    if (slot != last) {
        std::copy_n(
            impl_->vectors.data() + last * Dim(), Dim(), impl_->vectors.data() + slot * Dim());
        impl_->ids[slot] = impl_->ids[last];
        impl_->slots.at(impl_->ids[slot]) = slot;
    }
    impl_->slots.erase(found);
    impl_->ids.pop_back();
    impl_->vectors.resize(last * Dim());
    return true;
}

tl::expected<std::vector<Neighbor>, Error>
Index::Search(const float* query, uint64_t dim, uint64_t k) const {
    auto valid = validate(query, dim, Dim());
    if (not valid) {
        return tl::unexpected(valid.error());
    }
    try {
        k = std::min(k, Size());
        if (k == 0) {
            return std::vector<Neighbor>{};
        }
        std::priority_queue<Neighbor, std::vector<Neighbor>, NeighborWorseFirst> heap;
        for (uint64_t slot = 0; slot < Size(); ++slot) {
            const auto distance = simd::ComputeL2SqrImpl<simd::SimdTraits<simd::GenericTag>>(
                query, impl_->vectors.data() + slot * dim, dim);
            Neighbor next{impl_->ids[slot], distance};
            if (heap.size() < k) {
                heap.push(next);
            } else if (better(next, heap.top())) {
                heap.pop();
                heap.push(next);
            }
        }
        std::vector<Neighbor> result(heap.size());
        for (uint64_t i = result.size(); i > 0; --i) {
            result[i - 1] = heap.top();
            heap.pop();
        }
        return result;
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "search allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "search capacity exceeded");
    }
}

tl::expected<void, Error>
Index::Save(std::ostream& output) const {
    if (Dim() > (UINT64_MAX - 8) / 4 or Size() > (UINT64_MAX - K_HEADER_BYTES) / (8 + 4 * Dim())) {
        return failure(ErrorType::INVALID_ARGUMENT, "snapshot size overflow");
    }
    try {
        output.write(K_MAGIC, 8);
        write(output, 1);  // Format version.
        write(output, Dim());
        write(output, Size());
        write(output, Size() * (8 + 4 * Dim()));
        write(output, 1);  // FP32 squared-L2 representation.
        for (const auto& id : impl_->ids) {
            write(output, static_cast<uint64_t>(id));
        }
        for (float value : impl_->vectors) {
            uint32_t bits;
            std::memcpy(&bits, &value, 4);
            write(output, bits, 4);
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
        auto created = Create(dim);
        if (not created) {
            return tl::unexpected(created.error());
        }
        auto& data = *(*created)->impl_;
        // Bounds validated against the actual input length before any payload allocation.
        data.ids.reserve(count);
        data.slots.reserve(count);
        for (uint64_t slot = 0; slot < count; ++slot) {
            const uint64_t bits = read(input);
            int64_t id;
            std::memcpy(&id, &bits, sizeof(id));
            if (not data.slots.emplace(id, slot).second) {
                return failure(ErrorType::INVALID_BINARY, "duplicate snapshot ID");
            }
            data.ids.push_back(id);
        }
        data.vectors.reserve(count * dim);
        for (uint64_t i = 0; i < count * dim; ++i) {
            auto bits = static_cast<uint32_t>(read(input, 4));
            float value;
            std::memcpy(&value, &bits, 4);
            if (not std::isfinite(value)) {
                return failure(ErrorType::INVALID_BINARY, "non-finite snapshot vector");
            }
            data.vectors.push_back(value);
        }
        return std::move(*created);
    } catch (const std::ios_base::failure&) {
        return failure(ErrorType::INVALID_BINARY, "snapshot read failed");
    } catch (const std::bad_alloc&) {
        return failure(ErrorType::NO_ENOUGH_MEMORY, "load allocation failed");
    } catch (const std::length_error&) {
        return failure(ErrorType::INVALID_BINARY, "snapshot exceeds container capacity");
    }
}

}  // namespace vsag::lite
