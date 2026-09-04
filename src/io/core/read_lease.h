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

#include <cstdint>
#include <cstdlib>
#include <utility>

#include "vsag/allocator.h"

namespace vsag {

struct BorrowedOwner {};

class AllocatorOwner {
public:
    AllocatorOwner() = default;

    AllocatorOwner(Allocator* allocator, uint8_t* data) : allocator_(allocator), data_(data) {
    }

    ~AllocatorOwner() {
        Reset();
    }

    AllocatorOwner(const AllocatorOwner&) = delete;
    AllocatorOwner&
    operator=(const AllocatorOwner&) = delete;

    AllocatorOwner(AllocatorOwner&& other) noexcept
        : allocator_(other.allocator_), data_(other.data_) {
        other.allocator_ = nullptr;
        other.data_ = nullptr;
    }

    AllocatorOwner&
    operator=(AllocatorOwner&& other) noexcept {
        if (this != &other) {
            Reset();
            allocator_ = other.allocator_;
            data_ = other.data_;
            other.allocator_ = nullptr;
            other.data_ = nullptr;
        }
        return *this;
    }

private:
    void
    Reset() {
        if (allocator_ != nullptr and data_ != nullptr) {
            allocator_->Deallocate(data_);
        }
        allocator_ = nullptr;
        data_ = nullptr;
    }

    Allocator* allocator_{nullptr};
    uint8_t* data_{nullptr};
};

class AlignedOwner {
public:
    AlignedOwner() = default;

    explicit AlignedOwner(uint8_t* base) : base_(base) {
    }

    ~AlignedOwner() {
        std::free(base_);
    }

    AlignedOwner(const AlignedOwner&) = delete;
    AlignedOwner&
    operator=(const AlignedOwner&) = delete;

    AlignedOwner(AlignedOwner&& other) noexcept : base_(other.base_) {
        other.base_ = nullptr;
    }

    AlignedOwner&
    operator=(AlignedOwner&& other) noexcept {
        if (this != &other) {
            std::free(base_);
            base_ = other.base_;
            other.base_ = nullptr;
        }
        return *this;
    }

private:
    uint8_t* base_{nullptr};
};

class ConfigurableOwner {
public:
    ConfigurableOwner() = default;

    static ConfigurableOwner
    AllocatorOwned(Allocator* allocator, uint8_t* data) {
        return ConfigurableOwner(allocator, data, false);
    }

    static ConfigurableOwner
    AlignedOwned(uint8_t* data) {
        return ConfigurableOwner(nullptr, data, true);
    }

    ~ConfigurableOwner() {
        Reset();
    }

    ConfigurableOwner(const ConfigurableOwner&) = delete;
    ConfigurableOwner&
    operator=(const ConfigurableOwner&) = delete;

    ConfigurableOwner(ConfigurableOwner&& other) noexcept
        : allocator_(other.allocator_), data_(other.data_), aligned_(other.aligned_) {
        other.allocator_ = nullptr;
        other.data_ = nullptr;
        other.aligned_ = false;
    }

    ConfigurableOwner&
    operator=(ConfigurableOwner&& other) noexcept {
        if (this != &other) {
            Reset();
            allocator_ = other.allocator_;
            data_ = other.data_;
            aligned_ = other.aligned_;
            other.allocator_ = nullptr;
            other.data_ = nullptr;
            other.aligned_ = false;
        }
        return *this;
    }

private:
    ConfigurableOwner(Allocator* allocator, uint8_t* data, bool aligned)
        : allocator_(allocator), data_(data), aligned_(aligned) {
    }

    void
    Reset() {
        if (data_ != nullptr) {
            if (aligned_) {
                std::free(data_);
            } else if (allocator_ != nullptr) {
                allocator_->Deallocate(data_);
            }
        }
        allocator_ = nullptr;
        data_ = nullptr;
        aligned_ = false;
    }

    Allocator* allocator_{nullptr};
    uint8_t* data_{nullptr};
    bool aligned_{false};
};

template <typename Owner>
class BasicReadLease : private Owner {
public:
    BasicReadLease() = default;

    BasicReadLease(const uint8_t* data, uint64_t size, Owner owner)
        : Owner(std::move(owner)), data_(data), size_(size) {
    }

    BasicReadLease(const BasicReadLease&) = delete;
    BasicReadLease&
    operator=(const BasicReadLease&) = delete;

    BasicReadLease(BasicReadLease&& other) noexcept
        : Owner(std::move(static_cast<Owner&>(other))),
          data_(std::exchange(other.data_, nullptr)),
          size_(std::exchange(other.size_, 0)) {
    }

    BasicReadLease&
    operator=(BasicReadLease&& other) noexcept {
        if (this != &other) {
            static_cast<Owner&>(*this) = std::move(static_cast<Owner&>(other));
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    [[nodiscard]] explicit operator bool() const {
        return data_ != nullptr;
    }

    [[nodiscard]] const uint8_t*
    Data() const {
        return data_;
    }

    [[nodiscard]] uint64_t
    Size() const {
        return size_;
    }

private:
    const uint8_t* data_{nullptr};
    uint64_t size_{0};
};

using BorrowedLease = BasicReadLease<BorrowedOwner>;
using AllocatorLease = BasicReadLease<AllocatorOwner>;
using AlignedLease = BasicReadLease<AlignedOwner>;
using ConfigurableLease = BasicReadLease<ConfigurableOwner>;

static_assert(sizeof(BorrowedLease) == sizeof(const uint8_t*) + sizeof(uint64_t));

}  // namespace vsag
