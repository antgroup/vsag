
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

#pragma once

#include "noncontinuous_allocator.h"
#include "typing.h"

namespace vsag {
class Allocator;

/**
 * @brief Container for managing multiple IO objects with delayed creation.
 *
 * This template class manages a dynamic array of IO objects, creating them
 * on-demand when the array is resized. It supports both in-memory and file-based
 * IO implementations, with special handling for non-continuous storage allocation.
 *
 * @tparam IOTmpl The type of IO object to manage (e.g., MemoryIO, BufferIO).
 */
template <typename IOTmpl>
class IOArray {
public:
    /// Inherits InMemory property from the IO template type.
    static constexpr bool InMemory = IOTmpl::InMemory;

public:
    /**
     * @brief Constructs an IOArray with an allocator and creation arguments.
     *
     * Stores the arguments for delayed IO object creation when resizing.
     *
     * @tparam Args Types of the constructor arguments.
     * @param allocator A pointer to the Allocator for memory management.
     * @param args Arguments to pass when constructing IO objects.
     */
    template <typename... Args>
    explicit IOArray(Allocator* allocator, Args&&... args)
        : allocator_(allocator), datas_(allocator) {
        non_continuous_allocator_ = std::make_unique<NonContinuousAllocator>(allocator);
        using ArgsTuple = std::tuple<std::decay_t<Args>...>;
        ArgsTuple args_tuple(std::forward<Args>(args)...);
        if constexpr (InMemory) {
            io_create_func_ = [args_tuple =
                                   std::move(args_tuple)]() mutable -> std::shared_ptr<IOTmpl> {
                return std::apply(
                    [](auto&&... forwarded_args) -> std::shared_ptr<IOTmpl> {
                        return std::make_shared<IOTmpl>(
                            std::forward<decltype(forwarded_args)>(forwarded_args)...);
                    },
                    args_tuple);
            };
        } else {
            auto* non_continuous_allocator = non_continuous_allocator_.get();
            io_create_func_ = [non_continuous_allocator,
                               allocator,
                               args_tuple =
                                   std::move(args_tuple)]() mutable -> std::shared_ptr<IOTmpl> {
                return std::apply(
                    [non_continuous_allocator,
                     allocator](auto&&... forwarded_args) -> std::shared_ptr<IOTmpl> {
                        return std::make_shared<IOTmpl>(
                            non_continuous_allocator,
                            allocator,
                            std::forward<decltype(forwarded_args)>(forwarded_args)...);
                    },
                    args_tuple);
            };
        }
    }

    /**
     * @brief Access an IO object by index without bounds checking.
     *
     * @param index The index of the IO object.
     * @return Reference to the IO object.
     */
    IOTmpl&
    operator[](int64_t index) {
        return *datas_[index];
    }

    /**
     * @brief Access an IO object by index without bounds checking (const version).
     *
     * @param index The index of the IO object.
     * @return Const reference to the IO object.
     */
    const IOTmpl&
    operator[](int64_t index) const {
        return *datas_[index];
    }

    /**
     * @brief Access an IO object by index with bounds checking.
     *
     * @param index The index of the IO object.
     * @return Reference to the IO object.
     * @throws std::out_of_range if index is out of bounds.
     */
    IOTmpl&
    At(int64_t index) {
        if (index >= datas_.size()) {
            throw std::out_of_range("IOArray index out of range");
        }
        return *datas_[index];
    }

    /**
     * @brief Access an IO object by index with bounds checking (const version).
     *
     * @param index The index of the IO object.
     * @return Const reference to the IO object.
     * @throws std::out_of_range if index is out of bounds.
     */
    const IOTmpl&
    At(int64_t index) const {
        if (index >= datas_.size()) {
            throw std::out_of_range("IOArray index out of range");
        }
        return *datas_[index];
    }

    /**
     * @brief Resizes the array, creating new IO objects as needed.
     *
     * @param size The new size of the array.
     */
    void
    Resize(int64_t size) {
        auto cur_size = datas_.size();
        this->datas_.resize(size, nullptr);
        for (int64_t i = cur_size; i < size; i++) {
            datas_[i] = this->io_create_func_();
        }
    }

private:
    /// Allocator for memory management.
    Allocator* const allocator_{nullptr};

    /// Vector of shared pointers to IO objects.
    Vector<std::shared_ptr<IOTmpl>> datas_;

    /// Allocator for non-continuous storage (used by file-based IO).
    std::unique_ptr<NonContinuousAllocator> non_continuous_allocator_{nullptr};

    /// Function for creating IO objects with stored arguments.
    std::function<std::shared_ptr<IOTmpl>()> io_create_func_;
};
}  // namespace vsag
