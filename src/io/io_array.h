
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

template <typename IOTmpl>
class IOArray {
public:
    static constexpr bool InMemory = IOTmpl::InMemory;

public:
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

    IOTmpl&
    operator[](int64_t index) {
        return *datas_[index];
    }

    const IOTmpl&
    operator[](int64_t index) const {
        return *datas_[index];
    }

    IOTmpl&
    At(int64_t index) {
        if (index >= datas_.size()) {
            throw std::out_of_range("IOArray index out of range");
        }
        return *datas_[index];
    }

    const IOTmpl&
    At(int64_t index) const {
        if (index >= datas_.size()) {
            throw std::out_of_range("IOArray index out of range");
        }
        return *datas_[index];
    }

    void
    Resize(int64_t size) {
        auto cur_size = datas_.size();
        this->datas_.resize(size, nullptr);
        for (int64_t i = cur_size; i < size; i++) {
            datas_[i] = this->io_create_func_();
        }
    }

private:
    Allocator* const allocator_{nullptr};

    Vector<std::shared_ptr<IOTmpl>> datas_;

    std::unique_ptr<NonContinuousAllocator> non_continuous_allocator_{nullptr};

    std::function<std::shared_ptr<IOTmpl>()> io_create_func_;
};
}  // namespace vsag
