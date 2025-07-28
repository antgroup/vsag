
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

#include <cstdint>
#include <vector>

class AppendBuffer {
public:
    AppendBuffer() {
        buffer_.emplace_back(new char[BUFFER_SIZE]);
    }
    ~AppendBuffer() = default;

    const char*
    Append(const char* data, uint64_t size);

    constexpr static uint64_t BUFFER_SIZE = 4 * 1024;

private:
    std::vector<const char*> buffer_;
    uint64_t cursor_{0};
};
