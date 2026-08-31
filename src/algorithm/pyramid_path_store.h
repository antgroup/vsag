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

#include <mutex>
#include <shared_mutex>
#include <string>

#include "typing.h"

class StreamReader;
class StreamWriter;

namespace vsag {

class PyramidPathStore {
public:
    class Writer {
    public:
        Writer(const Writer&) = delete;
        Writer&
        operator=(const Writer&) = delete;
        Writer(Writer&&) = delete;
        Writer&
        operator=(Writer&&) = delete;

        void
        EnsureSlots(uint64_t slot_count);

        void
        Insert(InnerIdType inner_id, const std::string& path);

    private:
        friend class PyramidPathStore;

        explicit Writer(PyramidPathStore& store) : store_(store), lock_(store.mutex_) {
        }

        PyramidPathStore& store_;
        std::unique_lock<std::shared_mutex> lock_;
    };

    explicit PyramidPathStore(Allocator* allocator)
        : paths_by_inner_id_(allocator), has_path_(allocator), allocator_(allocator) {
    }

    // Holds the store's exclusive lock for the lifetime of the returned writer.
    [[nodiscard]] Writer
    AcquireWriter();

    [[nodiscard]] bool
    GetPaths(const Vector<InnerIdType>& inner_ids, std::string* paths) const;

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader, uint64_t expected_count);

private:
    mutable std::shared_mutex mutex_;
    Vector<std::string> paths_by_inner_id_;
    Vector<uint8_t> has_path_;
    Allocator* allocator_{nullptr};
};

}  // namespace vsag
