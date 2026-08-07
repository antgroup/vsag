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

#include <memory>
#include <string>

#include "algorithm/build_cache.h"

namespace vsag {

class PyramidBuildCache {
public:
    explicit PyramidBuildCache(Allocator* allocator);

    ~PyramidBuildCache() = default;

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader);

    BuildCache*
    GetGraphCache(const std::string& hierarchy_name, const std::string& node_path) const;

    BuildCache&
    CreateGraphCache(const std::string& hierarchy_name, const std::string& node_path);

    bool
    Empty() const {
        for (const auto& [key, graph_cache] : graph_caches_) {
            if (graph_cache != nullptr && not graph_cache->neighbors_.empty()) {
                return false;
            }
        }
        return true;
    }

private:
    static std::string
    MakeGraphKey(const std::string& hierarchy_name, const std::string& node_path);

private:
    Allocator* const allocator_;

    // mapping from a hierarchy/node path key to one graph's build cache
    UnorderedMap<std::string, std::unique_ptr<BuildCache>> graph_caches_;
};

}  // namespace vsag
