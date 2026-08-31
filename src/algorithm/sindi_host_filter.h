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
#include <utility>

#include "container_types.h"
#include "json_types.h"
#include "storage/stream_reader.h"
#include "storage/stream_writer.h"
#include "vsag/dataset.h"
#include "vsag/filter.h"

namespace vsag {

inline constexpr const char* SINDI_HOST_ID_METADATA_NAME = "host_id";
inline constexpr const char* SINDI_HAS_HOST_METADATA_KEY = "has_host_metadata";

enum class SindiHostRouteKind : uint8_t {
    UNFILTERED,
    EMPTY,
    WINDOW,
};

struct SindiHostSearchRoute {
    SindiHostRouteKind kind{SindiHostRouteKind::UNFILTERED};
    uint32_t begin{0};
    uint32_t end{0};
    uint32_t host_index{0};
};

struct SindiHostRange {
    uint32_t begin{0};
    uint32_t end{0};
};

class SindiHostBuildPlan {
public:
    explicit SindiHostBuildPlan(Allocator* allocator);

    [[nodiscard]] bool
    Enabled() const {
        return enabled_;
    }

    [[nodiscard]] uint32_t
    SourceIndex(uint32_t ordered_position) const {
        return enabled_ ? order_[ordered_position] : ordered_position;
    }

    void
    RecordSuccess(uint32_t ordered_position);

private:
    friend class SindiHostFilter;

    bool enabled_{false};
    uint32_t successful_host_cursor_{0};
    Vector<uint32_t> order_;
    Vector<uint32_t> host_ids_;
    Vector<uint32_t> input_offsets_;
    Vector<uint32_t> successful_counts_;
};

class SindiHostFilter {
public:
    explicit SindiHostFilter(Allocator* allocator);

    SindiHostBuildPlan
    PrepareBuild(const DatasetPtr& base, uint64_t current_element_count) const;

    void
    CommitBuild(SindiHostBuildPlan&& plan, uint32_t first_inner_id, uint32_t end_inner_id);

    void
    Clear();

    [[nodiscard]] bool
    HasMetadata() const {
        return not host_ids_.empty();
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const {
        return (host_ids_.size() + host_range_offsets_.size()) * sizeof(uint32_t) +
               host_ranges_.size() * sizeof(SindiHostRange);
    }

    [[nodiscard]] SindiHostSearchRoute
    Classify(const DatasetPtr& query) const;

    void
    ApplyFilter(const SindiHostSearchRoute& route, FilterPtr& filter) const;

    static void
    ApplyWindowRoute(const SindiHostSearchRoute& route,
                     uint32_t window_size,
                     int64_t& min_window_id,
                     int64_t& max_window_id);

    [[nodiscard]] int64_t
    NextMatchingWindow(const SindiHostSearchRoute& route,
                       uint32_t window_size,
                       int64_t current_window_id,
                       int64_t max_window_id) const;

    [[nodiscard]] bool
    RequiresFullTermScan(const SindiHostSearchRoute& route,
                         uint32_t window_id,
                         uint32_t window_size) const;

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader, uint64_t element_count);

private:
    Vector<uint32_t> host_ids_;
    Vector<uint32_t> host_range_offsets_;
    Vector<SindiHostRange> host_ranges_;
};

}  // namespace vsag
