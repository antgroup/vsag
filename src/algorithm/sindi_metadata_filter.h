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
#include <vector>

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

    [[nodiscard]] SindiHostBuildPlan
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

inline constexpr const char* SINDI_DATE_PATH_NAME = "date";
inline constexpr const char* SINDI_DATE_BEGIN_PATH_NAME = "date_begin";
inline constexpr const char* SINDI_DATE_END_PATH_NAME = "date_end";
inline constexpr const char* SINDI_DATE_METADATA_FORMAT_VERSION_KEY =
    "sindi_date_metadata_format_version";
inline constexpr uint32_t SINDI_DATE_METADATA_FORMAT_VERSION = 1;

class SindiDateBuildPlan {
public:
    explicit SindiDateBuildPlan(Allocator* allocator);

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
    friend class SindiDateFilter;

    bool enabled_{false};
    bool has_host_metadata_{false};
    uint32_t successful_group_cursor_{0};
    Vector<uint32_t> order_;
    Vector<uint32_t> source_buckets_;
    Vector<uint32_t> group_quarters_;
    Vector<uint32_t> group_hosts_;
    Vector<uint32_t> input_offsets_;
    Vector<uint32_t> successful_counts_;
    Vector<uint32_t> successful_buckets_;
};

struct SindiDateSearchRoute {
    // Search-local filters may reference these ranges and must not outlive this route.
    bool enabled{false};
    SindiHostRouteKind kind{SindiHostRouteKind::UNFILTERED};
    bool has_date_bucket{false};
    bool has_date_range{false};
    uint32_t query_bucket{0};
    uint32_t query_begin{0};
    uint32_t query_end{0};
    std::vector<std::pair<uint32_t, uint32_t>> inner_ranges;
    std::vector<std::pair<uint32_t, uint32_t>> window_ranges;
};

class SindiDateFilter {
public:
    explicit SindiDateFilter(Allocator* allocator);

    [[nodiscard]] SindiDateBuildPlan
    PrepareBuild(const DatasetPtr& base) const;

    void
    CommitBuild(SindiDateBuildPlan&& plan, uint64_t element_count);

    void
    Clear();

    [[nodiscard]] bool
    HasMetadata() const {
        return not partitions_.empty();
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const;

    [[nodiscard]] SindiDateSearchRoute
    Classify(const DatasetPtr& query, uint32_t window_size) const;

    void
    ApplyFilter(const SindiDateSearchRoute& route, FilterPtr& filter) const;

    static void
    ApplyWindowRoute(const SindiDateSearchRoute& route,
                     int64_t& min_window_id,
                     int64_t& max_window_id);

    [[nodiscard]] static int64_t
    NextMatchingWindow(const SindiDateSearchRoute& route,
                       int64_t current_window_id,
                       int64_t max_window_id);

    // Exact date predicates are evaluated per document, so date queries scan every term in each
    // routed window before applying the date filter.
    [[nodiscard]] static bool
    RequiresFullTermScan(const SindiDateSearchRoute& route,
                         uint32_t window_id,
                         uint32_t window_size);

    void
    Serialize(StreamWriter& writer) const;

    void
    Deserialize(StreamReader& reader, uint64_t element_count);

private:
    struct Partition {
        explicit Partition(Allocator* allocator) : host_ids(allocator), host_offsets(allocator) {
        }

        uint32_t quarter{0};
        uint32_t begin{0};
        uint32_t end{0};
        Vector<uint32_t> host_ids;
        Vector<uint32_t> host_offsets;
    };

    [[nodiscard]] FilterPtr
    create_filter(const SindiDateSearchRoute& route, FilterPtr filter) const;

    Allocator* allocator_{nullptr};
    bool has_host_metadata_{false};
    Vector<uint32_t> document_buckets_;
    Vector<Partition> partitions_;
};

class SindiMetadataBuildPlan {
public:
    explicit SindiMetadataBuildPlan(Allocator* allocator)
        : host_plan_(allocator), date_plan_(allocator) {
    }

    [[nodiscard]] bool
    Enabled() const {
        return host_plan_.Enabled() or date_plan_.Enabled();
    }

    [[nodiscard]] uint32_t
    SourceIndex(uint32_t ordered_position) const {
        return date_plan_.Enabled() ? date_plan_.SourceIndex(ordered_position)
                                    : host_plan_.SourceIndex(ordered_position);
    }

    void
    RecordSuccess(uint32_t ordered_position) {
        host_plan_.RecordSuccess(ordered_position);
        date_plan_.RecordSuccess(ordered_position);
    }

private:
    friend class SindiMetadataFilter;

    SindiHostBuildPlan host_plan_;
    SindiDateBuildPlan date_plan_;
};

struct SindiMetadataSearchRoute {
    SindiHostRouteKind kind{SindiHostRouteKind::UNFILTERED};
    SindiHostSearchRoute host_route;
    SindiDateSearchRoute date_route;
};

class SindiMetadataFilter {
public:
    explicit SindiMetadataFilter(Allocator* allocator);

    [[nodiscard]] SindiMetadataBuildPlan
    PrepareBuild(const DatasetPtr& base, uint64_t current_element_count) const;

    void
    CommitBuild(SindiMetadataBuildPlan&& plan, uint32_t first_inner_id, uint32_t end_inner_id);

    void
    Clear();

    [[nodiscard]] bool
    HasHostMetadata() const {
        return host_filter_.HasMetadata();
    }

    [[nodiscard]] bool
    HasDateMetadata() const {
        return date_filter_.HasMetadata();
    }

    [[nodiscard]] uint64_t
    GetMemoryUsage() const {
        return host_filter_.GetMemoryUsage() + date_filter_.GetMemoryUsage();
    }

    [[nodiscard]] SindiMetadataSearchRoute
    Classify(const DatasetPtr& query, uint32_t window_size) const;

    void
    ApplyFilter(const SindiMetadataSearchRoute& route, FilterPtr& filter) const;

    static void
    ApplyWindowRoute(const SindiMetadataSearchRoute& route,
                     uint32_t window_size,
                     int64_t& min_window_id,
                     int64_t& max_window_id);

    [[nodiscard]] int64_t
    NextMatchingWindow(const SindiMetadataSearchRoute& route,
                       uint32_t window_size,
                       int64_t current_window_id,
                       int64_t max_window_id) const;

    [[nodiscard]] bool
    RequiresFullTermScan(const SindiMetadataSearchRoute& route,
                         uint32_t window_id,
                         uint32_t window_size) const;

    void
    SerializeHostMetadata(StreamWriter& writer) const {
        host_filter_.Serialize(writer);
    }

    void
    SerializeDateMetadata(StreamWriter& writer) const {
        date_filter_.Serialize(writer);
    }

    void
    DeserializeHostMetadata(StreamReader& reader, uint64_t element_count);

    void
    DeserializeDateMetadata(StreamReader& reader, uint64_t element_count);

private:
    Allocator* allocator_{nullptr};
    SindiHostFilter host_filter_;
    SindiDateFilter date_filter_;
};

}  // namespace vsag
