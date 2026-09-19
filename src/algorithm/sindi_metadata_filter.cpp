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

#include "sindi_metadata_filter.h"

#include <fmt/format.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <string>

#include "common.h"
#include "utils/util_functions.h"
#include "vsag_exception.h"

namespace vsag {
namespace {

class InnerIdHostFilter : public Filter {
public:
    InnerIdHostFilter(const Vector<SindiHostRange>* ranges,
                      uint32_t range_begin,
                      uint32_t range_end,
                      FilterPtr filter)
        : ranges_(ranges),
          range_begin_(range_begin),
          range_end_(range_end),
          filter_(std::move(filter)) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        if (id < 0) {
            return false;
        }
        const auto inner_id = static_cast<uint64_t>(id);
        const auto range_begin = ranges_->begin() + range_begin_;
        const auto range_end = ranges_->begin() + range_end_;
        const auto range = std::upper_bound(
            range_begin, range_end, inner_id, [](uint64_t value, const SindiHostRange& candidate) {
                return value < static_cast<uint64_t>(candidate.begin);
            });
        if (range == range_begin or inner_id >= std::prev(range)->end) {
            return false;
        }
        return filter_ == nullptr or filter_->CheckValid(id);
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        if (filter_ != nullptr) {
            filter_->GetValidIds(valid_ids, count);
        }
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return filter_ == nullptr ? 1.0F : filter_->ValidRatio();
    }

    [[nodiscard]] Distribution
    FilterDistribution() const override {
        return filter_ == nullptr ? Distribution::NONE : filter_->FilterDistribution();
    }

private:
    const Vector<SindiHostRange>* ranges_;
    uint32_t range_begin_;
    uint32_t range_end_;
    FilterPtr filter_;
};

}  // namespace

SindiHostDictionary::SindiHostDictionary(Allocator* allocator)
    : allocator_(allocator), host_bytes_(allocator), host_offsets_(allocator) {
}

void
SindiHostDictionary::Encode(const std::string* hosts,
                            uint64_t count,
                            Vector<uint32_t>& host_ids,
                            std::vector<std::string_view>& new_hosts) const {
    CHECK_ARGUMENT(hosts != nullptr, "SINDI host metadata must not be null");
    host_ids.resize(count);
    new_hosts.clear();
    std::unordered_map<std::string_view, uint32_t> new_host_lookup;
    // Host ID zero is reserved for the empty-string sentinel even before the dictionary is
    // committed and Size() starts reporting it.
    const uint64_t existing_count = host_offsets_.empty() ? 1 : this->Size();
    for (uint64_t i = 0; i < count; ++i) {
        const auto& host = hosts[i];
        if (host.empty()) {
            host_ids[i] = 0;
            continue;
        }
        uint32_t host_id = 0;
        if (this->Lookup(host, host_id)) {
            host_ids[i] = host_id;
            continue;
        }
        const auto new_host = new_host_lookup.find(host);
        if (new_host != new_host_lookup.end()) {
            host_ids[i] = new_host->second;
            continue;
        }
        const auto next_id = existing_count + new_hosts.size();
        CHECK_ARGUMENT(next_id <= std::numeric_limits<uint32_t>::max(),
                       "SINDI host dictionary exceeds uint32_t capacity");
        host_id = static_cast<uint32_t>(next_id);
        new_host_lookup.emplace(host, host_id);
        new_hosts.push_back(host);
        host_ids[i] = host_id;
    }
}

void
SindiHostDictionary::Commit(const std::vector<std::string_view>& new_hosts) {
    if (host_offsets_.empty()) {
        host_offsets_.push_back(0);
        host_offsets_.push_back(0);
    }
    for (const auto& host : new_hosts) {
        CHECK_ARGUMENT(not host.empty(), "SINDI nonzero host dictionary entry must not be empty");
        host_bytes_.insert(host_bytes_.end(), host.begin(), host.end());
        host_offsets_.push_back(host_bytes_.size());
    }
    this->RebuildLookup();
}

bool
SindiHostDictionary::Lookup(std::string_view host, uint32_t& host_id) const {
    if (host.empty()) {
        // An empty host maps to the reserved missing-host ID only after a dictionary exists.
        host_id = 0;
        return not host_offsets_.empty();
    }
    const auto found = host_lookup_.find(host);
    if (found == host_lookup_.end()) {
        return false;
    }
    host_id = found->second;
    return true;
}

uint64_t
SindiHostDictionary::GetMemoryUsage() const {
    return host_bytes_.size() * sizeof(char) + host_offsets_.size() * sizeof(uint64_t) +
           host_lookup_.size() *
               (sizeof(std::pair<const std::string_view, uint32_t>) + sizeof(void*));
}

void
SindiHostDictionary::Clear() {
    Vector<char>(allocator_).swap(host_bytes_);
    Vector<uint64_t>(allocator_).swap(host_offsets_);
    host_lookup_.clear();
    host_lookup_.rehash(0);
}

void
SindiHostDictionary::RebuildLookup() {
    host_lookup_.clear();
    if (host_offsets_.empty()) {
        return;
    }
    host_lookup_.reserve(this->Size());
    for (uint64_t host_index = 0; host_index < this->Size(); ++host_index) {
        const auto host_id = static_cast<uint32_t>(host_index);
        const auto begin = host_offsets_[host_index];
        const auto end = host_offsets_[host_index + 1];
        const std::string_view host =
            host_id == 0 ? std::string_view{}
                         : std::string_view(host_bytes_.data() + begin, end - begin);
        const auto [unused, inserted] = host_lookup_.emplace(host, host_id);
        CHECK_ARGUMENT(inserted, "SINDI host dictionary entries must be unique");
    }
}

void
SindiHostDictionary::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteVector(writer, host_offsets_);
    StreamWriter::WriteVector(writer, host_bytes_);
}

void
SindiHostDictionary::Deserialize(StreamReader& reader, uint64_t element_count) {
    (void)element_count;
    uint64_t offset_count = 0;
    StreamReader::ReadObj(reader, offset_count);
    CHECK_ARGUMENT(reader.GetCursor() <= reader.Length(),
                   "serialized SINDI host dictionary offset position is invalid");
    const uint64_t remaining_offset_bytes = reader.Length() - reader.GetCursor();
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        offset_count >= 2 &&
            offset_count <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 2 &&
            offset_count <= remaining_offset_bytes / sizeof(uint64_t),
        "serialized SINDI host dictionary offset count is invalid");
    Vector<uint64_t> host_offsets(offset_count, allocator_);
    reader.Read(reinterpret_cast<char*>(host_offsets.data()), offset_count * sizeof(uint64_t));
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        host_offsets[0] == 0 && host_offsets[1] == 0,
        "serialized SINDI host dictionary must reserve ID zero for empty host");
    CHECK_ARGUMENT(std::is_sorted(host_offsets.begin(), host_offsets.end()),
                   "serialized SINDI host dictionary offsets must be ordered");

    uint64_t byte_count = 0;
    StreamReader::ReadObj(reader, byte_count);
    CHECK_ARGUMENT(byte_count == host_offsets.back(),
                   "serialized SINDI host dictionary byte count does not match offsets");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        reader.GetCursor() <= reader.Length() && byte_count <= reader.Length() - reader.GetCursor(),
        "serialized SINDI host dictionary exceeds the metadata payload");
    Vector<char> host_bytes(byte_count, allocator_);
    reader.Read(host_bytes.data(), byte_count);
    for (uint64_t host_id = 1; host_id + 1 < host_offsets.size(); ++host_id) {
        CHECK_ARGUMENT(host_offsets[host_id] < host_offsets[host_id + 1],
                       "serialized SINDI nonzero host dictionary entry must not be empty");
    }

    host_offsets_ = std::move(host_offsets);
    host_bytes_ = std::move(host_bytes);
    this->RebuildLookup();
}

SindiHostBuildPlan::SindiHostBuildPlan(Allocator* allocator)
    : order_(allocator),
      source_host_ids_(allocator),
      host_ids_(allocator),
      input_offsets_(allocator),
      successful_counts_(allocator) {
}

void
SindiHostBuildPlan::RecordSuccess(uint32_t ordered_position) {
    if (not enabled_) {
        return;
    }
    // Advance to the host group that contains this position in the sorted input batch.
    while (static_cast<uint64_t>(successful_host_cursor_) + 1 <
               static_cast<uint64_t>(input_offsets_.size()) and
           ordered_position >= input_offsets_[static_cast<uint64_t>(successful_host_cursor_) + 1]) {
        ++successful_host_cursor_;
    }
    ++successful_counts_[successful_host_cursor_];
}

SindiHostFilter::SindiHostFilter(Allocator* allocator)
    : host_dictionary_(allocator),
      host_ids_(allocator),
      host_range_offsets_(allocator),
      host_ranges_(allocator) {
}

SindiHostBuildPlan
SindiHostFilter::PrepareBuild(const DatasetPtr& base, uint64_t current_element_count) const {
    SindiHostBuildPlan plan(host_ids_.get_allocator().allocator_);
    CHECK_ARGUMENT(base->GetUInt32Metadata(SINDI_LEGACY_HOST_METADATA_NAME) == nullptr,
                   "numeric SINDI host_id metadata is unsupported; use string metadata host");
    const auto* source_hosts = base->GetStringMetadata(SINDI_HOST_METADATA_NAME);
    if (source_hosts == nullptr) {
        CHECK_ARGUMENT(not this->HasMetadata(), "SINDI host-aware Add requires host metadata");
        return plan;
    }

    if (current_element_count != 0) {
        CHECK_ARGUMENT(this->HasMetadata(),
                       "SINDI cannot add host metadata after host-unaware documents");
    }
    const auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(current_element_count + static_cast<uint64_t>(data_num) <=
                       std::numeric_limits<uint32_t>::max(),
                   "SINDI host-filtered build exceeds uint32_t document capacity");

    plan.enabled_ = true;
    host_dictionary_.Encode(
        source_hosts, static_cast<uint64_t>(data_num), plan.source_host_ids_, plan.new_hosts_);
    const auto* source_host_ids = plan.source_host_ids_.data();
    plan.order_.resize(static_cast<uint64_t>(data_num));
    std::iota(plan.order_.begin(), plan.order_.end(), 0);
    std::sort(
        plan.order_.begin(), plan.order_.end(), [source_host_ids](uint32_t lhs, uint32_t rhs) {
            if (source_host_ids[lhs] != source_host_ids[rhs]) {
                return source_host_ids[lhs] < source_host_ids[rhs];
            }
            return lhs < rhs;
        });

    for (uint32_t position = 0; position < plan.order_.size(); ++position) {
        const auto host_id = source_host_ids[plan.order_[position]];
        if (plan.host_ids_.empty() or plan.host_ids_.back() != host_id) {
            plan.host_ids_.push_back(host_id);
            plan.input_offsets_.push_back(position);
        }
    }
    plan.input_offsets_.push_back(static_cast<uint32_t>(data_num));
    plan.successful_counts_.resize(plan.host_ids_.size(), 0);
    return plan;
}

void
SindiHostFilter::CommitBuild(SindiHostBuildPlan&& plan,
                             uint32_t first_inner_id,
                             uint32_t end_inner_id) {
    if (not plan.Enabled()) {
        if (first_inner_id == 0) {
            this->Clear();
        }
        return;
    }

    uint32_t next_inner_id = first_inner_id;
    for (uint32_t i = 0; i < plan.successful_counts_.size(); ++i) {
        plan.input_offsets_[i] = next_inner_id;
        next_inner_id += plan.successful_counts_[i];
    }
    CHECK_ARGUMENT(next_inner_id == end_inner_id,
                   "SINDI host metadata count does not match inserted documents");

    host_dictionary_.Commit(plan.new_hosts_);

    Vector<uint32_t> merged_host_ids(host_ids_.get_allocator().allocator_);
    Vector<uint32_t> merged_range_offsets(host_ids_.get_allocator().allocator_);
    Vector<SindiHostRange> merged_ranges(host_ids_.get_allocator().allocator_);
    merged_host_ids.reserve(host_ids_.size() + plan.host_ids_.size());
    merged_range_offsets.reserve(host_ids_.size() + plan.host_ids_.size() + 1);
    merged_ranges.reserve(host_ranges_.size() + plan.host_ids_.size());
    merged_range_offsets.push_back(0);

    uint32_t existing = 0;
    uint32_t added = 0;
    while (existing < host_ids_.size() or added < plan.host_ids_.size()) {
        const bool take_existing =
            added == plan.host_ids_.size() or
            (existing < host_ids_.size() && host_ids_[existing] < plan.host_ids_[added]);
        const bool take_added =
            existing == host_ids_.size() or
            (added < plan.host_ids_.size() && plan.host_ids_[added] < host_ids_[existing]);
        const auto host_id = take_existing ? host_ids_[existing] : plan.host_ids_[added];
        const bool has_existing = not take_added;
        const bool has_added = not take_existing;
        const auto added_count = has_added ? plan.successful_counts_[added] : 0;

        if (has_existing or added_count != 0) {
            merged_host_ids.push_back(host_id);
            if (has_existing) {
                const auto range_begin = host_range_offsets_[existing];
                const auto range_end = host_range_offsets_[existing + 1];
                merged_ranges.insert(merged_ranges.end(),
                                     host_ranges_.begin() + range_begin,
                                     host_ranges_.begin() + range_end);
            }
            if (added_count != 0) {
                const auto begin = plan.input_offsets_[added];
                if (has_existing && merged_ranges.back().end == begin) {
                    merged_ranges.back().end += added_count;
                } else {
                    merged_ranges.push_back({begin, begin + added_count});
                }
            }
            merged_range_offsets.push_back(static_cast<uint32_t>(merged_ranges.size()));
        }
        if (has_existing) {
            ++existing;
        }
        if (has_added) {
            ++added;
        }
    }

    host_ids_ = std::move(merged_host_ids);
    host_range_offsets_ = std::move(merged_range_offsets);
    host_ranges_ = std::move(merged_ranges);
}

void
SindiHostFilter::Clear() {
    host_dictionary_.Clear();
    auto* allocator = host_ids_.get_allocator().allocator_;
    Vector<uint32_t>(allocator).swap(host_ids_);
    Vector<uint32_t>(allocator).swap(host_range_offsets_);
    Vector<SindiHostRange>(allocator).swap(host_ranges_);
}

SindiHostSearchRoute
SindiHostFilter::Classify(const DatasetPtr& query) const {
    CHECK_ARGUMENT(query->GetUInt32Metadata(SINDI_LEGACY_HOST_METADATA_NAME) == nullptr,
                   "numeric SINDI host_id metadata is unsupported; use string metadata host");
    const auto* query_host = query->GetStringMetadata(SINDI_HOST_METADATA_NAME);
    if (host_ids_.empty() or query_host == nullptr) {
        return {};
    }
    uint32_t query_host_id = 0;
    if (not host_dictionary_.Lookup(query_host[0], query_host_id)) {
        return {SindiHostRouteKind::EMPTY, 0, 0};
    }
    const auto host = std::lower_bound(host_ids_.begin(), host_ids_.end(), query_host_id);
    if (host == host_ids_.end() or *host != query_host_id) {
        return {SindiHostRouteKind::EMPTY, 0, 0};
    }
    const auto host_index = static_cast<uint32_t>(host - host_ids_.begin());
    const auto range_begin = host_range_offsets_[host_index];
    const auto range_end = host_range_offsets_[host_index + 1];
    return {SindiHostRouteKind::WINDOW,
            host_ranges_[range_begin].begin,
            host_ranges_[range_end - 1].end,
            host_index};
}

void
SindiHostFilter::ApplyFilter(const SindiHostSearchRoute& route, FilterPtr& filter) const {
    if (route.kind != SindiHostRouteKind::WINDOW) {
        return;
    }
    filter = std::make_shared<InnerIdHostFilter>(&host_ranges_,
                                                 host_range_offsets_[route.host_index],
                                                 host_range_offsets_[route.host_index + 1],
                                                 std::move(filter));
}

void
SindiHostFilter::ApplyWindowRoute(const SindiHostSearchRoute& route,
                                  uint32_t window_size,
                                  int64_t& min_window_id,
                                  int64_t& max_window_id) {
    if (route.kind != SindiHostRouteKind::WINDOW) {
        return;
    }
    min_window_id = std::max<int64_t>(min_window_id, route.begin / window_size);
    max_window_id = std::min<int64_t>(max_window_id, (route.end - 1) / window_size);
}

int64_t
SindiHostFilter::NextMatchingWindow(const SindiHostSearchRoute& route,
                                    uint32_t window_size,
                                    int64_t current_window_id,
                                    int64_t max_window_id) const {
    if (route.kind != SindiHostRouteKind::WINDOW) {
        return current_window_id;
    }

    const auto range_begin = host_range_offsets_[route.host_index];
    const auto range_end = host_range_offsets_[route.host_index + 1];
    const auto first = host_ranges_.begin() + range_begin;
    const auto last = host_ranges_.begin() + range_end;
    const auto window_begin = static_cast<uint64_t>(current_window_id) * window_size;
    const auto range = std::upper_bound(
        first, last, window_begin, [](uint64_t value, const SindiHostRange& candidate) {
            return value < static_cast<uint64_t>(candidate.begin);
        });
    if (range != first and std::prev(range)->end > window_begin) {
        return current_window_id;
    }
    if (range == last) {
        return max_window_id + 1;
    }
    const auto next_window_id = static_cast<int64_t>(range->begin / window_size);
    return next_window_id <= max_window_id ? std::max(current_window_id, next_window_id)
                                           : max_window_id + 1;
}

bool
SindiHostFilter::RequiresFullTermScan(const SindiHostSearchRoute& route,
                                      uint32_t window_id,
                                      uint32_t window_size) const {
    if (route.kind != SindiHostRouteKind::WINDOW) {
        return false;
    }
    const auto window_begin = static_cast<uint64_t>(window_id) * window_size;
    const auto window_end = window_begin + window_size;
    const auto range_begin = host_range_offsets_[route.host_index];
    const auto range_end = host_range_offsets_[route.host_index + 1];
    const auto range = std::upper_bound(host_ranges_.begin() + range_begin,
                                        host_ranges_.begin() + range_end,
                                        window_begin,
                                        [](uint64_t value, const SindiHostRange& candidate) {
                                            return value < static_cast<uint64_t>(candidate.begin);
                                        });
    if (range == host_ranges_.begin() + range_begin) {
        return true;
    }
    const auto& candidate = *std::prev(range);
    return candidate.begin > window_begin or candidate.end < window_end;
}

void
SindiHostFilter::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, SINDI_HOST_METADATA_MAGIC);
    StreamWriter::WriteObj(writer, SINDI_HOST_METADATA_FORMAT_VERSION);
    host_dictionary_.Serialize(writer);
    StreamWriter::WriteVector(writer, host_ids_);
    StreamWriter::WriteVector(writer, host_range_offsets_);
    const uint64_t range_count = host_ranges_.size();
    StreamWriter::WriteObj(writer, range_count);
    for (const auto& range : host_ranges_) {
        StreamWriter::WriteObj(writer, range.begin);
        StreamWriter::WriteObj(writer, range.end);
    }
}

void
SindiHostFilter::Deserialize(StreamReader& reader, uint64_t element_count) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        element_count > 0 && element_count <= std::numeric_limits<uint32_t>::max(),
        fmt::format("serialized SINDI host metadata element count must be in [1, {}], got {}",
                    std::numeric_limits<uint32_t>::max(),
                    element_count));

    auto* allocator = host_ids_.get_allocator().allocator_;
    uint32_t magic = 0;
    uint32_t version = 0;
    StreamReader::ReadObj(reader, magic);
    StreamReader::ReadObj(reader, version);
    CHECK_ARGUMENT(magic == SINDI_HOST_METADATA_MAGIC,
                   "serialized SINDI host metadata uses the unsupported numeric format");
    CHECK_ARGUMENT(version == SINDI_HOST_METADATA_FORMAT_VERSION,
                   fmt::format("unsupported SINDI host metadata version {}", version));
    SindiHostDictionary host_dictionary(allocator);
    host_dictionary.Deserialize(reader, element_count);
    Vector<uint32_t> host_ids(allocator);
    Vector<uint32_t> range_offsets(allocator);
    Vector<SindiHostRange> ranges(allocator);

    uint64_t host_count = 0;
    StreamReader::ReadObj(reader, host_count);
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        host_count > 0 && host_count <= element_count,
        fmt::format(
            "serialized SINDI host count must be in [1, {}], got {}", element_count, host_count));
    host_ids.resize(host_count);
    reader.Read(reinterpret_cast<char*>(host_ids.data()), host_count * sizeof(uint32_t));
    CHECK_ARGUMENT(
        std::adjacent_find(host_ids.begin(),
                           host_ids.end(),
                           [](uint32_t lhs, uint32_t rhs) { return lhs >= rhs; }) == host_ids.end(),
        "serialized SINDI host IDs must be strictly ordered");
    CHECK_ARGUMENT(host_dictionary.Contains(host_ids.back()),
                   "serialized SINDI host ID exceeds the host dictionary");

    uint64_t offset_count = 0;
    StreamReader::ReadObj(reader, offset_count);
    CHECK_ARGUMENT(offset_count == host_count + 1,
                   fmt::format("serialized SINDI host range offset count must be {}, got {}",
                               host_count + 1,
                               offset_count));
    range_offsets.resize(offset_count);
    reader.Read(reinterpret_cast<char*>(range_offsets.data()), offset_count * sizeof(uint32_t));

    uint64_t range_count = 0;
    StreamReader::ReadObj(reader, range_count);
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        range_count >= host_count && range_count <= element_count,
        fmt::format("serialized SINDI host range count must be in [{}, {}], got {}",
                    host_count,
                    element_count,
                    range_count));
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        range_offsets.front() == 0 && range_offsets.back() == range_count,
        fmt::format("serialized SINDI host range offsets must start at 0 and end at {}, got [{}, "
                    "{}]",
                    range_count,
                    range_offsets.front(),
                    range_offsets.back()));
    CHECK_ARGUMENT(std::adjacent_find(range_offsets.begin(),
                                      range_offsets.end(),
                                      [](uint32_t lhs, uint32_t rhs) { return lhs >= rhs; }) ==
                       range_offsets.end(),
                   "serialized SINDI host range offsets must be strictly ordered");

    ranges.resize(range_count);
    for (auto& range : ranges) {
        StreamReader::ReadObj(reader, range.begin);
        StreamReader::ReadObj(reader, range.end);
        CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
            range.begin < range.end && range.end <= element_count,
            fmt::format("serialized SINDI host range [{}, {}) is invalid for {} elements",
                        range.begin,
                        range.end,
                        element_count));
    }
    for (uint64_t host = 0; host < host_count; ++host) {
        const auto begin = range_offsets[host];
        const auto end = range_offsets[host + 1];
        for (uint32_t range = begin + 1; range < end; ++range) {
            CHECK_ARGUMENT(ranges[range - 1].end < ranges[range].begin,
                           "serialized SINDI ranges for one host must be ordered and disjoint");
        }
    }

    Vector<SindiHostRange> ranges_by_inner_id(ranges, allocator);
    std::sort(
        ranges_by_inner_id.begin(),
        ranges_by_inner_id.end(),
        [](const SindiHostRange& lhs, const SindiHostRange& rhs) { return lhs.begin < rhs.begin; });
    uint32_t next_inner_id = 0;
    for (const auto& range : ranges_by_inner_id) {
        CHECK_ARGUMENT(range.begin == next_inner_id,
                       fmt::format("serialized SINDI host ranges expected next inner ID {}, got {}",
                                   next_inner_id,
                                   range.begin));
        next_inner_id = range.end;
    }
    CHECK_ARGUMENT(next_inner_id == element_count,
                   fmt::format("serialized SINDI host ranges must cover {} elements, covered {}",
                               element_count,
                               next_inner_id));

    host_dictionary_ = std::move(host_dictionary);
    host_ids_ = std::move(host_ids);
    host_range_offsets_ = std::move(range_offsets);
    host_ranges_ = std::move(ranges);
}

namespace {

bool
contains_value(const std::vector<std::pair<uint32_t, uint32_t>>& ranges, uint32_t value) {
    const auto iter = std::upper_bound(
        ranges.begin(), ranges.end(), value, [](uint32_t candidate, const auto& r) {
            return candidate < r.first;
        });
    return iter != ranges.begin() and value < std::prev(iter)->second;
}

void
append_merged_range(std::vector<std::pair<uint32_t, uint32_t>>& ranges,
                    uint32_t begin,
                    uint32_t end) {
    if (begin >= end) {
        return;
    }
    if (not ranges.empty() and begin <= ranges.back().second) {
        ranges.back().second = std::max(ranges.back().second, end);
        return;
    }
    ranges.emplace_back(begin, end);
}

class TimeRouteFilter : public Filter {
public:
    TimeRouteFilter(const SindiTimeSearchRoute& route,
                    const Vector<int32_t>* document_days,
                    FilterPtr filter)
        : inner_ranges_(&route.inner_ranges),
          has_time_(route.has_time),
          query_begin_(route.query_begin),
          query_end_(route.query_end),
          document_days_(document_days),
          filter_(std::move(filter)) {
    }

    [[nodiscard]] bool
    CheckValid(int64_t id) const override {
        if (id < 0 or id > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const auto inner_id = static_cast<uint32_t>(id);
        if (not contains_value(*inner_ranges_, inner_id)) {
            return false;
        }
        if (has_time_ and inner_id >= document_days_->size()) {
            return false;
        }
        if (has_time_ and ((*document_days_)[inner_id] < query_begin_ or
                           (*document_days_)[inner_id] > query_end_)) {
            return false;
        }
        return filter_ == nullptr or filter_->CheckValid(id);
    }

    void
    GetValidIds(const int64_t** valid_ids, int64_t& count) const override {
        if (filter_ != nullptr) {
            filter_->GetValidIds(valid_ids, count);
        }
    }

    [[nodiscard]] float
    ValidRatio() const override {
        return filter_ == nullptr ? 1.0F : filter_->ValidRatio();
    }

    [[nodiscard]] Distribution
    FilterDistribution() const override {
        return filter_ == nullptr ? Distribution::NONE : filter_->FilterDistribution();
    }

private:
    // Borrows route.inner_ranges and must not outlive the owning search-local route.
    const std::vector<std::pair<uint32_t, uint32_t>>* inner_ranges_{nullptr};
    bool has_time_{false};
    int32_t query_begin_{0};
    int32_t query_end_{0};
    const Vector<int32_t>* document_days_{nullptr};
    FilterPtr filter_;
};

template <typename T>
void
read_vector(StreamReader& reader, Vector<T>& values, uint64_t max_count, const char* name) {
    uint64_t count = 0;
    StreamReader::ReadObj(reader, count);
    CHECK_ARGUMENT(count <= max_count, fmt::format("serialized SINDI {} count is invalid", name));
    values.resize(count);
    if (count > 0) {
        reader.Read(reinterpret_cast<char*>(values.data()), count * sizeof(T));
    }
}

}  // namespace

SindiTimeBuildPlan::SindiTimeBuildPlan(Allocator* allocator)
    : order_(allocator),
      source_days_(allocator),
      source_host_ids_(allocator),
      group_partitions_(allocator),
      group_hosts_(allocator),
      input_offsets_(allocator),
      successful_counts_(allocator),
      successful_days_(allocator) {
}

void
SindiTimeBuildPlan::RecordSuccess(uint32_t ordered_position) {
    if (not enabled_) {
        return;
    }
    while (static_cast<uint64_t>(successful_group_cursor_) + 1 < input_offsets_.size() and
           ordered_position >=
               input_offsets_[static_cast<uint64_t>(successful_group_cursor_) + 1]) {
        ++successful_group_cursor_;
    }
    ++successful_counts_[successful_group_cursor_];
    successful_days_.push_back(source_days_[order_[ordered_position]]);
}

SindiTimeFilter::SindiTimeFilter(Allocator* allocator)
    : allocator_(allocator),
      host_dictionary_(allocator),
      document_days_(allocator),
      partitions_(allocator) {
}

SindiTimeBuildPlan
SindiTimeFilter::PrepareBuild(const DatasetPtr& base, uint32_t window_size) const {
    SindiTimeBuildPlan plan(allocator_);
    CHECK_ARGUMENT(base->GetUInt32Metadata(SINDI_LEGACY_HOST_METADATA_NAME) == nullptr,
                   "numeric SINDI host_id metadata is unsupported; use string metadata host");
    const auto* source_timestamps = base->GetInt64Metadata(SINDI_PUBLISH_TIME_METADATA_NAME);
    if (source_timestamps == nullptr) {
        return plan;
    }
    CHECK_ARGUMENT(window_size > 0, "SINDI window_size must be greater than zero");

    const auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                   "SINDI time-filtered build exceeds uint32_t document capacity");

    const auto* source_hosts = base->GetStringMetadata(SINDI_HOST_METADATA_NAME);
    plan.enabled_ = true;
    plan.has_host_metadata_ = source_hosts != nullptr;
    if (source_hosts != nullptr) {
        host_dictionary_.Encode(
            source_hosts, static_cast<uint64_t>(data_num), plan.source_host_ids_, plan.new_hosts_);
    }
    const auto* source_host_ids = plan.has_host_metadata_ ? plan.source_host_ids_.data() : nullptr;
    plan.order_.resize(static_cast<uint64_t>(data_num));
    plan.source_days_.resize(static_cast<uint64_t>(data_num));
    std::iota(plan.order_.begin(), plan.order_.end(), 0);
    for (uint32_t i = 0; i < static_cast<uint32_t>(data_num); ++i) {
        const auto timestamp = source_timestamps[i];
        CHECK_ARGUMENT(timestamp >= 0, "SINDI publish_time_stamp base values must be non-negative");
        if (timestamp == 0) {
            plan.source_days_[i] = SINDI_MISSING_EPOCH_DAY;
            continue;
        }
        const auto day = timestamp / SINDI_SECONDS_PER_DAY;
        CHECK_ARGUMENT(day <= std::numeric_limits<int32_t>::max(),
                       "SINDI publish_time_stamp exceeds the supported epoch day range");
        plan.source_days_[i] = static_cast<int32_t>(day);
    }

    std::sort(plan.order_.begin(), plan.order_.end(), [&](uint32_t lhs, uint32_t rhs) {
        const bool lhs_missing = plan.source_days_[lhs] == SINDI_MISSING_EPOCH_DAY;
        const bool rhs_missing = plan.source_days_[rhs] == SINDI_MISSING_EPOCH_DAY;
        if (lhs_missing != rhs_missing) {
            return not lhs_missing;
        }
        if (not lhs_missing and source_timestamps[lhs] != source_timestamps[rhs]) {
            return source_timestamps[lhs] < source_timestamps[rhs];
        }
        return lhs < rhs;
    });

    const auto dated_end =
        std::find_if(plan.order_.begin(), plan.order_.end(), [&](uint32_t source) {
            return plan.source_days_[source] == SINDI_MISSING_EPOCH_DAY;
        });
    const auto dated_count = static_cast<uint32_t>(dated_end - plan.order_.begin());
    const auto dated_partition_count = dated_count == 0 ? 0 : (dated_count - 1) / window_size + 1;
    Vector<uint32_t> source_partitions(static_cast<uint64_t>(data_num), allocator_);
    for (uint32_t position = 0; position < dated_count; ++position) {
        source_partitions[plan.order_[position]] = position / window_size;
    }
    for (uint32_t position = dated_count; position < static_cast<uint32_t>(data_num); ++position) {
        source_partitions[plan.order_[position]] = dated_partition_count;
    }

    const auto sort_partition = [&](uint32_t begin, uint32_t end) {
        std::sort(
            plan.order_.begin() + begin,
            plan.order_.begin() + end,
            [&](uint32_t lhs, uint32_t rhs) {
                if (source_host_ids != nullptr and source_host_ids[lhs] != source_host_ids[rhs]) {
                    return source_host_ids[lhs] < source_host_ids[rhs];
                }
                return lhs < rhs;
            });
    };
    for (uint64_t begin = 0; begin < dated_count; begin += window_size) {
        const auto end = std::min<uint64_t>(dated_count, begin + window_size);
        sort_partition(static_cast<uint32_t>(begin), static_cast<uint32_t>(end));
    }
    if (dated_count < static_cast<uint32_t>(data_num)) {
        sort_partition(dated_count, static_cast<uint32_t>(data_num));
    }

    for (uint32_t position = 0; position < plan.order_.size(); ++position) {
        const auto source = plan.order_[position];
        const auto partition = source_partitions[source];
        const auto host = source_host_ids == nullptr ? 0 : source_host_ids[source];
        if (plan.group_partitions_.empty() or plan.group_partitions_.back() != partition or
            plan.group_hosts_.back() != host) {
            plan.group_partitions_.push_back(partition);
            plan.group_hosts_.push_back(host);
            plan.input_offsets_.push_back(position);
        }
    }
    plan.input_offsets_.push_back(static_cast<uint32_t>(data_num));
    plan.successful_counts_.resize(plan.group_partitions_.size(), 0);
    plan.successful_days_.reserve(static_cast<uint64_t>(data_num));
    return plan;
}

void
SindiTimeFilter::CommitBuild(SindiTimeBuildPlan&& plan, uint64_t element_count) {
    this->Clear();
    if (not plan.Enabled()) {
        return;
    }
    CHECK_ARGUMENT(plan.successful_days_.size() == element_count,
                   "SINDI successful epoch day count does not match element count");

    has_host_metadata_ = plan.has_host_metadata_;
    if (has_host_metadata_) {
        host_dictionary_.Commit(plan.new_hosts_);
    }
    document_days_ = std::move(plan.successful_days_);
    uint32_t inner_cursor = 0;
    for (uint64_t group_begin = 0; group_begin < plan.group_partitions_.size();) {
        uint64_t group_end = group_begin + 1;
        while (group_end < plan.group_partitions_.size() and
               plan.group_partitions_[group_end] == plan.group_partitions_[group_begin]) {
            ++group_end;
        }
        uint32_t partition_count = 0;
        for (uint64_t group = group_begin; group < group_end; ++group) {
            partition_count += plan.successful_counts_[group];
        }
        if (partition_count > 0) {
            partitions_.emplace_back(allocator_);
            auto& partition = partitions_.back();
            partition.begin = inner_cursor;
            partition.end = inner_cursor + partition_count;
            const auto day_range = std::minmax_element(document_days_.begin() + partition.begin,
                                                       document_days_.begin() + partition.end);
            partition.min_day = *day_range.first;
            partition.max_day = *day_range.second;
            CHECK_ARGUMENT((partition.min_day == SINDI_MISSING_EPOCH_DAY) ==
                               (partition.max_day == SINDI_MISSING_EPOCH_DAY),
                           "SINDI missing timestamps must be isolated in the final partition");
            if (has_host_metadata_) {
                partition.host_offsets.push_back(0);
                uint32_t local_offset = 0;
                for (uint64_t group = group_begin; group < group_end; ++group) {
                    const auto successful_count = plan.successful_counts_[group];
                    if (successful_count == 0) {
                        continue;
                    }
                    partition.host_ids.push_back(plan.group_hosts_[group]);
                    local_offset += successful_count;
                    partition.host_offsets.push_back(local_offset);
                }
            }
            inner_cursor = partition.end;
        }
        group_begin = group_end;
    }
    CHECK_ARGUMENT(inner_cursor == element_count,
                   "SINDI time partitions do not cover every indexed document");
}

void
SindiTimeFilter::Clear() {
    host_dictionary_.Clear();
    Vector<int32_t>(allocator_).swap(document_days_);
    Vector<Partition>(allocator_).swap(partitions_);
    has_host_metadata_ = false;
}

uint64_t
SindiTimeFilter::GetMemoryUsage() const {
    uint64_t memory = host_dictionary_.GetMemoryUsage();
    memory += document_days_.size() * sizeof(int32_t);
    memory += partitions_.size() * sizeof(Partition);
    for (const auto& partition : partitions_) {
        memory += (partition.host_ids.size() + partition.host_offsets.size()) * sizeof(uint32_t);
    }
    return memory;
}

SindiTimeSearchRoute
SindiTimeFilter::Classify(const DatasetPtr& query, uint32_t window_size) const {
    CHECK_ARGUMENT(query->GetUInt32Metadata(SINDI_LEGACY_HOST_METADATA_NAME) == nullptr,
                   "numeric SINDI host_id metadata is unsupported; use string metadata host");
    SindiTimeSearchRoute route;
    if (partitions_.empty()) {
        return route;
    }
    route.enabled = true;
    CHECK_ARGUMENT(window_size > 0, "SINDI window_size must be greater than zero");

    const auto* query_time = query->GetInt64Metadata(SINDI_PUBLISH_TIME_METADATA_NAME);
    const auto* query_begin = query->GetInt64Metadata(SINDI_PUBLISH_TIME_BEGIN_METADATA_NAME);
    const auto* query_end = query->GetInt64Metadata(SINDI_PUBLISH_TIME_END_METADATA_NAME);
    CHECK_ARGUMENT((query_begin == nullptr) == (query_end == nullptr),
                   "SINDI publish_time_stamp_begin and publish_time_stamp_end must be provided "
                   "together");
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        not(query_time != nullptr and query_begin != nullptr),
        "SINDI publish_time_stamp cannot be combined with a time range");
    const auto to_day = [](int64_t timestamp) {
        CHECK_ARGUMENT(timestamp > 0, "SINDI query timestamps must be greater than zero");
        const auto day = timestamp / SINDI_SECONDS_PER_DAY;
        CHECK_ARGUMENT(day <= std::numeric_limits<int32_t>::max(),
                       "SINDI query timestamp exceeds the supported epoch day range");
        return static_cast<int32_t>(day);
    };
    if (query_time != nullptr) {
        route.has_time = true;
        route.query_begin = to_day(query_time[0]);
        route.query_end = route.query_begin;
    } else if (query_begin != nullptr) {
        CHECK_ARGUMENT(query_begin[0] <= query_end[0],
                       "SINDI publish_time_stamp_begin must not exceed publish_time_stamp_end");
        route.has_time = true;
        route.query_begin = to_day(query_begin[0]);
        route.query_end = to_day(query_end[0]);
    }

    const auto* query_host = query->GetStringMetadata(SINDI_HOST_METADATA_NAME);
    const bool use_host = has_host_metadata_ and query_host != nullptr;
    uint32_t query_host_id = 0;
    if (use_host and not host_dictionary_.Lookup(query_host[0], query_host_id)) {
        route.kind = SindiHostRouteKind::EMPTY;
        return route;
    }
    if (not route.has_time and not use_host) {
        return route;
    }

    auto partition_begin = partitions_.begin();
    auto partition_end = partitions_.end();
    if (route.has_time) {
        partition_end =
            std::find_if(partition_begin, partition_end, [](const Partition& partition) {
                return partition.min_day == SINDI_MISSING_EPOCH_DAY;
            });
        partition_begin = std::lower_bound(
            partition_begin,
            partition_end,
            route.query_begin,
            [](const Partition& partition, int32_t day) { return partition.max_day < day; });
    }
    route.inner_ranges.reserve(static_cast<uint64_t>(partition_end - partition_begin));
    for (auto partition_iter = partition_begin; partition_iter != partition_end; ++partition_iter) {
        const auto& partition = *partition_iter;
        if (route.has_time and partition.min_day > route.query_end) {
            break;
        }
        uint32_t begin = partition.begin;
        uint32_t end = partition.end;
        if (use_host) {
            const auto host = std::lower_bound(
                partition.host_ids.begin(), partition.host_ids.end(), query_host_id);
            if (host == partition.host_ids.end() or *host != query_host_id) {
                continue;
            }
            const auto host_index = static_cast<uint64_t>(host - partition.host_ids.begin());
            begin += partition.host_offsets[host_index];
            end = partition.begin + partition.host_offsets[host_index + 1];
        }
        append_merged_range(route.inner_ranges, begin, end);
    }

    if (route.inner_ranges.empty()) {
        route.kind = SindiHostRouteKind::EMPTY;
        return route;
    }
    route.window_ranges.reserve(route.inner_ranges.size());
    for (const auto& [begin, end] : route.inner_ranges) {
        append_merged_range(route.window_ranges, begin / window_size, (end - 1) / window_size + 1);
    }
    route.kind = SindiHostRouteKind::WINDOW;
    return route;
}

FilterPtr
SindiTimeFilter::create_filter(const SindiTimeSearchRoute& route, FilterPtr filter) const {
    return std::make_shared<TimeRouteFilter>(route, &document_days_, std::move(filter));
}

void
SindiTimeFilter::ApplyFilter(const SindiTimeSearchRoute& route, FilterPtr& filter) const {
    if (not route.enabled or route.kind != SindiHostRouteKind::WINDOW) {
        return;
    }
    filter = create_filter(route, std::move(filter));
}

void
SindiTimeFilter::ApplyWindowRoute(const SindiTimeSearchRoute& route,
                                  int64_t& min_window_id,
                                  int64_t& max_window_id) {
    if (not route.enabled or route.kind != SindiHostRouteKind::WINDOW) {
        return;
    }
    min_window_id = std::max<int64_t>(min_window_id, route.window_ranges.front().first);
    max_window_id = std::min<int64_t>(max_window_id, route.window_ranges.back().second - 1);
}

int64_t
SindiTimeFilter::NextMatchingWindow(const SindiTimeSearchRoute& route,
                                    int64_t current_window_id,
                                    int64_t max_window_id) {
    if (not route.enabled or route.kind != SindiHostRouteKind::WINDOW) {
        return current_window_id;
    }
    const auto current = static_cast<uint32_t>(current_window_id);
    const auto range = std::upper_bound(
        route.window_ranges.begin(),
        route.window_ranges.end(),
        current,
        [](uint32_t value, const auto& candidate) { return value < candidate.first; });
    if (range != route.window_ranges.begin() and current < std::prev(range)->second) {
        return current_window_id;
    }
    if (range == route.window_ranges.end()) {
        return max_window_id + 1;
    }
    return range->first <= max_window_id ? std::max<int64_t>(current_window_id, range->first)
                                         : max_window_id + 1;
}

bool
SindiTimeFilter::RequiresFullTermScan(const SindiTimeSearchRoute& route,
                                      uint32_t window_id,
                                      uint32_t window_size) {
    if (not route.enabled or route.kind != SindiHostRouteKind::WINDOW) {
        return false;
    }
    if (route.has_time) {
        return true;
    }
    // A host-only route can include partial windows at time-partition or host boundaries.
    const uint64_t window_begin = static_cast<uint64_t>(window_id) * window_size;
    const uint64_t window_end = window_begin + window_size;
    const auto range = std::upper_bound(
        route.inner_ranges.begin(),
        route.inner_ranges.end(),
        window_begin,
        [](uint64_t value, const auto& candidate) { return value < candidate.first; });
    if (range == route.inner_ranges.begin()) {
        return true;
    }
    const auto& candidate = *std::prev(range);
    return candidate.first > window_begin or candidate.second < window_end;
}

void
SindiTimeFilter::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, SINDI_TIME_METADATA_FORMAT_VERSION);
    StreamWriter::WriteObj(writer, static_cast<uint32_t>(has_host_metadata_));
    if (has_host_metadata_) {
        host_dictionary_.Serialize(writer);
    }
    StreamWriter::WriteVector(writer, document_days_);
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(partitions_.size()));
    for (const auto& partition : partitions_) {
        StreamWriter::WriteObj(writer, partition.min_day);
        StreamWriter::WriteObj(writer, partition.max_day);
        StreamWriter::WriteObj(writer, partition.begin);
        StreamWriter::WriteObj(writer, partition.end);
        StreamWriter::WriteVector(writer, partition.host_ids);
        StreamWriter::WriteVector(writer, partition.host_offsets);
    }
}

void
SindiTimeFilter::Deserialize(StreamReader& reader, uint64_t element_count) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        element_count > 0 && element_count <= std::numeric_limits<uint32_t>::max(),
        fmt::format("serialized SINDI time metadata element count must be in [1, {}], got {}",
                    std::numeric_limits<uint32_t>::max(),
                    element_count));

    uint32_t version = 0;
    uint32_t has_host = 0;
    StreamReader::ReadObj(reader, version);
    StreamReader::ReadObj(reader, has_host);
    CHECK_ARGUMENT(has_host <= 1, "serialized SINDI time host flag is invalid");
    CHECK_ARGUMENT(IsSupportedSindiTimeMetadataVersion(version),
                   fmt::format("unsupported SINDI time metadata version {}", version));

    SindiHostDictionary host_dictionary(allocator_);
    if (has_host != 0) {
        host_dictionary.Deserialize(reader, element_count);
    }

    Vector<int32_t> document_days(allocator_);
    read_vector(reader, document_days, element_count, "document epoch day");
    CHECK_ARGUMENT(document_days.size() == element_count,
                   "serialized SINDI document epoch day count does not match element count");
    uint64_t partition_count = 0;
    StreamReader::ReadObj(reader, partition_count);
    CHECK_ARGUMENT(partition_count > 0, "serialized SINDI time partition count is invalid");
    CHECK_ARGUMENT(partition_count <= element_count,
                   "serialized SINDI time partition count exceeds element count");

    Vector<Partition> partitions(allocator_);
    partitions.reserve(partition_count);
    int32_t previous_max_day = SINDI_MISSING_EPOCH_DAY;
    bool seen_dated_partition = false;
    uint32_t inner_cursor = 0;
    for (uint64_t i = 0; i < partition_count; ++i) {
        partitions.emplace_back(allocator_);
        auto& partition = partitions.back();
        StreamReader::ReadObj(reader, partition.min_day);
        StreamReader::ReadObj(reader, partition.max_day);
        StreamReader::ReadObj(reader, partition.begin);
        StreamReader::ReadObj(reader, partition.end);
        const bool missing_partition = partition.min_day == SINDI_MISSING_EPOCH_DAY or
                                       partition.max_day == SINDI_MISSING_EPOCH_DAY;
        if (missing_partition) {
            CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
                partition.min_day == SINDI_MISSING_EPOCH_DAY and
                    partition.max_day == SINDI_MISSING_EPOCH_DAY and i + 1 == partition_count,
                "serialized SINDI missing-time partition must be last with range [-1, -1]");
        } else {
            CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
                partition.min_day >= 0 and partition.min_day <= partition.max_day,
                "serialized SINDI time partition day range is invalid");
            if (seen_dated_partition) {
                CHECK_ARGUMENT(previous_max_day <= partition.min_day,
                               "serialized SINDI time partition ranges must be ordered");
            }
            previous_max_day = partition.max_day;
            seen_dated_partition = true;
        }
        CHECK_ARGUMENT(partition.begin == inner_cursor,
                       "serialized SINDI time partitions must be contiguous");
        CHECK_ARGUMENT(partition.end > partition.begin,
                       "serialized SINDI time partition must not be empty");
        CHECK_ARGUMENT(partition.end <= element_count,
                       "serialized SINDI time partition exceeds element count");
        read_vector(reader, partition.host_ids, element_count, "partition host");
        read_vector(reader, partition.host_offsets, element_count + 1, "partition host offset");
        if (has_host != 0) {
            CHECK_ARGUMENT(not partition.host_ids.empty(),
                           "serialized SINDI partition host directory is empty");
            CHECK_ARGUMENT(partition.host_offsets.size() == partition.host_ids.size() + 1,
                           "serialized SINDI partition host directory is invalid");
            CHECK_ARGUMENT(std::adjacent_find(partition.host_ids.begin(),
                                              partition.host_ids.end(),
                                              [](uint32_t lhs, uint32_t rhs) {
                                                  return lhs >= rhs;
                                              }) == partition.host_ids.end(),
                           "serialized SINDI partition host IDs must be strictly ordered");
            CHECK_ARGUMENT(host_dictionary.Contains(partition.host_ids.back()),
                           "serialized SINDI partition host ID exceeds the host dictionary");
            CHECK_ARGUMENT(partition.host_offsets.front() == 0,
                           "serialized SINDI partition host offsets must start at zero");
            CHECK_ARGUMENT(std::adjacent_find(partition.host_offsets.begin(),
                                              partition.host_offsets.end(),
                                              [](uint32_t lhs, uint32_t rhs) {
                                                  return lhs >= rhs;
                                              }) == partition.host_offsets.end(),
                           "serialized SINDI partition host offsets must be strictly ordered");
            CHECK_ARGUMENT(partition.host_offsets.back() == partition.end - partition.begin,
                           "serialized SINDI partition host offsets must cover the partition");
        } else {
            CHECK_ARGUMENT(partition.host_ids.empty() and partition.host_offsets.empty(),
                           "serialized SINDI time metadata has unexpected host directory");
        }
        for (uint32_t inner_id = partition.begin; inner_id < partition.end; ++inner_id) {
            const auto day = document_days[inner_id];
            CHECK_ARGUMENT(missing_partition
                               ? day == SINDI_MISSING_EPOCH_DAY
                               : day >= partition.min_day and day <= partition.max_day,
                           "serialized SINDI document epoch day is outside its partition");
        }
        inner_cursor = partition.end;
    }
    CHECK_ARGUMENT(inner_cursor == element_count,
                   "serialized SINDI time partitions do not cover every indexed document");

    has_host_metadata_ = has_host != 0;
    host_dictionary_ = std::move(host_dictionary);
    document_days_ = std::move(document_days);
    partitions_ = std::move(partitions);
}

SindiMetadataFilter::SindiMetadataFilter(Allocator* allocator)
    : allocator_(allocator), host_filter_(allocator), time_filter_(allocator) {
}

SindiMetadataBuildPlan
SindiMetadataFilter::PrepareBuild(const DatasetPtr& base,
                                  uint64_t current_element_count,
                                  uint32_t window_size) const {
    SindiMetadataBuildPlan plan(allocator_);
    if (current_element_count != 0) {
        CHECK_ARGUMENT(not time_filter_.HasMetadata(),
                       "SINDI time-aware index does not support incremental Add");
    }
    plan.time_plan_ = time_filter_.PrepareBuild(base, window_size);
    if (plan.time_plan_.Enabled()) {
        CHECK_ARGUMENT(current_element_count == 0,
                       "SINDI cannot add time metadata after existing documents");
    } else {
        plan.host_plan_ = host_filter_.PrepareBuild(base, current_element_count);
    }
    return plan;
}

void
SindiMetadataFilter::CommitBuild(SindiMetadataBuildPlan&& plan,
                                 uint32_t first_inner_id,
                                 uint32_t end_inner_id) {
    if (plan.time_plan_.Enabled()) {
        host_filter_.Clear();
        time_filter_.CommitBuild(std::move(plan.time_plan_), end_inner_id);
        return;
    }
    host_filter_.CommitBuild(std::move(plan.host_plan_), first_inner_id, end_inner_id);
    if (first_inner_id == 0) {
        time_filter_.Clear();
    }
}

void
SindiMetadataFilter::Clear() {
    host_filter_.Clear();
    time_filter_.Clear();
}

SindiMetadataSearchRoute
SindiMetadataFilter::Classify(const DatasetPtr& query, uint32_t window_size) const {
    SindiMetadataSearchRoute route;
    if (time_filter_.HasMetadata()) {
        route.time_route = time_filter_.Classify(query, window_size);
        route.kind = route.time_route.kind;
    } else {
        const bool has_time_query =
            query->GetInt64Metadata(SINDI_PUBLISH_TIME_METADATA_NAME) != nullptr or
            query->GetInt64Metadata(SINDI_PUBLISH_TIME_BEGIN_METADATA_NAME) != nullptr or
            query->GetInt64Metadata(SINDI_PUBLISH_TIME_END_METADATA_NAME) != nullptr;
        CHECK_ARGUMENT(not has_time_query,
                       "SINDI time queries require an index built with publish_time_stamp");
        route.host_route = host_filter_.Classify(query);
        route.kind = route.host_route.kind;
    }
    return route;
}

void
SindiMetadataFilter::ApplyFilter(const SindiMetadataSearchRoute& route, FilterPtr& filter) const {
    if (route.time_route.enabled) {
        time_filter_.ApplyFilter(route.time_route, filter);
    } else {
        host_filter_.ApplyFilter(route.host_route, filter);
    }
}

void
SindiMetadataFilter::ApplyWindowRoute(const SindiMetadataSearchRoute& route,
                                      uint32_t window_size,
                                      int64_t& min_window_id,
                                      int64_t& max_window_id) {
    if (route.time_route.enabled) {
        SindiTimeFilter::ApplyWindowRoute(route.time_route, min_window_id, max_window_id);
    } else {
        SindiHostFilter::ApplyWindowRoute(
            route.host_route, window_size, min_window_id, max_window_id);
    }
}

int64_t
SindiMetadataFilter::NextMatchingWindow(const SindiMetadataSearchRoute& route,
                                        uint32_t window_size,
                                        int64_t current_window_id,
                                        int64_t max_window_id) const {
    if (route.time_route.enabled) {
        return SindiTimeFilter::NextMatchingWindow(
            route.time_route, current_window_id, max_window_id);
    }
    return host_filter_.NextMatchingWindow(
        route.host_route, window_size, current_window_id, max_window_id);
}

bool
SindiMetadataFilter::RequiresFullTermScan(const SindiMetadataSearchRoute& route,
                                          uint32_t window_id,
                                          uint32_t window_size) const {
    if (route.time_route.enabled) {
        return SindiTimeFilter::RequiresFullTermScan(route.time_route, window_id, window_size);
    }
    return host_filter_.RequiresFullTermScan(route.host_route, window_id, window_size);
}

void
SindiMetadataFilter::DeserializeHostMetadata(StreamReader& reader, uint64_t element_count) {
    host_filter_.Deserialize(reader, element_count);
    time_filter_.Clear();
}

void
SindiMetadataFilter::DeserializeTimeMetadata(StreamReader& reader, uint64_t element_count) {
    time_filter_.Deserialize(reader, element_count);
    host_filter_.Clear();
}

}  // namespace vsag
