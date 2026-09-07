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

SindiHostBuildPlan::SindiHostBuildPlan(Allocator* allocator)
    : order_(allocator),
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
    : host_ids_(allocator), host_range_offsets_(allocator), host_ranges_(allocator) {
}

SindiHostBuildPlan
SindiHostFilter::PrepareBuild(const DatasetPtr& base, uint64_t current_element_count) const {
    SindiHostBuildPlan plan(host_ids_.get_allocator().allocator_);
    const auto* source_host_ids = base->GetUInt32Metadata(SINDI_HOST_ID_METADATA_NAME);
    if (source_host_ids == nullptr) {
        CHECK_ARGUMENT(not this->HasMetadata(), "SINDI host-aware Add requires host_id metadata");
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
    auto* allocator = host_ids_.get_allocator().allocator_;
    Vector<uint32_t>(allocator).swap(host_ids_);
    Vector<uint32_t>(allocator).swap(host_range_offsets_);
    Vector<SindiHostRange>(allocator).swap(host_ranges_);
}

SindiHostSearchRoute
SindiHostFilter::Classify(const DatasetPtr& query) const {
    const auto* query_host_id = query->GetUInt32Metadata(SINDI_HOST_ID_METADATA_NAME);
    if (host_ids_.empty() or query_host_id == nullptr) {
        return {};
    }
    const auto host = std::lower_bound(host_ids_.begin(), host_ids_.end(), query_host_id[0]);
    if (host == host_ids_.end() or *host != query_host_id[0]) {
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

    host_ids_ = std::move(host_ids);
    host_range_offsets_ = std::move(range_offsets);
    host_ranges_ = std::move(ranges);
}

namespace {

constexpr uint32_t DATE_YEAR_SHIFT = 9;
constexpr uint32_t DATE_MONTH_SHIFT = 5;
constexpr uint32_t DATE_MONTH_MASK = 0xF;
constexpr uint32_t DATE_DAY_MASK = 0x1F;
constexpr uint32_t MAX_DATE_YEAR = 9999;

uint32_t
date_year(uint32_t bucket) {
    return bucket >> DATE_YEAR_SHIFT;
}

uint32_t
date_month(uint32_t bucket) {
    return (bucket >> DATE_MONTH_SHIFT) & DATE_MONTH_MASK;
}

uint32_t
date_day(uint32_t bucket) {
    return bucket & DATE_DAY_MASK;
}

bool
is_leap_year(uint32_t year) {
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0);
}

uint32_t
days_in_month(uint32_t year, uint32_t month) {
    constexpr uint32_t days_per_month[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    return month == 2 and is_leap_year(year) ? 29 : days_per_month[month - 1];
}

uint32_t
parse_date_component(const std::string& value, uint32_t begin, uint32_t count) {
    uint32_t result = 0;
    for (uint32_t i = begin; i < begin + count; ++i) {
        const bool valid_digit = value[i] >= '0' and value[i] <= '9';
        CHECK_ARGUMENT(valid_digit, fmt::format("invalid SINDI date bucket: {}", value));
        result = result * 10 + static_cast<uint32_t>(value[i] - '0');
    }
    return result;
}

uint32_t
parse_date_bucket(const std::string& value) {
    const bool valid_size = value.size() == 4 or value.size() == 7 or value.size() == 10;
    CHECK_ARGUMENT(valid_size, fmt::format("invalid SINDI date bucket: {}", value));
    const auto year = parse_date_component(value, 0, 4);
    const bool valid_year = year > 0 and year <= MAX_DATE_YEAR;
    CHECK_ARGUMENT(valid_year, fmt::format("invalid SINDI date bucket: {}", value));
    if (value.size() == 4) {
        return year << DATE_YEAR_SHIFT;
    }

    CHECK_ARGUMENT(value[4] == '/', fmt::format("invalid SINDI date bucket: {}", value));
    const auto month = parse_date_component(value, 5, 2);
    const bool valid_month = month > 0 and month <= 12;
    CHECK_ARGUMENT(valid_month, fmt::format("invalid SINDI date bucket: {}", value));
    if (value.size() == 7) {
        return (year << DATE_YEAR_SHIFT) | (month << DATE_MONTH_SHIFT);
    }

    CHECK_ARGUMENT(value[7] == '/', fmt::format("invalid SINDI date bucket: {}", value));
    const auto day = parse_date_component(value, 8, 2);
    const bool valid_day = day > 0 and day <= days_in_month(year, month);
    CHECK_ARGUMENT(valid_day, fmt::format("invalid SINDI date bucket: {}", value));
    return (year << DATE_YEAR_SHIFT) | (month << DATE_MONTH_SHIFT) | day;
}

bool
is_valid_date_bucket(uint32_t bucket) {
    const auto year = date_year(bucket);
    const auto month = date_month(bucket);
    const auto day = date_day(bucket);
    if (year == 0 or year > MAX_DATE_YEAR or month > 12) {
        return false;
    }
    if (month == 0) {
        return day == 0;
    }
    return day == 0 or day <= days_in_month(year, month);
}

uint32_t
date_bucket_to_quarter(uint32_t bucket) {
    const auto month = date_month(bucket);
    return date_year(bucket) * 4 + ((month == 0 ? 1 : month) - 1) / 3;
}

bool
date_bucket_matches(uint32_t base_bucket, uint32_t query_bucket) {
    if (date_year(base_bucket) != date_year(query_bucket)) {
        return false;
    }
    const auto query_month = date_month(query_bucket);
    if (query_month == 0) {
        return true;
    }
    if (date_month(base_bucket) != query_month) {
        return false;
    }
    const auto query_day = date_day(query_bucket);
    return query_day == 0 or date_day(base_bucket) == query_day;
}

uint32_t
date_bucket_first_day(uint32_t bucket) {
    const auto month = date_month(bucket);
    if (month == 0) {
        return bucket | (1U << DATE_MONTH_SHIFT) | 1U;
    }
    return date_day(bucket) == 0 ? bucket | 1U : bucket;
}

uint32_t
date_bucket_last_day(uint32_t bucket) {
    const auto month = date_month(bucket);
    if (month == 0) {
        return bucket | (12U << DATE_MONTH_SHIFT) | 31U;
    }
    return date_day(bucket) == 0 ? bucket | days_in_month(date_year(bucket), month) : bucket;
}

bool
date_bucket_within_range(uint32_t bucket, uint32_t query_begin, uint32_t query_end) {
    if (date_day(bucket) != 0) {
        return bucket >= query_begin and bucket <= query_end;
    }
    return date_bucket_first_day(bucket) >= query_begin and
           date_bucket_last_day(bucket) <= query_end;
}

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

class DateRouteFilter : public Filter {
public:
    DateRouteFilter(const SindiDateSearchRoute& route,
                    const Vector<uint32_t>* document_buckets,
                    FilterPtr filter)
        : inner_ranges_(&route.inner_ranges),
          has_date_bucket_(route.has_date_bucket),
          has_date_range_(route.has_date_range),
          query_bucket_(route.query_bucket),
          query_begin_(route.query_begin),
          query_end_(route.query_end),
          document_buckets_(document_buckets),
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
        if ((has_date_bucket_ or has_date_range_) and inner_id >= document_buckets_->size()) {
            return false;
        }
        if (has_date_bucket_ and
            not date_bucket_matches((*document_buckets_)[inner_id], query_bucket_)) {
            return false;
        }
        if (has_date_range_ and not date_bucket_within_range(
                                    (*document_buckets_)[inner_id], query_begin_, query_end_)) {
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
    bool has_date_bucket_{false};
    bool has_date_range_{false};
    uint32_t query_bucket_{0};
    uint32_t query_begin_{0};
    uint32_t query_end_{0};
    const Vector<uint32_t>* document_buckets_{nullptr};
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

SindiDateBuildPlan::SindiDateBuildPlan(Allocator* allocator)
    : order_(allocator),
      source_buckets_(allocator),
      group_quarters_(allocator),
      group_hosts_(allocator),
      input_offsets_(allocator),
      successful_counts_(allocator),
      successful_buckets_(allocator) {
}

void
SindiDateBuildPlan::RecordSuccess(uint32_t ordered_position) {
    if (not enabled_) {
        return;
    }
    while (static_cast<uint64_t>(successful_group_cursor_) + 1 < input_offsets_.size() and
           ordered_position >=
               input_offsets_[static_cast<uint64_t>(successful_group_cursor_) + 1]) {
        ++successful_group_cursor_;
    }
    ++successful_counts_[successful_group_cursor_];
    successful_buckets_.push_back(source_buckets_[order_[ordered_position]]);
}

SindiDateFilter::SindiDateFilter(Allocator* allocator)
    : allocator_(allocator), document_buckets_(allocator), partitions_(allocator) {
}

SindiDateBuildPlan
SindiDateFilter::PrepareBuild(const DatasetPtr& base) const {
    SindiDateBuildPlan plan(allocator_);
    const auto* date_buckets = base->GetPaths(SINDI_DATE_PATH_NAME);
    if (date_buckets == nullptr) {
        return plan;
    }

    const auto data_num = base->GetNumElements();
    CHECK_ARGUMENT(data_num <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                   "SINDI date-filtered build exceeds uint32_t document capacity");

    const auto* source_host_ids = base->GetUInt32Metadata(SINDI_HOST_ID_METADATA_NAME);
    plan.enabled_ = true;
    plan.has_host_metadata_ = source_host_ids != nullptr;
    plan.order_.resize(static_cast<uint64_t>(data_num));
    plan.source_buckets_.resize(static_cast<uint64_t>(data_num));
    Vector<uint32_t> source_quarters(static_cast<uint64_t>(data_num), allocator_);
    std::iota(plan.order_.begin(), plan.order_.end(), 0);
    for (uint32_t i = 0; i < static_cast<uint32_t>(data_num); ++i) {
        plan.source_buckets_[i] = parse_date_bucket(date_buckets[i]);
        source_quarters[i] = date_bucket_to_quarter(plan.source_buckets_[i]);
    }

    std::sort(plan.order_.begin(), plan.order_.end(), [&](uint32_t lhs, uint32_t rhs) {
        if (source_quarters[lhs] != source_quarters[rhs]) {
            return source_quarters[lhs] < source_quarters[rhs];
        }
        if (source_host_ids != nullptr and source_host_ids[lhs] != source_host_ids[rhs]) {
            return source_host_ids[lhs] < source_host_ids[rhs];
        }
        return lhs < rhs;
    });

    for (uint32_t position = 0; position < plan.order_.size(); ++position) {
        const auto source = plan.order_[position];
        const auto quarter = source_quarters[source];
        const auto host = source_host_ids == nullptr ? 0 : source_host_ids[source];
        if (plan.group_quarters_.empty() or plan.group_quarters_.back() != quarter or
            plan.group_hosts_.back() != host) {
            plan.group_quarters_.push_back(quarter);
            plan.group_hosts_.push_back(host);
            plan.input_offsets_.push_back(position);
        }
    }
    plan.input_offsets_.push_back(static_cast<uint32_t>(data_num));
    plan.successful_counts_.resize(plan.group_quarters_.size(), 0);
    plan.successful_buckets_.reserve(static_cast<uint64_t>(data_num));
    return plan;
}

void
SindiDateFilter::CommitBuild(SindiDateBuildPlan&& plan, uint64_t element_count) {
    this->Clear();
    if (not plan.Enabled()) {
        return;
    }
    CHECK_ARGUMENT(plan.successful_buckets_.size() == element_count,
                   "SINDI successful date bucket count does not match element count");

    has_host_metadata_ = plan.has_host_metadata_;
    document_buckets_ = std::move(plan.successful_buckets_);
    uint32_t inner_cursor = 0;
    for (uint64_t group_begin = 0; group_begin < plan.group_quarters_.size();) {
        uint64_t group_end = group_begin + 1;
        while (group_end < plan.group_quarters_.size() and
               plan.group_quarters_[group_end] == plan.group_quarters_[group_begin]) {
            ++group_end;
        }
        uint32_t quarter_count = 0;
        for (uint64_t group = group_begin; group < group_end; ++group) {
            quarter_count += plan.successful_counts_[group];
        }
        if (quarter_count > 0) {
            partitions_.emplace_back(allocator_);
            auto& partition = partitions_.back();
            partition.quarter = plan.group_quarters_[group_begin];
            partition.begin = inner_cursor;
            partition.end = inner_cursor + quarter_count;
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
                   "SINDI date partitions do not cover every indexed document");
}

void
SindiDateFilter::Clear() {
    Vector<uint32_t>(allocator_).swap(document_buckets_);
    Vector<Partition>(allocator_).swap(partitions_);
    has_host_metadata_ = false;
}

uint64_t
SindiDateFilter::GetMemoryUsage() const {
    uint64_t memory = document_buckets_.size() * sizeof(uint32_t);
    memory += partitions_.size() * sizeof(Partition);
    for (const auto& partition : partitions_) {
        memory += (partition.host_ids.size() + partition.host_offsets.size()) * sizeof(uint32_t);
    }
    return memory;
}

SindiDateSearchRoute
SindiDateFilter::Classify(const DatasetPtr& query, uint32_t window_size) const {
    SindiDateSearchRoute route;
    if (partitions_.empty()) {
        return route;
    }
    route.enabled = true;

    uint32_t begin_quarter = 0;
    uint32_t end_quarter = std::numeric_limits<uint32_t>::max();
    const auto* query_date = query->GetPaths(SINDI_DATE_PATH_NAME);
    const auto* query_date_begin = query->GetPaths(SINDI_DATE_BEGIN_PATH_NAME);
    const auto* query_date_end = query->GetPaths(SINDI_DATE_END_PATH_NAME);
    CHECK_ARGUMENT((query_date_begin == nullptr) == (query_date_end == nullptr),
                   "SINDI date_begin and date_end must be provided together");
    const bool conflicting_date_query = query_date != nullptr and query_date_begin != nullptr;
    CHECK_ARGUMENT(not conflicting_date_query,
                   "SINDI date cannot be combined with date_begin and date_end");
    if (query_date != nullptr) {
        route.has_date_bucket = true;
        route.query_bucket = parse_date_bucket(query_date[0]);
        begin_quarter = date_bucket_to_quarter(route.query_bucket);
        // A year-only bucket is anchored in Q1, so include Q1 through Q4.
        end_quarter = date_month(route.query_bucket) == 0 ? begin_quarter + 3 : begin_quarter;
    } else if (query_date_begin != nullptr) {
        route.has_date_range = true;
        route.query_begin = date_bucket_first_day(parse_date_bucket(query_date_begin[0]));
        route.query_end = date_bucket_last_day(parse_date_bucket(query_date_end[0]));
        CHECK_ARGUMENT(route.query_begin <= route.query_end,
                       "SINDI date_begin must not exceed date_end");
        begin_quarter = date_bucket_to_quarter(route.query_begin);
        end_quarter = date_bucket_to_quarter(route.query_end);
    }

    const auto* query_host_id = query->GetUInt32Metadata(SINDI_HOST_ID_METADATA_NAME);
    const bool use_host = has_host_metadata_ and query_host_id != nullptr;
    if (not route.has_date_bucket and not route.has_date_range and not use_host) {
        return route;
    }

    const bool has_date_query = route.has_date_bucket or route.has_date_range;
    const uint64_t route_quarter_count =
        has_date_query ? static_cast<uint64_t>(end_quarter) - begin_quarter + 1
                       : partitions_.size();
    route.inner_ranges.reserve(std::min<uint64_t>(partitions_.size(), route_quarter_count));
    for (const auto& partition : partitions_) {
        if (partition.quarter < begin_quarter or partition.quarter > end_quarter) {
            continue;
        }
        uint32_t begin = partition.begin;
        uint32_t end = partition.end;
        if (use_host) {
            const auto host = std::lower_bound(
                partition.host_ids.begin(), partition.host_ids.end(), query_host_id[0]);
            if (host == partition.host_ids.end() or *host != query_host_id[0]) {
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
SindiDateFilter::create_filter(const SindiDateSearchRoute& route, FilterPtr filter) const {
    return std::make_shared<DateRouteFilter>(route, &document_buckets_, std::move(filter));
}

void
SindiDateFilter::ApplyFilter(const SindiDateSearchRoute& route, FilterPtr& filter) const {
    if (not route.enabled or route.kind != SindiHostRouteKind::WINDOW) {
        return;
    }
    filter = create_filter(route, std::move(filter));
}

void
SindiDateFilter::ApplyWindowRoute(const SindiDateSearchRoute& route,
                                  int64_t& min_window_id,
                                  int64_t& max_window_id) {
    if (not route.enabled or route.kind != SindiHostRouteKind::WINDOW) {
        return;
    }
    min_window_id = std::max<int64_t>(min_window_id, route.window_ranges.front().first);
    max_window_id = std::min<int64_t>(max_window_id, route.window_ranges.back().second - 1);
}

int64_t
SindiDateFilter::NextMatchingWindow(const SindiDateSearchRoute& route,
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
SindiDateFilter::RequiresFullTermScan(const SindiDateSearchRoute& route,
                                      uint32_t window_id,
                                      uint32_t window_size) {
    if (not route.enabled or route.kind != SindiHostRouteKind::WINDOW) {
        return false;
    }
    if (route.has_date_bucket or route.has_date_range) {
        return true;
    }
    // A host-only route can include partial windows at quarter or host boundaries.
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
SindiDateFilter::Serialize(StreamWriter& writer) const {
    StreamWriter::WriteObj(writer, SINDI_DATE_METADATA_FORMAT_VERSION);
    StreamWriter::WriteObj(writer, static_cast<uint32_t>(has_host_metadata_));
    StreamWriter::WriteVector(writer, document_buckets_);
    StreamWriter::WriteObj(writer, static_cast<uint64_t>(partitions_.size()));
    for (const auto& partition : partitions_) {
        StreamWriter::WriteObj(writer, partition.quarter);
        StreamWriter::WriteObj(writer, partition.begin);
        StreamWriter::WriteObj(writer, partition.end);
        StreamWriter::WriteVector(writer, partition.host_ids);
        StreamWriter::WriteVector(writer, partition.host_offsets);
    }
}

void
SindiDateFilter::Deserialize(StreamReader& reader, uint64_t element_count) {
    CHECK_ARGUMENT(  // NOLINT(readability-simplify-boolean-expr)
        element_count > 0 && element_count <= std::numeric_limits<uint32_t>::max(),
        fmt::format("serialized SINDI date metadata element count must be in [1, {}], got {}",
                    std::numeric_limits<uint32_t>::max(),
                    element_count));

    uint32_t version = 0;
    uint32_t has_host = 0;
    StreamReader::ReadObj(reader, version);
    StreamReader::ReadObj(reader, has_host);
    CHECK_ARGUMENT(version == SINDI_DATE_METADATA_FORMAT_VERSION,
                   fmt::format("unsupported SINDI date metadata version {}", version));
    CHECK_ARGUMENT(has_host <= 1, "serialized SINDI date host flag is invalid");

    Vector<uint32_t> document_buckets(allocator_);
    read_vector(reader, document_buckets, element_count, "document date bucket");
    CHECK_ARGUMENT(document_buckets.size() == element_count,
                   "serialized SINDI document date bucket count does not match element count");
    uint64_t partition_count = 0;
    StreamReader::ReadObj(reader, partition_count);
    CHECK_ARGUMENT(partition_count > 0, "serialized SINDI date partition count is invalid");
    CHECK_ARGUMENT(partition_count <= element_count,
                   "serialized SINDI date partition count exceeds element count");

    Vector<Partition> partitions(allocator_);
    partitions.reserve(partition_count);
    uint32_t previous_quarter = 0;
    uint32_t inner_cursor = 0;
    for (uint64_t i = 0; i < partition_count; ++i) {
        partitions.emplace_back(allocator_);
        auto& partition = partitions.back();
        StreamReader::ReadObj(reader, partition.quarter);
        StreamReader::ReadObj(reader, partition.begin);
        StreamReader::ReadObj(reader, partition.end);
        if (i > 0) {
            CHECK_ARGUMENT(partition.quarter > previous_quarter,
                           "serialized SINDI quarters must be strictly ordered");
        }
        CHECK_ARGUMENT(partition.begin == inner_cursor,
                       "serialized SINDI date partitions must be contiguous");
        CHECK_ARGUMENT(partition.end > partition.begin,
                       "serialized SINDI date partition must not be empty");
        CHECK_ARGUMENT(partition.end <= element_count,
                       "serialized SINDI date partition exceeds element count");
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
                           "serialized SINDI date metadata has unexpected host directory");
        }
        for (uint32_t inner_id = partition.begin; inner_id < partition.end; ++inner_id) {
            CHECK_ARGUMENT(is_valid_date_bucket(document_buckets[inner_id]),
                           "serialized SINDI document date bucket is invalid");
            CHECK_ARGUMENT(date_bucket_to_quarter(document_buckets[inner_id]) == partition.quarter,
                           "serialized SINDI document date bucket does not match its quarter");
        }
        previous_quarter = partition.quarter;
        inner_cursor = partition.end;
    }
    CHECK_ARGUMENT(inner_cursor == element_count,
                   "serialized SINDI date partitions do not cover every indexed document");

    has_host_metadata_ = has_host != 0;
    document_buckets_ = std::move(document_buckets);
    partitions_ = std::move(partitions);
}

SindiMetadataFilter::SindiMetadataFilter(Allocator* allocator)
    : allocator_(allocator), host_filter_(allocator), date_filter_(allocator) {
}

SindiMetadataBuildPlan
SindiMetadataFilter::PrepareBuild(const DatasetPtr& base, uint64_t current_element_count) const {
    SindiMetadataBuildPlan plan(allocator_);
    if (current_element_count != 0) {
        CHECK_ARGUMENT(not date_filter_.HasMetadata(),
                       "SINDI date-aware index does not support incremental Add");
    }
    plan.date_plan_ = date_filter_.PrepareBuild(base);
    if (plan.date_plan_.Enabled()) {
        CHECK_ARGUMENT(current_element_count == 0,
                       "SINDI cannot add date metadata after existing documents");
    } else {
        plan.host_plan_ = host_filter_.PrepareBuild(base, current_element_count);
    }
    return plan;
}

void
SindiMetadataFilter::CommitBuild(SindiMetadataBuildPlan&& plan,
                                 uint32_t first_inner_id,
                                 uint32_t end_inner_id) {
    if (plan.date_plan_.Enabled()) {
        host_filter_.Clear();
        date_filter_.CommitBuild(std::move(plan.date_plan_), end_inner_id);
        return;
    }
    host_filter_.CommitBuild(std::move(plan.host_plan_), first_inner_id, end_inner_id);
    if (first_inner_id == 0) {
        date_filter_.Clear();
    }
}

void
SindiMetadataFilter::Clear() {
    host_filter_.Clear();
    date_filter_.Clear();
}

SindiMetadataSearchRoute
SindiMetadataFilter::Classify(const DatasetPtr& query, uint32_t window_size) const {
    SindiMetadataSearchRoute route;
    if (date_filter_.HasMetadata()) {
        route.date_route = date_filter_.Classify(query, window_size);
        route.kind = route.date_route.kind;
    } else {
        route.host_route = host_filter_.Classify(query);
        route.kind = route.host_route.kind;
    }
    return route;
}

void
SindiMetadataFilter::ApplyFilter(const SindiMetadataSearchRoute& route, FilterPtr& filter) const {
    if (route.date_route.enabled) {
        date_filter_.ApplyFilter(route.date_route, filter);
    } else {
        host_filter_.ApplyFilter(route.host_route, filter);
    }
}

void
SindiMetadataFilter::ApplyWindowRoute(const SindiMetadataSearchRoute& route,
                                      uint32_t window_size,
                                      int64_t& min_window_id,
                                      int64_t& max_window_id) {
    if (route.date_route.enabled) {
        SindiDateFilter::ApplyWindowRoute(route.date_route, min_window_id, max_window_id);
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
    if (route.date_route.enabled) {
        return SindiDateFilter::NextMatchingWindow(
            route.date_route, current_window_id, max_window_id);
    }
    return host_filter_.NextMatchingWindow(
        route.host_route, window_size, current_window_id, max_window_id);
}

bool
SindiMetadataFilter::RequiresFullTermScan(const SindiMetadataSearchRoute& route,
                                          uint32_t window_id,
                                          uint32_t window_size) const {
    if (route.date_route.enabled) {
        return SindiDateFilter::RequiresFullTermScan(route.date_route, window_id, window_size);
    }
    return host_filter_.RequiresFullTermScan(route.host_route, window_id, window_size);
}

void
SindiMetadataFilter::DeserializeHostMetadata(StreamReader& reader, uint64_t element_count) {
    host_filter_.Deserialize(reader, element_count);
    date_filter_.Clear();
}

void
SindiMetadataFilter::DeserializeDateMetadata(StreamReader& reader, uint64_t element_count) {
    date_filter_.Deserialize(reader, element_count);
    host_filter_.Clear();
}

}  // namespace vsag
