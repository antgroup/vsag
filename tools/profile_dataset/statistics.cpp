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

#include "statistics.h"

#include <limits>

namespace vsag::dataset_profile {

void
Moments::Add(double value) {
    const auto previous = static_cast<double>(count);
    count += 1;
    const auto n = static_cast<double>(count);
    const double delta = value - mean;
    const double delta_n = delta / n;
    const double delta_n2 = delta_n * delta_n;
    const double term = delta * delta_n * previous;
    mean += delta_n;
    m4 += term * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * m2 - 4.0 * delta_n * m3;
    m3 += term * delta_n * (n - 2.0) - 3.0 * delta_n * m2;
    m2 += term;
}

void
Moments::Merge(const Moments& other) {
    if (other.count == 0) {
        return;
    }
    if (count == 0) {
        *this = other;
        return;
    }
    const auto na = static_cast<double>(count);
    const auto nb = static_cast<double>(other.count);
    const double n = na + nb;
    const double delta = other.mean - mean;
    const double delta2 = delta * delta;
    const double delta3 = delta2 * delta;
    const double delta4 = delta2 * delta2;
    const double merged_mean = mean + delta * nb / n;
    const double merged_m2 = m2 + other.m2 + delta2 * na * nb / n;
    const double merged_m3 = m3 + other.m3 + delta3 * na * nb * (na - nb) / (n * n) +
                             3.0 * delta * (na * other.m2 - nb * m2) / n;
    const double merged_m4 = m4 + other.m4 +
                             delta4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n) +
                             6.0 * delta2 * (na * na * other.m2 + nb * nb * m2) / (n * n) +
                             4.0 * delta * (na * other.m3 - nb * m3) / n;
    count += other.count;
    mean = merged_mean;
    m2 = merged_m2;
    m3 = merged_m3;
    m4 = merged_m4;
}

double
Moments::Variance() const {
    if (count == 0) {
        return 0.0;
    }
    return std::max(0.0, m2 / static_cast<double>(count));
}

double
Moments::Std() const {
    return std::sqrt(Variance());
}

namespace {

bool
negligible_variance(const Moments& moments) {
    const double std = moments.Std();
    return moments.count < 2 || std <= 1e-12 * std::max(1.0, std::abs(moments.mean));
}

}  // namespace

double
Moments::Skewness() const {
    if (negligible_variance(*this)) {
        return 0.0;
    }
    const auto n = static_cast<double>(count);
    return std::sqrt(n) * m3 / std::pow(m2, 1.5);
}

double
Moments::ExcessKurtosis() const {
    if (negligible_variance(*this)) {
        return 0.0;
    }
    const auto n = static_cast<double>(count);
    return n * m4 / (m2 * m2) - 3.0;
}

Histogram::Histogram(double lower, double upper, uint64_t bins)
    : lower_(lower), upper_(upper), counts_(std::max<uint64_t>(bins, 1), 0) {
    if (!(upper_ > lower_)) {
        upper_ = lower_;
        counts_.assign(1, 0);
    }
    width_ = (upper_ - lower_) / static_cast<double>(counts_.size());
}

void
Histogram::Add(double value) {
    if (counts_.empty()) {
        return;
    }
    if (value < lower_) {
        ++below_;
        return;
    }
    if (value > upper_) {
        ++above_;
        return;
    }
    uint64_t index = 0;
    if (width_ > 0.0) {
        index = static_cast<uint64_t>((value - lower_) / width_);
        index = std::min<uint64_t>(index, counts_.size() - 1);
    }
    ++counts_[index];
}

void
Histogram::Merge(const Histogram& other) {
    if (counts_.size() != other.counts_.size()) {
        throw std::logic_error("cannot merge histograms with different bin counts");
    }
    for (uint64_t i = 0; i < counts_.size(); ++i) {
        counts_[i] += other.counts_[i];
    }
    below_ += other.below_;
    above_ += other.above_;
}

uint64_t
Histogram::Total() const {
    uint64_t total = below_ + above_;
    for (const auto count : counts_) {
        total += count;
    }
    return total;
}

JsonType
Histogram::ToJson() const {
    return JsonType{{"lower", json_number(lower_)},
                    {"upper", json_number(upper_)},
                    {"bin_width", json_number(width_)},
                    {"counts", counts_},
                    {"below", below_},
                    {"above", above_}};
}

double
quantile_sorted(const std::vector<double>& sorted, double probability) {
    if (sorted.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (sorted.size() == 1) {
        return sorted.front();
    }
    const double position = static_cast<double>(sorted.size() - 1) * probability;
    const auto lower = static_cast<uint64_t>(std::floor(position));
    const uint64_t upper = std::min<uint64_t>(lower + 1, sorted.size() - 1);
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

JsonType
summarize_values(std::vector<double>& values) {
    values.erase(
        std::remove_if(
            values.begin(), values.end(), [](double value) { return !std::isfinite(value); }),
        values.end());
    std::sort(values.begin(), values.end());
    if (values.empty()) {
        return JsonType{{"count", 0}};
    }
    Moments moments;
    for (const double value : values) {
        moments.Add(value);
    }
    return JsonType{{"count", values.size()},
                    {"min", values.front()},
                    {"max", values.back()},
                    {"mean", json_number(moments.mean)},
                    {"std", json_number(moments.Std())},
                    {"p01", json_number(quantile_sorted(values, 0.01))},
                    {"p05", json_number(quantile_sorted(values, 0.05))},
                    {"p25", json_number(quantile_sorted(values, 0.25))},
                    {"p50", json_number(quantile_sorted(values, 0.50))},
                    {"p75", json_number(quantile_sorted(values, 0.75))},
                    {"p95", json_number(quantile_sorted(values, 0.95))},
                    {"p99", json_number(quantile_sorted(values, 0.99))}};
}

bool
robust_range(const std::vector<double>& sorted,
             double minimum,
             double maximum,
             double& lower,
             double& upper) {
    lower = minimum;
    upper = maximum;
    if (sorted.size() < ROBUST_RANGE_MIN_VALUES) {
        return false;
    }
    const double low = quantile_sorted(sorted, ROBUST_RANGE_TAIL);
    const double high = quantile_sorted(sorted, 1.0 - ROBUST_RANGE_TAIL);
    if (high > low && maximum - minimum > ROBUST_RANGE_SPREAD * (high - low)) {
        lower = low;
        upper = high;
        return true;
    }
    return false;
}

JsonType
histogram_of_sorted(const std::vector<double>& sorted, uint64_t bins) {
    if (sorted.empty()) {
        return Histogram(0.0, 0.0, 1).ToJson();
    }
    double lower = 0.0;
    double upper = 0.0;
    const bool clipped = robust_range(sorted, sorted.front(), sorted.back(), lower, upper);
    Histogram histogram(lower, upper, bins);
    for (const double value : sorted) {
        histogram.Add(value);
    }
    JsonType json = histogram.ToJson();
    json["clipped"] = clipped;
    return json;
}

JsonType
integer_histogram(const std::vector<double>& sorted, uint64_t max_bins) {
    if (sorted.empty()) {
        return Histogram(0.0, 0.0, 1).ToJson();
    }
    const double minimum = sorted.front();
    const double maximum = sorted.back();
    const auto span = static_cast<uint64_t>(maximum - minimum) + 1;
    const uint64_t bins = std::max<uint64_t>(1, std::min(span, max_bins));
    const uint64_t width = (span + bins - 1) / bins;
    const double lower = minimum - 0.5;
    const double upper = lower + static_cast<double>(width * bins);
    Histogram histogram(lower, upper, bins);
    for (const double value : sorted) {
        histogram.Add(value);
    }
    return histogram.ToJson();
}

double
ks_statistic(const std::vector<double>& sorted_a, const std::vector<double>& sorted_b) {
    if (sorted_a.empty() || sorted_b.empty()) {
        return 0.0;
    }
    const auto na = static_cast<double>(sorted_a.size());
    const auto nb = static_cast<double>(sorted_b.size());
    uint64_t i = 0;
    uint64_t j = 0;
    double statistic = 0.0;
    while (i < sorted_a.size() && j < sorted_b.size()) {
        const double x = std::min(sorted_a[i], sorted_b[j]);
        while (i < sorted_a.size() && sorted_a[i] <= x) {
            ++i;
        }
        while (j < sorted_b.size() && sorted_b[j] <= x) {
            ++j;
        }
        statistic = std::max(statistic,
                             std::abs(static_cast<double>(i) / na - static_cast<double>(j) / nb));
    }
    return statistic;
}

double
gini_sorted(const std::vector<double>& sorted) {
    if (sorted.empty()) {
        return 0.0;
    }
    double weighted = 0.0;
    double total = 0.0;
    const auto n = static_cast<double>(sorted.size());
    for (uint64_t i = 0; i < sorted.size(); ++i) {
        weighted += (2.0 * static_cast<double>(i + 1) - n - 1.0) * sorted[i];
        total += sorted[i];
    }
    return total == 0.0 ? 0.0 : weighted / (n * total);
}

double
round_significant(double value, int digits) {
    if (value == 0.0 || !std::isfinite(value)) {
        return value;
    }
    const double magnitude = std::floor(std::log10(std::abs(value)));
    const double factor = std::pow(10.0, static_cast<double>(digits - 1) - magnitude);
    return std::round(value * factor) / factor;
}

JsonType
json_number(double value) {
    if (!std::isfinite(value)) {
        return nullptr;
    }
    return value;
}

}  // namespace vsag::dataset_profile
