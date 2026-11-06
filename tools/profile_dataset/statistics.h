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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "profile_types.h"

namespace vsag::dataset_profile {

/// Central moments up to order four. Partial results merge with Pebay's pairwise formulas
/// (SAND2008-6212), so block-wise accumulation is numerically stable.
struct Moments {
    uint64_t count{0};
    double mean{0.0};
    double m2{0.0};
    double m3{0.0};
    double m4{0.0};

    void
    Add(double value);

    void
    Merge(const Moments& other);

    /// Population variance (divides by count).
    [[nodiscard]] double
    Variance() const;

    [[nodiscard]] double
    Std() const;

    /// Sample skewness g1; 0 when the variance is 0.
    [[nodiscard]] double
    Skewness() const;

    /// Excess kurtosis g2 - 3; 0 when the variance is 0.
    [[nodiscard]] double
    ExcessKurtosis() const;
};

/// Equal-width histogram over [lower, upper]. Values outside the range are counted in
/// `below` / `above` instead of being clamped.
class Histogram {
public:
    Histogram() = default;

    Histogram(double lower, double upper, uint64_t bins);

    void
    Add(double value);

    void
    Merge(const Histogram& other);

    [[nodiscard]] const std::vector<uint64_t>&
    Counts() const {
        return counts_;
    }

    [[nodiscard]] uint64_t
    Total() const;

    /// {"lower", "upper", "bin_width", "counts", "below", "above"}.
    [[nodiscard]] JsonType
    ToJson() const;

private:
    double lower_{0.0};
    double upper_{0.0};
    double width_{0.0};
    std::vector<uint64_t> counts_;
    uint64_t below_{0};
    uint64_t above_{0};
};

/// Linear-interpolated quantile of ascending values (NumPy's default "linear" method).
double
quantile_sorted(const std::vector<double>& sorted, double probability);

/// Sorts finite values in place and summarizes them:
/// {"count", "min", "max", "mean", "std", "p01", "p05", "p25", "p50", "p75", "p95", "p99"}.
/// Returns {"count": 0} for an empty input.
JsonType
summarize_values(std::vector<double>& values);

/// Charts clip to the [0.5%, 99.5%] quantile range when the full range is more than this many
/// times wider, so a few extreme values do not squash the bulk into one bin. Values outside
/// the plotted range stay counted in `below` / `above`; summaries always use every value.
constexpr double ROBUST_RANGE_TAIL = 0.005;
constexpr double ROBUST_RANGE_SPREAD = 4.0;
constexpr uint64_t ROBUST_RANGE_MIN_VALUES = 200;

/// Chooses the histogram range for ascending values: [min, max], or the quantile range above
/// when the tails dominate. Returns true when the range was clipped.
bool
robust_range(const std::vector<double>& sorted,
             double minimum,
             double maximum,
             double& lower,
             double& upper);

/// Histogram of ascending values over robust_range(); adds a "clipped" flag.
JsonType
histogram_of_sorted(const std::vector<double>& sorted, uint64_t bins);

/// Histogram of non-negative integers with unit-aligned bins (at most `max_bins` bins).
JsonType
integer_histogram(const std::vector<double>& sorted, uint64_t max_bins);

/// Two-sample Kolmogorov-Smirnov statistic of two ascending samples.
double
ks_statistic(const std::vector<double>& sorted_a, const std::vector<double>& sorted_b);

/// Gini coefficient of ascending non-negative values (0 = all equal, 1 = concentrated).
double
gini_sorted(const std::vector<double>& sorted);

/// Rounds to a fixed number of significant digits; keeps reports readable and makes
/// derived ratios stable when printed.
double
round_significant(double value, int digits);

/// JSON number, or null for NaN/Inf so reports never contain invalid JSON values.
JsonType
json_number(double value);

/// Safe ratio: returns 0 when the denominator is 0.
inline double
safe_ratio(double numerator, double denominator) {
    return denominator == 0.0 ? 0.0 : numerator / denominator;
}

}  // namespace vsag::dataset_profile
