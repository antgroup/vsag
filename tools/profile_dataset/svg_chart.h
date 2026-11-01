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
#include <string>
#include <vector>

#include "profile_types.h"

namespace vsag::dataset_profile {

// Inline SVG chart primitives for the HTML report. Colors come from CSS classes, so charts
// follow the page theme (light or dark) and the output contains no external references.

std::string
escape_html(const std::string& text);

/// Compact human-readable number: thousands separators for integers, 4 significant digits
/// otherwise, scientific notation for very small or large magnitudes.
std::string
format_number_html(double value);

struct ChartOptions {
    double width{520.0};
    double height{210.0};
    std::string x_label;
    std::string y_label;
};

/// Bars of a {"lower", "bin_width", "counts"} histogram. Renders linear and log10 count
/// variants; the page script toggles between them.
std::string
histogram_svg(const JsonType& histogram, const ChartOptions& options);

struct LineSeries {
    std::string name;
    std::vector<double> values;
    /// CSS series index (0-7).
    int color{0};
    bool dashed{false};
};

/// Line chart over x = 0..n-1 (or `x_values`), with an optional shaded band.
std::string
line_svg(const std::vector<LineSeries>& series,
         const std::vector<double>& x_values,
         const std::vector<double>& band_low,
         const std::vector<double>& band_high,
         const ChartOptions& options);

/// Horizontal categorical bars with value labels.
std::string
bar_svg(const std::vector<std::string>& labels,
        const std::vector<double>& values,
        const ChartOptions& options);

/// Scatter of [x, y, group] points with an optional overlay drawn as hollow markers.
std::string
scatter_svg(const JsonType& points, const JsonType& overlay, const ChartOptions& options);

}  // namespace vsag::dataset_profile
