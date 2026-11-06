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

#include "svg_chart.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <sstream>

namespace vsag::dataset_profile {

namespace {

constexpr double MARGIN_LEFT = 58.0;
constexpr double MARGIN_RIGHT = 14.0;
constexpr double MARGIN_TOP = 12.0;
constexpr double MARGIN_BOTTOM = 38.0;

struct Scale {
    double lower{0.0};
    double upper{1.0};
    double pixel_lower{0.0};
    double pixel_upper{1.0};

    [[nodiscard]] double
    Map(double value) const {
        if (!(upper > lower)) {
            return (pixel_lower + pixel_upper) / 2.0;
        }
        return pixel_lower + (value - lower) / (upper - lower) * (pixel_upper - pixel_lower);
    }
};

std::string
fixed(double value) {
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.1f", value);
    return buffer;
}

std::string
tick_label(double value) {
    const double magnitude = std::abs(value);
    char buffer[32];
    if (magnitude >= 1e9) {
        std::snprintf(buffer, sizeof(buffer), "%.3gG", value / 1e9);
    } else if (magnitude >= 1e6) {
        std::snprintf(buffer, sizeof(buffer), "%.3gM", value / 1e6);
    } else if (magnitude >= 1e4) {
        std::snprintf(buffer, sizeof(buffer), "%.3gk", value / 1e3);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.3g", value);
    }
    return buffer;
}

std::vector<double>
nice_ticks(double lower, double upper, int target) {
    std::vector<double> ticks;
    if (!std::isfinite(lower) || !std::isfinite(upper)) {
        return ticks;
    }
    if (!(upper > lower)) {
        ticks.push_back(lower);
        return ticks;
    }
    const double raw = (upper - lower) / target;
    const double magnitude = std::pow(10.0, std::floor(std::log10(raw)));
    const double normalized = raw / magnitude;
    double step = 10.0;
    if (normalized < 1.5) {
        step = 1.0;
    } else if (normalized < 3.0) {
        step = 2.0;
    } else if (normalized < 7.0) {
        step = 5.0;
    }
    step *= magnitude;
    const double first = std::ceil(lower / step) * step;
    const auto count = static_cast<int64_t>(std::floor((upper - first) / step + 1e-9)) + 1;
    for (int64_t i = 0; i < std::min<int64_t>(count, 21); ++i) {
        const double value = first + static_cast<double>(i) * step;
        ticks.push_back(std::abs(value) < step * 1e-9 ? 0.0 : value);
    }
    return ticks;
}

void
open_svg(std::ostringstream& out, const ChartOptions& options) {
    out << R"(<svg class="chart" viewBox="0 0 )" << fixed(options.width) << ' '
        << fixed(options.height) << R"(" role="img" preserveAspectRatio="xMidYMid meet">)";
}

void
axis_titles(std::ostringstream& out, const ChartOptions& options) {
    if (!options.x_label.empty()) {
        out << R"(<text class="axis-title" x=")"
            << fixed((MARGIN_LEFT + options.width - MARGIN_RIGHT) / 2) << "\" y=\""
            << fixed(options.height - 4) << R"(" text-anchor="middle">)"
            << escape_html(options.x_label) << "</text>";
    }
    if (!options.y_label.empty()) {
        const double y = (MARGIN_TOP + options.height - MARGIN_BOTTOM) / 2;
        out << R"(<text class="axis-title" x="12" y=")" << fixed(y)
            << R"(" text-anchor="middle" transform="rotate(-90 12 )" << fixed(y) << ")\">"
            << escape_html(options.y_label) << "</text>";
    }
}

void
x_axis(std::ostringstream& out, const Scale& x, double baseline) {
    out << R"(<line class="axis" x1=")" << fixed(x.pixel_lower) << "\" y1=\"" << fixed(baseline)
        << "\" x2=\"" << fixed(x.pixel_upper) << "\" y2=\"" << fixed(baseline) << "\"/>";
    for (const double tick : nice_ticks(x.lower, x.upper, 6)) {
        const double px = x.Map(tick);
        out << R"(<line class="tick-mark" x1=")" << fixed(px) << "\" y1=\"" << fixed(baseline)
            << "\" x2=\"" << fixed(px) << "\" y2=\"" << fixed(baseline + 4) << "\"/>";
        out << R"(<text class="tick" x=")" << fixed(px) << "\" y=\"" << fixed(baseline + 16)
            << R"(" text-anchor="middle">)" << tick_label(tick) << "</text>";
    }
}

void
y_axis(std::ostringstream& out, const Scale& y, double left, double right, bool log_labels) {
    for (const double tick : nice_ticks(y.lower, y.upper, 4)) {
        const double py = y.Map(tick);
        out << R"(<line class="grid" x1=")" << fixed(left) << "\" y1=\"" << fixed(py) << "\" x2=\""
            << fixed(right) << "\" y2=\"" << fixed(py) << "\"/>";
        const double label = log_labels ? std::pow(10.0, tick) - 1.0 : tick;
        out << R"(<text class="tick" x=")" << fixed(left - 6) << "\" y=\"" << fixed(py + 4)
            << R"(" text-anchor="end">)" << tick_label(log_labels ? std::round(label) : label)
            << "</text>";
    }
}

}  // namespace

std::string
escape_html(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (const char character : text) {
        switch (character) {
            case '&':
                escaped += "&amp;";
                break;
            case '<':
                escaped += "&lt;";
                break;
            case '>':
                escaped += "&gt;";
                break;
            case '"':
                escaped += "&quot;";
                break;
            case '\'':
                escaped += "&#39;";
                break;
            default:
                escaped += character;
        }
    }
    return escaped;
}

std::string
format_number_html(double value) {
    if (!std::isfinite(value)) {
        return "n/a";
    }
    char buffer[48];
    const double magnitude = std::abs(value);
    if (value == std::floor(value) && magnitude < 1e15) {
        std::snprintf(buffer, sizeof(buffer), "%.0f", value);
        std::string digits = buffer;
        const bool negative = !digits.empty() && digits[0] == '-';
        std::string body = negative ? digits.substr(1) : digits;
        std::string grouped;
        for (uint64_t i = 0; i < body.size(); ++i) {
            if (i > 0 && (body.size() - i) % 3 == 0) {
                grouped += ',';
            }
            grouped += body[i];
        }
        return (negative ? "-" : "") + grouped;
    }
    if (magnitude != 0.0 && (magnitude < 1e-3 || magnitude >= 1e7)) {
        std::snprintf(buffer, sizeof(buffer), "%.3e", value);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.4g", value);
    }
    return buffer;
}

std::string
histogram_svg(const JsonType& histogram, const ChartOptions& options) {
    std::ostringstream out;
    open_svg(out, options);
    const auto& counts = histogram.at("counts");
    const double lower = histogram.value("lower", 0.0);
    const double width = histogram.value("bin_width", 0.0);
    const uint64_t bins = counts.size();
    double upper = lower + width * static_cast<double>(bins);
    if (!(upper > lower)) {
        upper = lower + 1.0;
    }
    double max_count = 0.0;
    for (const auto& count : counts) {
        max_count = std::max(max_count, count.get<double>());
    }
    const double plot_left = MARGIN_LEFT;
    const double plot_right = options.width - MARGIN_RIGHT;
    const double baseline = options.height - MARGIN_BOTTOM;
    const Scale x{lower, upper, plot_left, plot_right};
    const double bar_width =
        (plot_right - plot_left) / static_cast<double>(std::max<uint64_t>(bins, 1));
    for (const bool log_scale : {false, true}) {
        const double top = log_scale ? std::log10(max_count + 1.0) : max_count;
        const Scale y{0.0, top > 0.0 ? top : 1.0, baseline, MARGIN_TOP};
        out << "<g class=\"" << (log_scale ? "log" : "lin") << "\">";
        y_axis(out, y, plot_left, plot_right, log_scale);
        for (uint64_t i = 0; i < bins; ++i) {
            const double count = counts[i].get<double>();
            if (count <= 0.0) {
                continue;
            }
            const double value = log_scale ? std::log10(count + 1.0) : count;
            const double y_top = y.Map(value);
            const double bin_low = lower + width * static_cast<double>(i);
            out << R"(<rect class="bar" x=")"
                << fixed(plot_left + bar_width * static_cast<double>(i)) << "\" y=\""
                << fixed(y_top) << "\" width=\"" << fixed(std::max(bar_width - 1.0, 0.8))
                << "\" height=\"" << fixed(std::max(baseline - y_top, 0.5)) << "\"><title>["
                << format_number_html(bin_low) << ", " << format_number_html(bin_low + width)
                << "): " << format_number_html(count) << "</title></rect>";
        }
        out << "</g>";
    }
    x_axis(out, x, baseline);
    axis_titles(out, options);
    const auto below = histogram.value("below", 0ULL);
    const auto above = histogram.value("above", 0ULL);
    if (below + above > 0) {
        out << R"(<text class="note" x=")" << fixed(plot_right) << "\" y=\""
            << fixed(MARGIN_TOP + 2) << R"(" text-anchor="end">outside plotted range: )"
            << format_number_html(static_cast<double>(below)) << " below, "
            << format_number_html(static_cast<double>(above)) << " above</text>";
    }
    out << "</svg>";
    return out.str();
}

std::string
line_svg(const std::vector<LineSeries>& series,
         const std::vector<double>& x_values,
         const std::vector<double>& band_low,
         const std::vector<double>& band_high,
         const ChartOptions& options) {
    std::ostringstream out;
    open_svg(out, options);
    uint64_t points = 0;
    double y_low = std::numeric_limits<double>::infinity();
    double y_high = -std::numeric_limits<double>::infinity();
    auto extend = [&](const std::vector<double>& values) {
        for (const double value : values) {
            if (std::isfinite(value)) {
                y_low = std::min(y_low, value);
                y_high = std::max(y_high, value);
            }
        }
    };
    for (const auto& line : series) {
        points = std::max<uint64_t>(points, line.values.size());
        extend(line.values);
    }
    extend(band_low);
    extend(band_high);
    if (points == 0 || !std::isfinite(y_low)) {
        out << "</svg>";
        return out.str();
    }
    if (!(y_high > y_low)) {
        y_high = y_low + 1.0;
    }
    auto x_at = [&](uint64_t i) {
        return i < x_values.size() ? x_values[i] : static_cast<double>(i);
    };
    const double plot_left = MARGIN_LEFT;
    const double plot_right = options.width - MARGIN_RIGHT;
    const double baseline = options.height - MARGIN_BOTTOM;
    const Scale x{x_at(0), x_at(points - 1), plot_left, plot_right};
    const double padding = (y_high - y_low) * 0.05;
    const Scale y{y_low - padding, y_high + padding, baseline, MARGIN_TOP};
    y_axis(out, y, plot_left, plot_right, false);
    if (!band_low.empty() && band_low.size() == band_high.size()) {
        out << R"(<path class="band" d=")";
        for (uint64_t i = 0; i < band_high.size(); ++i) {
            out << (i == 0 ? 'M' : 'L') << fixed(x.Map(x_at(i))) << ','
                << fixed(y.Map(band_high[i]));
        }
        for (uint64_t i = band_low.size(); i-- > 0;) {
            out << 'L' << fixed(x.Map(x_at(i))) << ',' << fixed(y.Map(band_low[i]));
        }
        out << "Z\"/>";
    }
    double legend_x = plot_left + 8;
    for (const auto& line : series) {
        out << "<path class=\"line s" << line.color << (line.dashed ? " dashed" : "") << "\" d=\"";
        bool started = false;
        for (uint64_t i = 0; i < line.values.size(); ++i) {
            if (!std::isfinite(line.values[i])) {
                started = false;
                continue;
            }
            out << (started ? 'L' : 'M') << fixed(x.Map(x_at(i))) << ','
                << fixed(y.Map(line.values[i]));
            started = true;
        }
        out << "\"/>";
        if (!line.name.empty()) {
            out << "<rect class=\"legend s" << line.color << "\" x=\"" << fixed(legend_x)
                << "\" y=\"" << fixed(MARGIN_TOP)
                << R"(" width="10" height="3"/><text class="legend-text" x=")"
                << fixed(legend_x + 14) << "\" y=\"" << fixed(MARGIN_TOP + 5) << "\">"
                << escape_html(line.name) << "</text>";
            legend_x += 24 + 6.5 * static_cast<double>(line.name.size());
        }
    }
    if (points <= 64) {
        for (const auto& line : series) {
            for (uint64_t i = 0; i < line.values.size(); ++i) {
                if (std::isfinite(line.values[i])) {
                    out << "<circle class=\"dot s" << line.color << "\" cx=\""
                        << fixed(x.Map(x_at(i))) << "\" cy=\"" << fixed(y.Map(line.values[i]))
                        << R"(" r="2.5"><title>)" << escape_html(line.name) << " @ "
                        << format_number_html(x_at(i)) << ": " << format_number_html(line.values[i])
                        << "</title></circle>";
                }
            }
        }
    }
    x_axis(out, x, baseline);
    axis_titles(out, options);
    out << "</svg>";
    return out.str();
}

std::string
bar_svg(const std::vector<std::string>& labels,
        const std::vector<double>& values,
        const ChartOptions& options) {
    std::ostringstream out;
    const double row_height = 22.0;
    ChartOptions sized = options;
    sized.height = MARGIN_TOP + row_height * static_cast<double>(values.size()) + 26.0;
    open_svg(out, sized);
    double maximum = 0.0;
    for (const double value : values) {
        if (std::isfinite(value)) {
            maximum = std::max(maximum, value);
        }
    }
    if (maximum <= 0.0) {
        maximum = 1.0;
    }
    const double label_width = 120.0;
    const double plot_left = label_width;
    const double plot_right = sized.width - 70.0;
    for (uint64_t i = 0; i < values.size(); ++i) {
        const double y = MARGIN_TOP + row_height * static_cast<double>(i);
        const double value = std::isfinite(values[i]) ? values[i] : 0.0;
        const double length = (plot_right - plot_left) * value / maximum;
        out << R"(<text class="tick" x=")" << fixed(label_width - 8) << "\" y=\"" << fixed(y + 14)
            << R"(" text-anchor="end">)" << escape_html(labels[i]) << "</text>";
        out << "<rect class=\"bar c" << (i % 8) << "\" x=\"" << fixed(plot_left) << "\" y=\""
            << fixed(y + 3) << "\" width=\"" << fixed(std::max(length, 0.5)) << "\" height=\""
            << fixed(row_height - 7) << "\"><title>" << escape_html(labels[i]) << ": "
            << format_number_html(values[i]) << "</title></rect>";
        out << R"(<text class="value" x=")" << fixed(plot_left + length + 6) << "\" y=\""
            << fixed(y + 14) << "\">" << format_number_html(values[i]) << "</text>";
    }
    out << "</svg>";
    return out.str();
}

std::string
scatter_svg(const JsonType& points, const JsonType& overlay, const ChartOptions& options) {
    std::ostringstream out;
    open_svg(out, options);
    double x_low = std::numeric_limits<double>::infinity();
    double x_high = -x_low;
    double y_low = x_low;
    double y_high = -x_low;
    for (const auto* set : {&points, &overlay}) {
        for (const auto& point : *set) {
            if (point.size() < 2 || !point[0].is_number() || !point[1].is_number()) {
                continue;
            }
            x_low = std::min(x_low, point[0].get<double>());
            x_high = std::max(x_high, point[0].get<double>());
            y_low = std::min(y_low, point[1].get<double>());
            y_high = std::max(y_high, point[1].get<double>());
        }
    }
    if (!std::isfinite(x_low)) {
        out << "</svg>";
        return out.str();
    }
    const double plot_left = MARGIN_LEFT;
    const double plot_right = options.width - MARGIN_RIGHT;
    const double baseline = options.height - MARGIN_BOTTOM;
    const double x_pad = std::max((x_high - x_low) * 0.03, 1e-9);
    const double y_pad = std::max((y_high - y_low) * 0.03, 1e-9);
    const Scale x{x_low - x_pad, x_high + x_pad, plot_left, plot_right};
    const Scale y{y_low - y_pad, y_high + y_pad, baseline, MARGIN_TOP};
    y_axis(out, y, plot_left, plot_right, false);
    for (const auto& point : points) {
        if (point.size() < 2 || !point[0].is_number() || !point[1].is_number()) {
            continue;
        }
        const uint64_t group = point.size() > 2 ? point[2].get<uint64_t>() : 0;
        out << "<circle class=\"pt c" << (group % 8) << "\" cx=\""
            << fixed(x.Map(point[0].get<double>())) << "\" cy=\""
            << fixed(y.Map(point[1].get<double>())) << R"(" r="1.8"/>)";
    }
    for (const auto& point : overlay) {
        if (point.size() < 2 || !point[0].is_number() || !point[1].is_number()) {
            continue;
        }
        out << R"(<circle class="overlay" cx=")" << fixed(x.Map(point[0].get<double>()))
            << "\" cy=\"" << fixed(y.Map(point[1].get<double>())) << R"(" r="2.6"/>)";
    }
    x_axis(out, x, baseline);
    axis_titles(out, options);
    out << "</svg>";
    return out.str();
}

}  // namespace vsag::dataset_profile
