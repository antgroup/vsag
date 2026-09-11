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

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
#include <stdexcept>

#include "autotune_internal.h"

namespace vsag::autotune::internal {

namespace {

using Emit = std::function<void(const JsonType&)>;

JsonType
candidate_choices(JsonType values) {
    return {{"$choices", std::move(values)}};
}

int64_t
range_integer(const JsonType& value) {
    if (value.is_number_unsigned()) {
        const auto number = value.get<uint64_t>();
        if (number > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument("$range integer values must fit int64");
        }
        return static_cast<int64_t>(number);
    }
    return value.get<int64_t>();
}

uint64_t
positive_uint64(const JsonType& value, const std::string& path) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(path + " must be a positive integer");
    }
    if (value.is_number_unsigned()) {
        const auto number = value.get<uint64_t>();
        if (number == 0) {
            throw std::invalid_argument(path + " must be a positive integer");
        }
        return number;
    }
    const auto number = value.get<int64_t>();
    if (number <= 0) {
        throw std::invalid_argument(path + " must be a positive integer");
    }
    return static_cast<uint64_t>(number);
}

void
expand(const JsonType& value, const Emit& emit);

void
expand_object(const JsonType& object,
              const JsonType::const_iterator& field,
              JsonType partial,
              const Emit& emit) {
    if (field == object.end()) {
        emit(partial);
        return;
    }

    const auto& key = field.key();
    const auto next = std::next(field);
    expand(field.value(), [&](const JsonType& expanded) {
        auto next_partial = partial;
        next_partial[key] = expanded;
        expand_object(object, next, std::move(next_partial), emit);
    });
}

void
expand_range(const JsonType& range, const Emit& emit) {
    if (!range.is_object() || range.size() != 3 || !range.contains("start") ||
        !range.contains("stop") || !range.contains("step")) {
        throw std::invalid_argument("$range requires start, stop and step");
    }
    if (!range["start"].is_number() || !range["stop"].is_number() || !range["step"].is_number()) {
        throw std::invalid_argument("$range start, stop and step must be numbers");
    }

    if (range["start"].is_number_integer() && range["stop"].is_number_integer() &&
        range["step"].is_number_integer()) {
        const auto start = range_integer(range["start"]);
        const auto stop = range_integer(range["stop"]);
        const auto step = range_integer(range["step"]);
        if (step == 0 || (start < stop && step < 0) || (start > stop && step > 0)) {
            throw std::invalid_argument("$range step does not reach stop");
        }
        auto current = start;
        for (;;) {
            emit(current);
            if (current == stop ||
                (step > 0 && current > std::numeric_limits<int64_t>::max() - step) ||
                (step < 0 && current < std::numeric_limits<int64_t>::min() - step)) {
                break;
            }
            const auto next = current + step;
            if ((step > 0 && next > stop) || (step < 0 && next < stop)) {
                break;
            }
            current = next;
        }
        return;
    }

    const auto start = range["start"].get<double>();
    const auto stop = range["stop"].get<double>();
    const auto step = range["step"].get<double>();
    if (!std::isfinite(start) || !std::isfinite(stop) || !std::isfinite(step) || step == 0.0 ||
        (start < stop && step < 0.0) || (start > stop && step > 0.0)) {
        throw std::invalid_argument("$range step does not reach stop");
    }
    double previous = 0.0;
    bool emitted = false;
    for (uint64_t i = 0;; ++i) {
        auto current = start + static_cast<double>(i) * step;
        const auto tolerance = std::max(std::abs(std::nextafter(current, stop) - current),
                                        std::abs(std::nextafter(stop, current) - stop));
        if (!std::isfinite(current) ||
            (step > 0.0 && current > stop && current - stop > tolerance) ||
            (step < 0.0 && current < stop && stop - current > tolerance)) {
            return;
        }
        if (emitted && current == previous) {
            throw std::invalid_argument("$range step is too small to advance");
        }
        if (emitted && std::abs(current - stop) <= tolerance) {
            current = stop;
        }
        emit(current);
        if (current == stop) {
            return;
        }
        previous = current;
        emitted = true;
    }
}

void
expand_choices(const JsonType& choices, const Emit& emit) {
    if (!choices.is_array()) {
        throw std::invalid_argument("$choices must be an array");
    }
    if (choices.empty()) {
        throw std::invalid_argument("$choices must not be empty");
    }

    std::set<std::string> seen;
    for (const auto& choice : choices) {
        if (seen.emplace(choice.dump()).second) {
            emit(choice);
        }
    }
}

void
expand(const JsonType& value, const Emit& emit) {
    if (value.is_array()) {
        emit(value);
        return;
    }
    if (value.is_object()) {
        if (value.contains("$choices")) {
            if (value.size() != 1) {
                throw std::invalid_argument("$choices cannot be mixed with other keys");
            }
            expand_choices(value["$choices"], emit);
            return;
        }
        if (value.contains("$range")) {
            if (value.size() != 1) {
                throw std::invalid_argument("$range cannot be mixed with other keys");
            }
            expand_range(value["$range"], emit);
            return;
        }
        expand_object(value, value.begin(), JsonType::object(), emit);
        return;
    }
    emit(value);
}

void
fill_hgraph_create(JsonType& create_params) {
    if (!create_params.contains("index_param")) {
        create_params["index_param"] = JsonType::object();
    }
    auto& params = create_params["index_param"];
    if (!params.is_object()) {
        throw std::invalid_argument("hgraph create_params.index_param must be an object");
    }
    if (!params.contains("base_quantization_type")) {
        params["base_quantization_type"] =
            candidate_choices(JsonType::array({"fp32", "sq8_uniform"}));
    }
    if (!params.contains("max_degree")) {
        params["max_degree"] = candidate_choices(JsonType::array({16, 32}));
    }
    if (!params.contains("ef_construction")) {
        params["ef_construction"] = candidate_choices(JsonType::array({100, 200}));
    }
}

void
fill_ivf_create(JsonType& create_params, uint64_t base_count) {
    if (!create_params.contains("index_param")) {
        create_params["index_param"] = JsonType::object();
    }
    auto& params = create_params["index_param"];
    if (!params.is_object()) {
        throw std::invalid_argument("ivf create_params.index_param must be an object");
    }
    if (!params.contains("base_quantization_type")) {
        params["base_quantization_type"] =
            candidate_choices(JsonType::array({"fp32", "sq8_uniform"}));
    }
    if (!params.contains("buckets_count")) {
        const auto first = std::min<uint64_t>(1024, base_count);
        const auto second = std::min<uint64_t>(2048, base_count);
        params["buckets_count"] =
            first == second ? JsonType(first) : candidate_choices(JsonType::array({first, second}));
    }
}

void
fill_hgraph_search(JsonType& search_params, uint64_t top_k) {
    if (!search_params.contains("hgraph")) {
        search_params["hgraph"] = JsonType::object();
    }
    auto& params = search_params["hgraph"];
    if (!params.is_object()) {
        throw std::invalid_argument("hgraph search_params.hgraph must be an object");
    }
    if (!params.contains("ef_search")) {
        std::set<uint64_t> values{std::max<uint64_t>(40, top_k),
                                  std::max<uint64_t>(80, top_k * 2),
                                  std::max<uint64_t>(120, top_k * 4)};
        params["ef_search"] = candidate_choices(values);
    }
}

void
fill_pyramid_search(JsonType& search_params, uint64_t top_k) {
    if (!search_params.contains("pyramid")) {
        search_params["pyramid"] = JsonType::object();
    }
    auto& params = search_params["pyramid"];
    if (!params.is_object()) {
        throw std::invalid_argument("pyramid search_params.pyramid must be an object");
    }
    if (!params.contains("ef_search")) {
        std::set<uint64_t> values{std::max<uint64_t>(40, top_k),
                                  std::max<uint64_t>(80, top_k * 2),
                                  std::max<uint64_t>(120, top_k * 4)};
        params["ef_search"] = candidate_choices(values);
    }
}

void
fill_ivf_search(JsonType& search_params, const JsonType& create_params) {
    if (!search_params.contains("ivf")) {
        search_params["ivf"] = JsonType::object();
    }
    auto& params = search_params["ivf"];
    if (!params.is_object()) {
        throw std::invalid_argument("ivf search_params.ivf must be an object");
    }
    if (params.contains("scan_buckets_count")) {
        return;
    }

    std::set<uint64_t> values;
    if (!create_params.contains("index_param") ||
        !create_params["index_param"].contains("buckets_count")) {
        params["scan_buckets_count"] = candidate_choices(std::set<uint64_t>{1, 4, 16, 64});
        return;
    }

    const auto buckets = positive_uint64(create_params["index_param"]["buckets_count"],
                                         "ivf create_params.index_param.buckets_count");
    if (buckets <= 16) {
        values = {
            1, std::max<uint64_t>(1, buckets / 4), std::max<uint64_t>(1, buckets / 2), buckets};
    } else {
        values = {std::min<uint64_t>(16, buckets),
                  std::min<uint64_t>(32, buckets),
                  std::min<uint64_t>(64, buckets)};
    }
    params["scan_buckets_count"] = candidate_choices(values);
}

bool
supports_adaptive_ef_search_objective(const std::string& objective) {
    return objective == "latency_avg_ms" || objective == "latency_p99_ms" || objective == "qps" ||
           objective == "search_seconds" || objective == "build_and_search_seconds";
}

int64_t
positive_int64(const JsonType& value, const std::string& path) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument(path + " must be a positive integer");
    }
    if (value.is_number_unsigned()) {
        const auto number = value.get<uint64_t>();
        if (number == 0 || number > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument(path + " must be a positive integer");
        }
        return static_cast<int64_t>(number);
    }
    const auto number = value.get<int64_t>();
    if (number <= 0) {
        throw std::invalid_argument(path + " must be a positive integer");
    }
    return number;
}

std::optional<HGraphEfSearchRange>
take_hgraph_ef_search_range(JsonType& search_params, const RequestContext& request) {
    if (!search_params.contains("hgraph") || !search_params["hgraph"].is_object()) {
        return std::nullopt;
    }
    auto& hgraph = search_params["hgraph"];
    if (!hgraph.contains("ef_search") || !hgraph["ef_search"].is_object() ||
        !hgraph["ef_search"].contains("$range")) {
        return std::nullopt;
    }

    auto& expression = hgraph["ef_search"];
    const auto& range = expression["$range"];
    if (range.is_object() && range.contains("step")) {
        return std::nullopt;
    }
    if (expression.size() != 1 || !range.is_object() || range.size() != 2 ||
        !range.contains("start") || !range.contains("stop")) {
        throw std::invalid_argument(
            "hgraph ef_search $range without step requires exactly start and stop");
    }
    if (request.constraints.find("recall_at_k") == request.constraints.end()) {
        throw std::invalid_argument(
            "hgraph ef_search $range without step requires a recall_at_k constraint");
    }
    if (!supports_adaptive_ef_search_objective(request.objective)) {
        throw std::invalid_argument(
            "hgraph ef_search $range without step requires a query-cost objective");
    }
    if (hgraph.contains("timeout_ms") || hgraph.contains("hops_limit")) {
        throw std::invalid_argument(
            "hgraph ef_search $range without step does not support timeout_ms or hops_limit");
    }

    HGraphEfSearchRange result{positive_int64(range["start"], "hgraph ef_search $range.start"),
                               positive_int64(range["stop"], "hgraph ef_search $range.stop")};
    if (result.start > result.stop) {
        throw std::invalid_argument("hgraph ef_search $range.start must not exceed stop");
    }
    hgraph.erase("ef_search");
    return result;
}

uint64_t
maximum_trial_count(const std::optional<HGraphEfSearchRange>& range) {
    if (!range.has_value() || range->start == range->stop) {
        return 1;
    }

    uint64_t probes = 1;
    uint64_t maximum = 1;
    auto low = range->start;
    while (low < range->stop) {
        const auto high = low > range->stop / 2 ? range->stop : low * 2;
        ++probes;

        uint64_t binary_trials = 0;
        auto width = static_cast<uint64_t>(high - low);
        while (width > 1) {
            ++binary_trials;
            width = (width + 1) / 2;
        }
        maximum = std::max(maximum, probes + binary_trials);
        low = high;
    }
    return maximum;
}

std::vector<Candidate>
generate_candidates(const RequestContext& context,
                    const std::vector<IndexInput>& indexes,
                    bool generate_create_candidates) {
    std::vector<Candidate> candidates;
    std::set<std::string> seen;
    uint64_t maximum_trials = 0;
    for (const auto& index : indexes) {
        auto create_space = index.create_params;
        if (generate_create_candidates) {
            if (index.name == "hgraph") {
                fill_hgraph_create(create_space);
            } else if (index.name == "ivf") {
                fill_ivf_create(create_space, context.base_count);
            }
        }

        const auto expand_search = [&](const JsonType& create_params) {
            auto search_space = index.search_params;
            if (index.name == "hgraph") {
                fill_hgraph_search(search_space, context.top_k);
            } else if (index.name == "pyramid") {
                fill_pyramid_search(search_space, context.top_k);
            } else {
                fill_ivf_search(search_space, create_params);
            }
            const auto ef_search_range = index.name == "hgraph"
                                             ? take_hgraph_ef_search_range(search_space, context)
                                             : std::nullopt;
            expand(search_space, [&](const JsonType& search_params) {
                Candidate candidate{index.name, create_params, search_params, ef_search_range};
                JsonType identity{{"index_name", candidate.index_name},
                                  {"create_params", candidate.create_params},
                                  {"search_params", candidate.search_params}};
                if (ef_search_range.has_value()) {
                    identity["ef_search_range"] = {{"start", ef_search_range->start},
                                                   {"stop", ef_search_range->stop}};
                }
                const auto key = identity.dump();
                if (!seen.emplace(key).second) {
                    return;
                }
                const auto trial_count = maximum_trial_count(ef_search_range);
                if (trial_count > context.max_trials - maximum_trials) {
                    throw std::invalid_argument(
                        "planned trial count exceeds tuning_config.max_trials");
                }
                maximum_trials += trial_count;
                candidates.emplace_back(std::move(candidate));
            });
        };
        if (!generate_create_candidates) {
            expand_search(create_space);
        } else {
            expand(create_space, expand_search);
        }
    }
    if (candidates.empty()) {
        throw std::invalid_argument("candidate generation produced no trials");
    }
    return candidates;
}

}  // namespace

std::vector<Candidate>
GenerateCandidates(const IndexTuningRequest& request) {
    return generate_candidates(request.context, request.indexes, true);
}

std::vector<Candidate>
GenerateCandidates(const SearchTuningRequest& request) {
    return generate_candidates(request.context, {request.index_input}, false);
}

}  // namespace vsag::autotune::internal
