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

#include "test_budget.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace fixtures::budget {
namespace {

uint64_t
Add(uint64_t left, uint64_t right) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::overflow_error("test budget point addition overflow");
    }
    return left + right;
}

uint64_t
Multiply(uint64_t left, uint64_t right) {
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        throw std::overflow_error("test budget point multiplication overflow");
    }
    return left * right;
}

uint64_t
Product(std::initializer_list<uint64_t> factors) {
    if (std::find(factors.begin(), factors.end(), 0) != factors.end()) {
        return 0;
    }
    uint64_t result = 1;
    for (auto factor : factors) {
        result = Multiply(result, factor);
    }
    return result;
}

uint64_t
Next(uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    uint64_t result = state;
    result = (result ^ (result >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    result = (result ^ (result >> 27U)) * 0x94d049bb133111ebULL;
    return result ^ (result >> 31U);
}

uint64_t
Bounded(uint64_t& state, uint64_t bound) {
    // Unsigned wraparound is deliberate here: rejection avoids modulo bias.
    const auto threshold = (uint64_t{0} - bound) % bound;
    for (;;) {
        const auto value = Next(state);
        if (value >= threshold) {
            return value % bound;
        }
    }
}

std::string
Quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char byte : value) {
        if (byte == '"' || byte == '\\') {
            out << '\\' << static_cast<char>(byte);
        } else if (byte < 0x20) {
            out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                << static_cast<unsigned>(byte) << std::dec;
        } else {
            out << static_cast<char>(byte);
        }
    }
    out << '"';
    return out.str();
}

uint64_t
BaseQuantizerWeight(const std::string& quantizer) {
    if (quantizer == "fp32" || quantizer == "fp16" || quantizer == "bf16") {
        return 1;
    }
    if (quantizer == "sq8" || quantizer == "sq4" || quantizer == "sq8_uniform" ||
        quantizer == "sq4_uniform") {
        return 2;
    }
    if (quantizer == "rabitq" || quantizer == "pq" || quantizer == "pqfs" || quantizer == "tq") {
        return 8;
    }
    throw std::invalid_argument("unknown test budget quantizer: " + quantizer);
}

uint64_t
PositiveInteger(const std::string& text) {
    if (text.empty()) {
        throw std::invalid_argument("empty test budget quantizer bit width");
    }
    uint64_t result = 0;
    for (auto digit : text) {
        if (digit < '0' || digit > '9') {
            throw std::invalid_argument("invalid test budget quantizer bit width");
        }
        result = Add(Multiply(result, 10), static_cast<uint64_t>(digit - '0'));
    }
    if (result == 0) {
        throw std::invalid_argument("zero test budget quantizer bit width");
    }
    return result;
}

auto
Coverage(const Workload& work) {
    return std::tie(work.metric,
                    work.quantizer,
                    work.data_runs,
                    work.train_runs,
                    work.build_runs,
                    work.search_runs,
                    work.oracle_runs,
                    work.serialize_runs,
                    work.train_iterations,
                    work.train_centroids,
                    work.build_candidates,
                    work.search_candidates,
                    work.repetitions,
                    work.threads,
                    work.min_dim,
                    work.min_num,
                    work.min_train_num);
}

void
LogCost(std::ostream& output, const Cost& cost) {
    output << "{\"data\":" << cost.data << ",\"train\":" << cost.train
           << ",\"build\":" << cost.build << ",\"search\":" << cost.search
           << ",\"oracle\":" << cost.oracle << ",\"serialize\":" << cost.serialize
           << ",\"update\":" << cost.update << '}';
}

}  // namespace

uint64_t
Cost::Total() const {
    uint64_t total = 0;
    for (auto phase : {data, train, build, search, oracle, serialize, update}) {
        total = Add(total, phase);
    }
    return total;
}

Cost&
Cost::operator+=(const Cost& other) {
    Cost updated{Add(data, other.data),
                 Add(train, other.train),
                 Add(build, other.build),
                 Add(search, other.search),
                 Add(oracle, other.oracle),
                 Add(serialize, other.serialize),
                 Add(update, other.update)};
    updated.Total();
    *this = updated;
    return *this;
}

Policy
Policy::PullRequest() {
    return {"pr", 2000000000ULL, 47};
}

Policy
Policy::Daily() {
    return {"daily", 24000000000ULL, 47};
}

uint64_t
CaseSeed(const std::string& case_key, uint64_t global_seed) {
    uint64_t state = 14695981039346656037ULL;
    for (unsigned char byte : case_key) {
        state = (state ^ byte) * 1099511628211ULL;
    }
    for (uint64_t byte = 0; byte < 8; ++byte) {
        state = (state ^ ((global_seed >> (byte * 8U)) & 0xffU)) * 1099511628211ULL;
    }
    return Next(state);
}

uint64_t
SampleInclusive(uint64_t seed, uint64_t minimum, uint64_t maximum) {
    if (minimum > maximum) {
        throw std::invalid_argument("invalid test budget sample range");
    }
    if (minimum == 0 && maximum == std::numeric_limits<uint64_t>::max()) {
        return Next(seed);
    }
    return minimum + Bounded(seed, maximum - minimum + 1);
}

std::vector<uint64_t>
ShuffledIndices(uint64_t count, uint64_t seed) {
    std::vector<uint64_t> result(count);
    std::iota(result.begin(), result.end(), uint64_t{0});
    for (uint64_t remaining = count; remaining > 1; --remaining) {
        std::swap(result[remaining - 1], result[Bounded(seed, remaining)]);
    }
    return result;
}

uint64_t
MetricWeight(const std::string& metric) {
    if (metric == "l2" || metric == "ip") {
        return 1;
    }
    if (metric == "cosine") {
        return 3;
    }
    throw std::invalid_argument("unknown test budget metric: " + metric);
}

uint64_t
QuantizerWeight(const std::string& quantizer) {
    const auto transition = quantizer.find("->");
    if (transition != std::string::npos) {
        return Add(QuantizerWeight(quantizer.substr(0, transition)),
                   QuantizerWeight(quantizer.substr(transition + 2)));
    }
    std::vector<std::string> parts;
    std::istringstream input(quantizer);
    std::string part;
    while (std::getline(input, part, ',')) {
        parts.push_back(part);
    }
    if (parts.empty() || quantizer.back() == ',') {
        throw std::invalid_argument("empty test budget quantizer");
    }
    auto weight = BaseQuantizerWeight(parts[0]);
    if (parts.size() >= 2) {
        weight = Add(weight, BaseQuantizerWeight(parts[1]));
    }
    if (parts.size() == 5 && parts[0] == "rabitq" && !parts[2].empty()) {
        const auto query_bits = PositiveInteger(parts[3]);
        const auto base_bits = PositiveInteger(parts[4]);
        // Bit widths affect logical work; the storage label remains available in the log.
        weight = Add(weight, Add(Add(query_bits, 7) / 8, base_bits));
    } else if (parts.size() > 2) {
        throw std::invalid_argument("unsupported test budget quantizer configuration");
    }
    return weight;
}

Cost
EstimateCost(const Workload& work) {
    if (work.min_dim == 0 || work.min_num == 0 || work.dim < work.min_dim ||
        work.num < work.min_num || work.train_num < work.min_train_num || work.repetitions == 0 ||
        work.threads == 0 || work.train_iterations == 0 || work.train_centroids == 0 ||
        work.build_candidates == 0 || work.search_candidates == 0) {
        throw std::invalid_argument("test budget workload violates its legal minimum or factors");
    }
    if ((work.train_runs != 0 && work.train_num == 0) ||
        ((work.search_runs != 0 || work.oracle_runs != 0) && work.query_num == 0)) {
        throw std::invalid_argument("test budget phase requires nonzero samples");
    }
    const auto metric = MetricWeight(work.metric);
    const auto quantizer = QuantizerWeight(work.quantizer);
    Cost result;
    result.data =
        Product({work.dim, Add(work.num, work.query_num), work.data_runs, work.repetitions});
    result.train = Product({work.dim,
                            work.train_num,
                            work.train_runs,
                            work.train_iterations,
                            work.train_centroids,
                            quantizer,
                            work.repetitions});
    result.build = Product({work.dim,
                            work.num,
                            work.build_runs,
                            work.build_candidates,
                            metric,
                            quantizer,
                            work.repetitions});
    result.search = Product({work.dim,
                             work.query_num,
                             work.search_runs,
                             work.search_candidates,
                             metric,
                             quantizer,
                             work.repetitions});
    result.oracle =
        Product({work.dim, work.num, work.query_num, work.oracle_runs, metric, work.repetitions});
    result.serialize = Product({work.dim, work.num, work.serialize_runs, work.repetitions});
    result.update = Product(
        {work.dim, work.update_num, work.build_candidates, metric, quantizer, work.repetitions});
    if (result.Total() == 0) {
        throw std::invalid_argument("test budget workload must declare nonzero work");
    }
    return result;
}

CasePlan
PlanCase(const CaseIdentity& identity,
         const Policy& policy,
         const std::vector<std::vector<RunSpec>>& complete_candidates) {
    if (identity.name.empty() || identity.source_file.empty() || policy.name.empty() ||
        policy.cap_points == 0 || complete_candidates.empty() ||
        complete_candidates.front().empty()) {
        throw std::invalid_argument("test budget requires a named nonempty case and positive cap");
    }
    const auto& baseline = complete_candidates.front();
    std::set<std::string> keys;
    for (const auto& run : baseline) {
        if (run.key.empty() || !keys.insert(run.key).second) {
            throw std::invalid_argument("test budget requires unique nonempty run keys");
        }
    }
    std::vector<Cost> totals;
    uint64_t previous = 0;
    const auto* previous_candidate = &baseline;
    for (const auto& candidate : complete_candidates) {
        if (candidate.size() != baseline.size()) {
            throw std::invalid_argument("test budget candidate drops required runs");
        }
        Cost total;
        for (uint64_t index = 0; index < candidate.size(); ++index) {
            const auto& run = candidate[index];
            const auto& required = baseline[index];
            if (run.key != required.key || Coverage(run.workload) != Coverage(required.workload)) {
                throw std::invalid_argument("test budget candidate changes required coverage");
            }
            const auto& old = (*previous_candidate)[index].workload;
            const auto& work = run.workload;
            if (work.dim < old.dim || work.num < old.num || work.query_num < old.query_num ||
                work.train_num < old.train_num || work.update_num < old.update_num) {
                throw std::invalid_argument("test budget candidates must not shrink required work");
            }
            total += EstimateCost(run.workload);
        }
        if (total.Total() < previous) {
            throw std::invalid_argument("test budget candidates must increase in total work");
        }
        previous = total.Total();
        totals.push_back(total);
        previous_candidate = &candidate;
    }
    if (totals.front().Total() > policy.cap_points) {
        throw std::invalid_argument("test budget cannot fit minimum legal case: " + identity.name +
                                    " minimum=" + std::to_string(totals.front().Total()) +
                                    " cap=" + std::to_string(policy.cap_points));
    }
    CasePlan plan;
    plan.identity_ = identity;
    plan.policy_ = policy;
    plan.seed_ = CaseSeed(identity.name, policy.seed);
    for (uint64_t index = 0; index < totals.size(); ++index) {
        if (totals[index].Total() <= policy.cap_points) {
            plan.selected_candidate_ = index;
        }
    }
    plan.runs_ = complete_candidates[plan.selected_candidate_];
    plan.total_ = totals[plan.selected_candidate_];
    return plan;
}

const std::vector<RunSpec>&
CasePlan::Runs() const {
    return runs_;
}

const Cost&
CasePlan::TotalCost() const {
    return total_;
}

uint64_t
CasePlan::Cap() const {
    return policy_.cap_points;
}

uint64_t
CasePlan::Seed() const {
    return seed_;
}

uint64_t
CasePlan::SelectedCandidate() const {
    return selected_candidate_;
}

void
CasePlan::Log(std::ostream& output) const {
    std::ostringstream line;
    line << "[test-budget] {\"schema_version\":1,\"model_version\":\"logical-v1\","
            "\"unit\":\"logical_points\",\"counts_kind\":\"declared_upper_bounds\","
            "\"case\":{\"name\":"
         << Quote(identity_.name) << ",\"tags\":" << Quote(identity_.tags)
         << ",\"source_file\":" << Quote(identity_.source_file)
         << ",\"source_line\":" << identity_.source_line << "},\"policy\":" << Quote(policy_.name)
         << ",\"cap_points\":" << Cap() << ",\"total_points\":" << total_.Total()
         << ",\"seed\":" << seed_ << ",\"selected_candidate\":" << selected_candidate_
         << ",\"run_count\":" << runs_.size() << ",\"phase_costs\":";
    LogCost(line, total_);
    line << ",\"runs\":[";
    bool first = true;
    for (const auto& run : runs_) {
        if (!first) {
            line << ',';
        }
        first = false;
        const auto& work = run.workload;
        const auto cost = EstimateCost(work);
        line << "{\"key\":" << Quote(run.key) << ",\"dim\":" << work.dim << ",\"num\":" << work.num
             << ",\"query_num\":" << work.query_num << ",\"train_num\":" << work.train_num
             << ",\"metric\":" << Quote(work.metric) << ",\"quantizer\":" << Quote(work.quantizer)
             << ",\"data_runs\":" << work.data_runs << ",\"train_runs\":" << work.train_runs
             << ",\"build_runs\":" << work.build_runs << ",\"search_runs\":" << work.search_runs
             << ",\"oracle_runs\":" << work.oracle_runs
             << ",\"serialize_runs\":" << work.serialize_runs
             << ",\"update_num\":" << work.update_num
             << ",\"train_iterations\":" << work.train_iterations
             << ",\"train_centroids\":" << work.train_centroids
             << ",\"build_candidates\":" << work.build_candidates
             << ",\"search_candidates\":" << work.search_candidates
             << ",\"repetitions\":" << work.repetitions << ",\"threads\":" << work.threads
             << ",\"min_dim\":" << work.min_dim << ",\"min_num\":" << work.min_num
             << ",\"min_train_num\":" << work.min_train_num << ",\"cost_points\":" << cost.Total()
             << ",\"phase_costs\":";
        LogCost(line, cost);
        line << '}';
    }
    line << "]}\n";
    output << line.str();
}

}  // namespace fixtures::budget
