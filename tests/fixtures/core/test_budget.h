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
#include <ostream>
#include <string>
#include <vector>

namespace fixtures::budget {

// Logical work points, deliberately not seconds or a runtime guarantee.
struct Cost {
    uint64_t data{0};
    uint64_t train{0};
    uint64_t build{0};
    uint64_t search{0};
    uint64_t oracle{0};
    uint64_t serialize{0};
    uint64_t update{0};

    uint64_t
    Total() const;

    Cost&
    operator+=(const Cost& other);
};

// Callers normalize model-specific requirements before constructing a candidate. Every count
// describes declared operation counts or conservative upper envelopes, including repeated builds
// and validation searches. These are model inputs, not observed operation counters.
struct Workload {
    uint64_t dim{1};
    uint64_t num{1};
    uint64_t query_num{0};
    uint64_t train_num{0};
    std::string metric{"l2"};
    std::string quantizer{"fp32"};

    uint64_t data_runs{1};
    uint64_t train_runs{0};
    uint64_t build_runs{0};
    uint64_t search_runs{0};
    uint64_t oracle_runs{0};
    uint64_t serialize_runs{0};
    uint64_t update_num{0};
    uint64_t train_iterations{1};
    uint64_t train_centroids{1};
    uint64_t build_candidates{1};
    uint64_t search_candidates{1};
    uint64_t repetitions{1};
    uint64_t threads{1};

    uint64_t min_dim{1};
    uint64_t min_num{1};
    uint64_t min_train_num{0};
};

struct RunSpec {
    std::string key;
    Workload workload;
};

struct CaseIdentity {
    std::string name;
    std::string tags;
    std::string source_file;
    uint64_t source_line{0};
};

struct Policy {
    std::string name{"pr"};
    uint64_t cap_points{2000000000ULL};
    uint64_t seed{47};

    static Policy
    PullRequest();

    static Policy
    Daily();
};

uint64_t
CaseSeed(const std::string& case_key, uint64_t global_seed = 47);

// Deterministic across standard libraries; no random_device or process-global RNG state.
uint64_t
SampleInclusive(uint64_t seed, uint64_t minimum, uint64_t maximum);

std::vector<uint64_t>
ShuffledIndices(uint64_t count, uint64_t seed);

uint64_t
MetricWeight(const std::string& metric);

uint64_t
QuantizerWeight(const std::string& quantizer);

Cost
EstimateCost(const Workload& workload);

class CasePlan {
public:
    const std::vector<RunSpec>&
    Runs() const;

    const Cost&
    TotalCost() const;

    uint64_t
    Cap() const;

    uint64_t
    Seed() const;

    uint64_t
    SelectedCandidate() const;

    // Emit once per registered testcase, after the complete plan has been selected.
    void
    Log(std::ostream& output) const;

private:
    friend CasePlan
    PlanCase(const CaseIdentity&, const Policy&, const std::vector<std::vector<RunSpec>>&);

    CaseIdentity identity_;
    Policy policy_;
    std::vector<RunSpec> runs_;
    Cost total_;
    uint64_t seed_{0};
    uint64_t selected_candidate_{0};
};

// Each candidate is a COMPLETE case, ordered from the smallest legal workload to the desired
// workload. Keys, metrics, quantizers and operation counts must retain the same coverage.
// All candidates are validated before selection. If the minimum does not fit, this throws;
// it never drops a run, skips an assertion or resets the cap for the next variant.
CasePlan
PlanCase(const CaseIdentity& identity,
         const Policy& policy,
         const std::vector<std::vector<RunSpec>>& complete_candidates);

}  // namespace fixtures::budget
