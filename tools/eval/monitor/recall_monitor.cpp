
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

#include "recall_monitor.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "../eval_dataset.h"
namespace vsag::eval {

static const double THRESHOLD_ERROR = 2e-6;

static double
get_recall(const float* distances,
           const float* ground_truth_distances,
           uint64_t recall_num,
           uint64_t top_k) {
    std::vector<float> gt_distances(ground_truth_distances, ground_truth_distances + top_k);
    std::sort(gt_distances.begin(), gt_distances.end());
    float threshold = gt_distances[top_k - 1];
    uint64_t count = 0;
    for (uint64_t i = 0; i < recall_num; ++i) {
        if (distances[i] <= threshold + THRESHOLD_ERROR) {
            ++count;
        }
    }
    return static_cast<double>(count) / static_cast<double>(top_k);
}

static double
get_id_recall(const int64_t* neighbors,
              const int64_t* ground_truth_neighbors,
              uint64_t recall_num,
              uint64_t top_k) {
    std::unordered_set<int64_t> expected(ground_truth_neighbors, ground_truth_neighbors + top_k);
    uint64_t count = 0;
    for (uint64_t i = 0; i < recall_num; ++i) {
        count += expected.erase(neighbors[i]);
    }
    return static_cast<double>(count) / static_cast<double>(top_k);
}

RecallMonitor::RecallMonitor(uint64_t max_record_counts, bool use_id_based_recall)
    : Monitor("recall_monitor"), use_id_based_recall_(use_id_based_recall) {
    if (max_record_counts > 0) {
        this->recall_records_.reserve(max_record_counts);
    }
}
void
RecallMonitor::Start() {
}

void
RecallMonitor::Stop() {
}

Monitor::JsonType
RecallMonitor::GetResult() {
    JsonType result;
    for (auto& metric : metrics_) {
        this->cal_and_set_result(metric, result);
    }
    return result;
}
void
RecallMonitor::Record(void* input) {
    std::lock_guard<std::mutex> lock(record_mutex_);

    const auto& record = *static_cast<const SearchRecord*>(input);
    const auto* neighbors = record.neighbors;
    const auto* gt_neighbors = record.ground_truth_neighbors;
    auto* dataset = record.dataset;
    const auto* query_data = record.query_data;
    const auto requested_top_k = record.requested_top_k;
    const auto result_count = std::min(record.result_count, requested_top_k);
    if (requested_top_k == 0) {
        this->recall_records_.emplace_back(0.0);
        return;
    }
    if (use_id_based_recall_) {
        this->recall_records_.emplace_back(
            get_id_recall(neighbors, gt_neighbors, result_count, requested_top_k));
        return;
    }
    uint64_t dim = dataset->GetDim();
    auto distance_func = dataset->GetDistanceFunc();
    std::vector<float> gt_distances(requested_top_k);
    std::vector<float> distances(result_count);
    for (uint64_t i = 0; i < result_count; ++i) {
        const auto* result_vector = dataset->GetOneTrainById(neighbors[i]);
        distances[i] = result_vector == nullptr ? std::numeric_limits<float>::infinity()
                                                : distance_func(query_data, result_vector, &dim);
    }
    for (uint64_t i = 0; i < requested_top_k; ++i) {
        const auto* ground_truth_vector = dataset->GetOneTrainById(gt_neighbors[i]);
        gt_distances[i] = ground_truth_vector == nullptr
                              ? std::numeric_limits<float>::infinity()
                              : distance_func(query_data, ground_truth_vector, &dim);
    }

    const double val =
        get_recall(distances.data(), gt_distances.data(), result_count, requested_top_k);
    this->recall_records_.emplace_back(val);
}
void
RecallMonitor::SetMetrics(std::string metric) {
    this->metrics_.emplace_back(std::move(metric));
}

void
RecallMonitor::cal_and_set_result(const std::string& metric, Monitor::JsonType& result) {
    if (metric == "avg_recall") {
        auto val = this->cal_avg_recall();
        result["recall_avg"] = val;
    } else if (metric == "percent_recall") {
        std::vector<double> percents = {0, 10, 30, 50, 70, 90};
        for (auto& percent : percents) {
            auto val = this->cal_recall_rate(percent * 0.01);
            result["recall_detail"]["p" + std::to_string(int(percent))] = val;
        }
    }
}

double
RecallMonitor::cal_avg_recall() {
    double sum =
        std::accumulate(this->recall_records_.begin(), this->recall_records_.end(), double(0));
    return sum / static_cast<double>(recall_records_.size());
}

double
RecallMonitor::cal_recall_rate(double rate) {
    std::sort(this->recall_records_.begin(), this->recall_records_.end());
    auto pos = static_cast<uint64_t>(rate * static_cast<double>(this->recall_records_.size() - 1));
    return recall_records_[pos];
}
}  // namespace vsag::eval
