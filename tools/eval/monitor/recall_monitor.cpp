
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

#include <iostream>
#include <unordered_set>
namespace vsag::eval {

static double
get_recall(const int64_t* neighbors, const int64_t* ground_truth, size_t recall_num, size_t top_k) {
    std::unordered_set<int64_t> neighbors_set(neighbors, neighbors + recall_num);
    std::unordered_set<int64_t> intersection;
    for (size_t i = 0; i < top_k; ++i) {
        if (i < top_k && neighbors_set.count(ground_truth[i])) {
            intersection.insert(ground_truth[i]);
        }
    }
    return static_cast<double>(intersection.size()) / static_cast<double>(top_k);
}

static const double THRESHOLD_ERROR = 1e-5;

static double
get_recall_by_distance(const float* distances,
                       const float* ground_truth_distances,
                       size_t recall_num,
                       size_t top_k) {
    std::vector<float> gt_distances(ground_truth_distances, ground_truth_distances + top_k);
    std::sort(gt_distances.begin(), gt_distances.end());
    float threshold = gt_distances[top_k - 1];
    size_t count = 0;
    for (size_t i = 0; i < recall_num; ++i) {
        if (distances[i] <= threshold + THRESHOLD_ERROR) {
            ++count;
        }
    }
    return static_cast<double>(count) / static_cast<double>(top_k);
}

RecallMonitor::RecallMonitor(uint64_t max_record_counts) : Monitor("recall_monitor") {
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
    auto [gt_neighbors, neighbors, gt_distances, distances, topk] =
        *(reinterpret_cast<std::tuple<int64_t*, int64_t*, float*, float*, uint64_t>*>(input));
    auto id_recall_val = get_recall(neighbors, gt_neighbors, topk, topk);
    auto distance_recall_val = get_recall_by_distance(distances, gt_distances, topk, topk);
    this->recall_records_.emplace_back(id_recall_val);
    this->distance_recall_records_.emplace_back(distance_recall_val);
}
void
RecallMonitor::SetMetrics(std::string metric) {
    this->metrics_.emplace_back(std::move(metric));
}
void
RecallMonitor::cal_and_set_result(const std::string& metric, Monitor::JsonType& result) {
    if (metric == "avg_recall") {
        auto [val, distance_recall_val] = this->cal_avg_recall();
        result["recall_avg_by_id"] = val;
        result["recall_avg_by_distance"] = distance_recall_val;
    } else if (metric == "percent_recall") {
        std::vector<double> percents = {0, 10, 30, 50, 70, 90};
        for (auto& percent : percents) {
            auto [distance_recall_val, id_recall_val] = this->cal_recall_rate(percent * 0.01);
            result["recall_detail_by_id"]["p" + std::to_string(int(percent))] = id_recall_val;
            result["recall_detail_by_distance"]["p" + std::to_string(int(percent))] =
                distance_recall_val;
        }
    }
}

std::tuple<double, double>
RecallMonitor::cal_avg_recall() {
    double ids_recall_sum =
        std::accumulate(this->recall_records_.begin(), this->recall_records_.end(), double(0));
    double distance_recall_sum = std::accumulate(
        this->distance_recall_records_.begin(), this->distance_recall_records_.end(), double(0));
    double count = static_cast<double>(recall_records_.size());
    return std::make_tuple(ids_recall_sum / count, distance_recall_sum / count);
}

std::tuple<double, double>
RecallMonitor::cal_recall_rate(double rate) {
    std::sort(this->recall_records_.begin(), this->recall_records_.end());
    std::sort(this->distance_recall_records_.begin(), this->distance_recall_records_.end());
    auto pos = static_cast<uint64_t>(rate * static_cast<double>(this->recall_records_.size() - 1));
    return std::make_tuple(recall_records_[pos], distance_recall_records_[pos]);
}
}  // namespace vsag::eval
