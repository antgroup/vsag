
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

#include <mutex>
#include <unordered_set>

#include "../eval_dataset.h"
namespace vsag::eval {

static const double THRESHOLD_ERROR = 2e-6;

static double
get_recall(const float* distances,
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
    std::lock_guard<std::mutex> lock(record_mutex_);

    auto [neighbors, gt_neighbors, dataset, query_data, topk] =
        *(reinterpret_cast<std::tuple<int64_t*, int64_t*, EvalDataset*, const void*, uint64_t>*>(
            input));
    size_t dim = dataset->GetDim();
    auto distance_func = dataset->GetDistanceFunc();
    auto gt_distances = std::shared_ptr<float[]>(new float[topk]);
    auto distances = std::shared_ptr<float[]>(new float[topk]);
    for (int i = 0; i < topk; ++i) {
        distances[i] = distance_func(query_data, dataset->GetOneTrain(neighbors[i]), &dim);
        gt_distances[i] = distance_func(query_data, dataset->GetOneTrain(gt_neighbors[i]), &dim);
    }

    float val = 0;
    if (topk != 0) {
        val = get_recall(distances.get(), gt_distances.get(), topk, topk);
    }
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
