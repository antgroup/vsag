
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

#include "./formatter.h"
#include "fmt/core.h"

namespace vsag::eval {

int64_t
current_time_in_ns() {
    using namespace std::chrono;
    time_point<system_clock, nanoseconds> tp = time_point_cast<nanoseconds>(system_clock::now());
    return duration_cast<nanoseconds>(tp.time_since_epoch()).count();
}

class LineProtocolFormatter : public Formatter {
public:
    std::string
    Format(vsag::eval::JsonType& results) override {
        auto now = current_time_in_ns();

        std::string rows;
        for (const auto& [key, value] : results.items()) {
            JSON_GET(index_name, value["index"], "-1");
            JSON_GET(
                num_vectors, std::to_string(value["dataset_info"]["base_count"].get<int>()), "N/A");
            JSON_GET(dim, std::to_string(value["dataset_info"]["dim"].get<int>()), "-1");
            JSON_GET(data_type, value["dataset_info"]["data_type"], "-1");
            JSON_GET(metric_type, value["index_info"]["metric_type"], "-1");
            JSON_GET(index_param, value["index_info"]["index_param"].dump(), "-1");
            JSON_GET(build_time, std::to_string(value["duration(s)"].get<float>()), "-1");
            JSON_GET(tps, std::to_string(value["tps"].get<float>()), "-1");

            JSON_GET(search_mode, value["search_mode"], "-1");
            JSON_GET(search_param, value["search_param"], "-1");
            JSON_GET(qps, std::to_string(value["qps"].get<float>()), "-1");
            JSON_GET(latency_avg, std::to_string(value["latency_avg(ms)"].get<float>()), "-1");
            JSON_GET(
                latency_p90, std::to_string(value["latency_detail(ms)"]["p90"].get<float>()), "-1");
            JSON_GET(
                latency_p95, std::to_string(value["latency_detail(ms)"]["p95"].get<float>()), "-1");
            JSON_GET(
                latency_p99, std::to_string(value["latency_detail(ms)"]["p99"].get<float>()), "-1");
            JSON_GET(recall_avg, std::to_string(value["recall_avg"].get<float>()), "-1");
            // JSON_GET(recall_p50, std::to_string(value["recall_detail"]["p50"].get<float>()), "-1");
            // JSON_GET(recall_p90, std::to_string(value["recall_detail"]["p90"].get<float>()), "-1");

            constexpr static const char* row_template =
                "performance,case_name={0},index_name={1},dtype={2}"
                " "
                "num_vectors={3},dim={4},build_time={5},tps={6},qps={7}"
                ",latency_avg={8},latency_p90={9},latency_p95={10},latency_p99={11},recall_avg={12}"
                " "
                "{13}";
            auto row = fmt::format(row_template,
                                   key,
                                   index_name,
                                   data_type,
                                   num_vectors,
                                   dim,
                                   build_time,
                                   tps,
                                   qps,
                                   latency_avg,
                                   latency_p90,
                                   latency_p95,
                                   latency_p99,
                                   recall_avg,
                                   now);
            rows.append(row);
            rows.append("\n");
        }

        return rows;
    }
};

}  // namespace vsag::eval
