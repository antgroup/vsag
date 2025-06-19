
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

#include <string>
#include <tabulate/tabulate.hpp>

#include "./formatter.h"

namespace vsag::eval {

class TableFormatter : public Formatter {
public:
    std::string
    Format(vsag::eval::JsonType& results) override {
        using namespace tabulate;
        Table table;
        table.add_row({"Name",
                       "Index",
                       "NumVectors",
                       "Dim",
                       "DataType",
                       "MetricType",
                       "IndexParam",
                       "Memory(build)",
                       "BuildTime",
                       "TPS",
                       "SearchParam",
                       "Memory(search)",
                       "QPS",
                       "LatencyAvg(ms)",
                       "RecallAvg"});
        for (const auto& [key, value] : results.items()) {
            JSON_GET(index_name, value["index"], "N/A");
            JSON_GET(
                num_vectors, std::to_string(value["dataset_info"]["base_count"].get<int>()), "N/A");
            JSON_GET(dim, std::to_string(value["dataset_info"]["dim"].get<int>()), "N/A");
            JSON_GET(data_type, value["dataset_info"]["data_type"], "N/A");
            JSON_GET(metric_type, value["index_info"]["metric_type"], "N/A");
            JSON_GET(index_param, value["index_info"]["index_param"].dump(), "N/A");
            JSON_GET(memory_build, value["memory_peak(build)"], "N/A");
            JSON_GET(build_time, std::to_string(value["duration(s)"].get<float>()), "N/A");
            JSON_GET(tps, std::to_string(value["tps"].get<float>()), "N/A");
            JSON_GET(search_param, value["search_param"], "N/A");
            JSON_GET(memory_search, value["memory_peak(search)"], "N/A");
            JSON_GET(qps, std::to_string(value["qps"].get<float>()), "N/A");
            JSON_GET(latency_avg, std::to_string(value["latency_avg(ms)"].get<float>()), "N/A");
            JSON_GET(recall_avg, std::to_string(value["recall_avg"].get<float>()), "N/A");

            table.add_row({key,
                           index_name,
                           num_vectors,
                           dim,
                           data_type,
                           metric_type,
                           index_param,
                           memory_build,
                           build_time,
                           tps,
                           search_param,
                           memory_search,
                           qps,
                           latency_avg,
                           recall_avg});
        }

        table.column(6).format().width(40);
        return table.str();
    }
};

}  // namespace vsag::eval
