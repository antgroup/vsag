
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

#include <filesystem>

#include "./exporter_appendfile.h"
#include "./exporter_file.h"
#include "./exporter_influxdb.h"
#include "./exporter_stdout.h"

namespace vsag::eval {

bool
starts_with(const std::string& the_long_text, const std::string& the_short_text) {
    return the_long_text.compare(0, the_short_text.length(), the_short_text) == 0;
}

bool
is_file_path(const std::string& str) {
    return starts_with(str, "file://");
}

bool
is_appendfile_path(const std::string& str) {
    return starts_with(str, "appendfile://");
}

bool
is_influxdb(const std::string& str) {
    return starts_with(str, "influxdb://");
}

ExporterPtr
Exporter::Create(const std::string& to, const std::unordered_map<std::string, std::string>& vars) {
    if (to == "stdout") {
        return std::make_shared<StdoutExporter>();
    }
    if (is_file_path(to)) {
        return std::make_shared<FileExporter>(to);
    }
    if (is_appendfile_path(to)) {
        return std::make_shared<AppendfileExporter>(to);
    }
    if (is_influxdb(to)) {
        return std::make_shared<InfluxdbExporter>(to, vars);
    }
    abort();
}

}  // namespace vsag::eval
