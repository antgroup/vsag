
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

#include <cpr/cpr.h>

#include <iostream>
#include <string>

#include "./exporter.h"

namespace vsag::eval {

std::string
replace_string(std::string text, const std::string& from, const std::string& to) {
    while (text.find(from) != std::string::npos) {
        text.replace(text.find(from), from.length(), to);
    }
    return text;
}

class InfluxdbExporter : public Exporter {
public:
    bool
    Export(const std::string& formatted_result) override {
        cpr::Header header1 = cpr::Header{{"Authorization", token_},
                                          {"Content-Type", "text/plain; charset=utf-8"},
                                          {"Accept", "application/json"}};
        cpr::Response r = cpr::Post(cpr::Url{endpoint_}, header1, cpr::Body{formatted_result});
        if (r.status_code == 204) {
            return true;
        }
        // TODO(wxyu): use vsag logger
        std::cerr << r.text << std::endl;

        return false;
    }

public:
    InfluxdbExporter(const std::string& endpoint,
                     const std::unordered_map<std::string, std::string>& vars) {
        endpoint_ = replace_string(endpoint, "influxdb", "http");
        token_ = vars.at("token");
    }

private:
    // e.g., "http://127.0.0.1:8086/api/v2/write?org=vsag&bucket=example&precision=ns"
    std::string endpoint_{};
    // e.g., "Token mlIiP-zVfcooHhMbGG9Yk-KfrkHyDc2h-rphnIBda8UMe_6Qocy8tNmV323yxOPEAsC8uIs6_nb-XUSMEAO76A=="
    std::string token_{};
};

}  // namespace vsag::eval
