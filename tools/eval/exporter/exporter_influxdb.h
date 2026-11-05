
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

#include <httplib.h>

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
        httplib::Client client(host_.c_str());
        client.set_default_headers({{"Authorization", token_},
                                    {"Content-Type", "text/plain; charset=utf-8"},
                                    {"Accept", "application/json"}});

        auto res = client.Post(path_.c_str(), formatted_result, "text/plain");
        if (res && res->status == 204) {
            return true;
        }
        // TODO(wxyu): use vsag logger
        if (res) {
            std::cerr << "HTTP Error: " << res->status << " - " << res->body << std::endl;
        } else {
            std::cerr << "HTTP Request failed" << std::endl;
        }

        return false;
    }

public:
    InfluxdbExporter(const std::string& endpoint,
                     const std::unordered_map<std::string, std::string>& vars) {
        // Parse endpoint from "http://host:port/path?query" to host and path
        // e.g., "http://127.0.0.1:8086/api/v2/write?org=vsag&bucket=example&precision=ns"
        std::string full_url = replace_string(endpoint, "influxdb", "http");
        ParseEndpoint(full_url);
        token_ = vars.at("token");
    }

private:
    void
    ParseEndpoint(const std::string& full_url) {
        // Find the path separator after host:port
        // Format: http://host:port/path or http://host/path
        size_t protocol_end = full_url.find("://");
        if (protocol_end == std::string::npos) {
            host_ = full_url;
            path_ = "/";
            return;
        }

        size_t path_start = full_url.find('/', protocol_end + 3);
        if (path_start == std::string::npos) {
            host_ = full_url;
            path_ = "/";
        } else {
            host_ = full_url.substr(0, path_start);
            path_ = full_url.substr(path_start);
        }
    }

private:
    // Host part, e.g., "http://127.0.0.1:8086"
    std::string host_{};
    // Path part, e.g., "/api/v2/write?org=vsag&bucket=example&precision=ns"
    std::string path_{};
    // e.g., "Token mlIiP-zVfcooHhMbGG9Yk-KfrkHyDc2h-rphnIBda8UMe_6Qocy8tNmV323yxOPEAsC8uIs6_nb-XUSMEAO76A=="
    std::string token_{};
};

}  // namespace vsag::eval
