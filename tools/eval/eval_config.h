
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

#include "argparse/argparse.hpp"

namespace vsag::eval {
class EvalConfig {
public:
    static EvalConfig
    Load(argparse::ArgumentParser& parser);

public:
    std::string dataset_path;
    std::string action_type;
    std::string index_name;
    std::string build_param;
    std::string index_path;

    std::string search_param;
    std::string search_mode;
    int top_k{10};
    float radius{0.5f};

    bool enable_recall{true};
    bool enable_percent_recall{true};
    bool enable_qps{true};
    bool enable_tps{true};
    bool enable_memory{true};
    bool enable_latency{true};
    bool enable_percent_latency{true};

private:
    EvalConfig() = default;
};

}  // namespace vsag::eval
