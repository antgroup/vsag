
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

#include <memory>
#include <unordered_map>

namespace vsag::eval {

class Exporter;
using ExporterPtr = std::shared_ptr<Exporter>;

class Exporter {
public:
    static ExporterPtr
    Create(const std::string& to, const std::unordered_map<std::string, std::string>& vars);

    virtual bool
    Export(const std::string& result) = 0;

protected:
    Exporter() = default;
    virtual ~Exporter() = default;
};

}  // namespace vsag::eval
