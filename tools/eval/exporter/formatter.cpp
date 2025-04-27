
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

#include "formatter.h"

#include <memory>
#include <string>

#include "../common.h"
#include "formatter_json.h"
#include "formatter_lineproto.h"
#include "formatter_table.h"

namespace vsag::eval {

FormatterPtr
Formatter::Create(const std::string& format) {
    if (format == "json") {
        return std::make_shared<JsonFormatter>();
    }
    if (format == "text" or format == "table") {
        return std::make_shared<TableFormatter>();
    }
    if (format == "line_protocol") {
        return std::make_shared<LineProtocolFormatter>();
    }
    abort();
}

}  // namespace vsag::eval
