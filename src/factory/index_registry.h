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
#include <string>

#include "index_common_param.h"
#include "typing.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/index.h"

namespace vsag {

using IndexCreator = tl::expected<std::shared_ptr<Index>, Error> (*)(
    JsonType& parsed_params, const IndexCommonParam& index_common_params);

void
register_index_creator(const std::string& index_name, IndexCreator creator);

tl::expected<std::shared_ptr<Index>, Error>
create_registered_index(const std::string& index_name,
                        JsonType& parsed_params,
                        const IndexCommonParam& index_common_params);

}  // namespace vsag
