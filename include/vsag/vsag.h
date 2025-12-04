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

namespace vsag {

/**
  * @brief Get the version based on git revision
  * 
  * @return the version text
  */
extern std::string
version();

/**
  * @brief Init the vsag library
  * 
  * @return true always
  */
extern bool
init();

}  // namespace vsag

#include "vsag/allocator.h"
#include "vsag/attribute.h"
#include "vsag/binaryset.h"
#include "vsag/bitset.h"
#include "vsag/constants.h"
#include "vsag/dataset.h"
#include "vsag/engine.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/factory.h"
#include "vsag/index.h"
#include "vsag/index_detail_info.h"
#include "vsag/index_features.h"
#include "vsag/iterator_context.h"
#include "vsag/logger.h"
#include "vsag/options.h"
#include "vsag/readerset.h"
#include "vsag/resource.h"
#include "vsag/search_request.h"
#include "vsag/thread_pool.h"
#include "vsag/utils.h"
