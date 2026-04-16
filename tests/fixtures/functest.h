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

/**
 * @file functest.h
 * @brief Entry point header for functional tests, includes all test utilities.
 */

#pragma once

#include <fstream>
#include <stdexcept>
#include <string>

#include "allocator/memory_record_allocator.h"
#include "allocator/random_allocator.h"
#include "framework/recall_checker.h"
#include "framework/test_dataset.h"
#include "framework/test_dataset_pool.h"
#include "framework/test_logger.h"
#include "framework/test_reader.h"
#include "framework/test_thread_pool.h"
#include "unittest.h"

namespace fixtures {

template <typename T>
void
test_serialization_file(T& old_instance, T& new_instance, const std::string name) {
    auto temp_dir = TempDir(name);
    auto file = temp_dir.GenerateRandomFile();
    std::ofstream ofs(file);
    old_instance.Serialize(ofs);
    ofs.close();

    std::ifstream ifs(file);
    auto value = new_instance.Deserialize(ifs);
    ifs.close();
    if (not value.has_value()) {
        throw std::runtime_error("deserialize failed: " + value.error().message);
    }
}

template <typename T>
void
test_serializion_file(T& old_instance, T& new_instance, const std::string name) {
    test_serialization_file(old_instance, new_instance, name);
}

}  // namespace fixtures
