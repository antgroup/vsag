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
 * @file temp_dir.h
 * @brief Temporary directory management for tests.
 */

#pragma once

#include <filesystem>
#include <string>

namespace fixtures {

/**
 * @class TempDir
 * @brief RAII wrapper for creating and automatically cleaning up a temporary directory.
 * The directory is created in /tmp with a unique name and removed when destroyed.
 */
class TempDir {
public:
    /**
     * @brief Creates a temporary directory with the given prefix.
     * @param prefix Prefix for the directory name.
     */
    explicit TempDir(const std::string& prefix);

    /**
     * @brief Destructor that removes the temporary directory and its contents.
     */
    ~TempDir();

    /**
     * @brief Generates a random file path within the temporary directory.
     * @param create_file If true, creates an empty file at the path.
     * @return Full path to the generated file.
     */
    [[nodiscard]] std::string
    GenerateRandomFile(bool create_file = true) const;

    std::string path;  // Full path to the temporary directory.
};

}  // namespace fixtures
