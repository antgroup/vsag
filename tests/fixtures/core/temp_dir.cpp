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

#include "temp_dir.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "random.h"

namespace fixtures {

TempDir::TempDir(const std::string& prefix) {
    namespace fs = std::filesystem;
    std::stringstream dirname;
    do {
        auto epoch_time = std::chrono::system_clock::now().time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch_time).count();

        int random_number = RandomValue<int>(1000, 9999);

        dirname << "vsagtest_" << prefix << "_" << std::setfill('0') << std::setw(14) << seconds
                << "_" << std::to_string(random_number);
        path = (fs::temp_directory_path() / dirname.str()).string() + "/";
        dirname.clear();
    } while (fs::exists(path));

    std::filesystem::create_directory(path);
}

TempDir::~TempDir() {
    std::filesystem::remove_all(path);
}

std::string
TempDir::GenerateRandomFile(bool create_file) const {
    namespace fs = std::filesystem;
    const std::string chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string fileName;
    do {
        fileName = "";
        for (int i = 0; i < 10; i++) {
            fileName += chars[RandomValue<uint64_t>(0, chars.length() - 1)];
        }
    } while (fs::exists(path + fileName));

    if (create_file) {
        std::ofstream file(path + fileName);
        if (file.is_open()) {
            file.close();
        }
    }
    return path + fileName;
}

}  // namespace fixtures
