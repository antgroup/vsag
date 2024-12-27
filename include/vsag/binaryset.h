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

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vsag {
/**
 * @struct Binary
 * @brief A structure representing binary data and its size.
 */
struct Binary {
    std::shared_ptr<int8_t[]> data;  ///< The binary data.
    size_t size = 0;                 ///< The size of the binary data.
};

/**
 * @class BinarySet
 * @brief A class to store and manage Binary objects. Always used for serialize
 */
class BinarySet {
public:
    /// Default constructor.
    BinarySet() = default;

    /// Default destructor.
    ~BinarySet() = default;

    /**
     * @brief Stores a Binary object with a specified name.
     *
     * @param name The name associated with the Binary object.
     * @param binary The Binary object to be stored.
     */
    void
    Set(const std::string& name, Binary binary) {
        data_[name] = std::move(binary);
    }

    /**
     * @brief Retrieves a Binary object by name.
     *
     * @param name The name associated with the Binary object.
     * @return The Binary object retrieved, or an empty Binary object if not found.
     */
    Binary
    Get(const std::string& name) const {
        if (data_.find(name) == data_.end()) {
            return Binary();
        }
        return data_.at(name);
    }

    /**
     * @brief Gets a list of all stored names.
     *
     * @return A vector of all names in the BinarySet.
     */
    std::vector<std::string>
    GetKeys() const {
        std::vector<std::string> keys;
        keys.resize(data_.size());

        std::transform(data_.begin(),
                       data_.end(),
                       keys.begin(),
                       [](const std::pair<std::string, Binary>& pair) { return pair.first; });

        return keys;
    }

    /**
     * @brief Checks if a Binary object with the specified name exists.
     *
     * @param key The name of the Binary object to check for.
     * @return True if the Binary object exists, false otherwise.
     */
    bool
    Contains(const std::string& key) const {
        return data_.find(key) != data_.end();
    }

private:
    std::unordered_map<std::string, Binary> data_;  ///< The map storing Binary objects by name.
};

}  // namespace vsag
