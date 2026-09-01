
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

#include <cstdint>
#include <string_view>

#include "parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(IOParam, IOParameter);

enum class IOKind : uint8_t {
    UNKNOWN,
    MEMORY,
    BLOCK_MEMORY,
    MMAP,
    BUFFER,
    ASYNC,
    URING,
    READER,
};

/**
 * @brief Base class for IO configuration parameters.
 *
 * This class serves as the base for all IO-related parameter classes,
 * providing a common interface for parameter creation from JSON configuration
 * and type identification.
 */
class IOParameter : public Parameter {
public:
    static IOParamPtr
    CreateDefault(const std::string& type_name);

    /**
     * @brief Creates an IO parameter object from JSON configuration.
     *
     * @param json The JSON object containing IO configuration.
     * @return A shared pointer to the created IOParameter.
     */
    static IOParamPtr
    GetIOParameterByJson(const JsonType& json);

    [[nodiscard]] static IOKind
    KindFromName(std::string_view name);

    void
    LoadReadCacheConfig(const JsonType& json);

    void
    AppendReadCacheConfig(JsonType& json) const;

    bool enable_read_cache_{false};
    uint64_t read_cache_total_size_{256ULL * 1024 * 1024};

public:
    /**
     * @brief Returns the type name of this IO parameter.
     *
     * @return The name string identifying the IO type.
     */
    inline const std::string&
    GetTypeName() const {
        return this->name_;
    }

    [[nodiscard]] IOKind
    Kind() const;

protected:
    /**
     * @brief Constructs an IOParameter with a type name.
     *
     * @param name The type name for this IO parameter.
     */
    explicit IOParameter(std::string name);

    /**
     * @brief Default destructor.
     */
    ~IOParameter() override = default;

private:
    /// Type name identifying the IO implementation type.
    std::string name_{};
};

}  // namespace vsag
