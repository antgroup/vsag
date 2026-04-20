
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

#include "parameter.h"
#include "utils/pointer_define.h"

namespace vsag {
DEFINE_POINTER2(IOParam, IOParameter);

/**
 * @brief Base class for IO configuration parameters.
 *
 * This class serves as the base for all IO-related parameter classes,
 * providing a common interface for parameter creation from JSON configuration
 * and type identification.
 */
class IOParameter : public Parameter {
public:
    /**
     * @brief Creates an IO parameter object from JSON configuration.
     *
     * @param json The JSON object containing IO configuration.
     * @return A shared pointer to the created IOParameter.
     */
    static IOParamPtr
    GetIOParameterByJson(const JsonType& json);

public:
    /**
     * @brief Returns the type name of this IO parameter.
     *
     * @return The name string identifying the IO type.
     */
    inline std::string
    GetTypeName() {
        return this->name_;
    }

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
