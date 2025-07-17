
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

namespace vsag {

class Filter {
public:
    enum class Distribution {
        NONE = 0,
        RELATED_TO_VECTOR,
    };

public:
    virtual ~Filter() = default;

    /**
      * @brief Check if a vector is filtered out by pre-filter, true means
      * not been filtered out, false means have been filtered out, the result
      * of KnnSearch/RangeSearch will only contain non-filtered-out vectors
      * 
      * @param id of the vector
      * @return true if vector is valid, otherwise false
      */
    [[nodiscard]] virtual bool
    CheckValid(int64_t id) const = 0;

    /**
      * @brief Check if a vector is filtered out by pre-filter, true means
      * not been filtered out, false means have been filtered out, the result
      * of KnnSearch/RangeSearch will only contain non-filtered-out vectors
      * 
      * @param data extra info of the vector
      * @return true if vector is valid, otherwise false
      */
    [[nodiscard]] virtual bool
    CheckValid(const char* data) const {
        return true;
    }

    /**
      * @brief Get valid ratio of pre-filter, 1.0 means all the vectors valid, 
      * none of them have been filter out.
      * 
      * @return the valid ratio
      */
    [[nodiscard]] virtual float
    ValidRatio() const {
        return 1.0f;  // (default) all vectors is valid
    }

    /**
      * @brief Get the distribution of pre-filter
      * 
      * @return distribution type of this filter
      */
    [[nodiscard]] virtual Distribution
    FilterDistribution() const {
        return Distribution::NONE;  // (default) no distribution information hints provides
    }
};

using FilterPtr = std::shared_ptr<Filter>;

};  // namespace vsag
