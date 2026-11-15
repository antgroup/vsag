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

#include "vsag/dataset.h"

namespace vsag {

/**
 * @brief Opt-in extension interface for packed binary vector storage on a Dataset.
 *
 * The public Dataset interface exposes BinaryVectors()/GetBinaryVectors() as non-virtual
 * accessors so that adding binary support keeps the existing virtual table layout intact.
 * A Dataset implementation that stores packed binary vectors additionally derives from this
 * interface, and the accessors dispatch to it through dynamic_cast.
 */
class BinaryDatasetAccessor {
public:
    virtual ~BinaryDatasetAccessor() = default;

    virtual DatasetPtr
    SetBinaryVectorsInternal(const uint8_t* vectors) = 0;

    [[nodiscard]] virtual const uint8_t*
    GetBinaryVectorsInternal() const = 0;
};

}  // namespace vsag
