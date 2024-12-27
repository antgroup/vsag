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
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

#include "allocator.h"
#include "constants.h"

namespace vsag {

class Dataset;
using DatasetPtr = std::shared_ptr<Dataset>;

/**
 * @class Dataset
 *
 * @brief This class represents a dataset with various attributes such as
 * dimensions, ids, vectors, and more. It serves a dual purpose:
 * - As an input provider for index.
 * - As a container for storing search results.
 */
class Dataset : public std::enable_shared_from_this<Dataset> {
public:
    /**
     * @brief Creates a new instance of a Dataset.
     *
     * @return DatasetPtr A shared pointer to the new dataset instance.
     */
    static DatasetPtr
    Make();

    virtual ~Dataset() = default;

    /**
     * @brief Configures ownership and sets an optional allocator.
     *
     * @param is_owner Boolean indicating if this dataset is owned.
     * @param allocator An allocator for resource management.
     * @return DatasetPtr A shared pointer to the configured dataset instance.
     */
    virtual DatasetPtr
    Owner(bool is_owner, Allocator* allocator) = 0;

    /**
     * @brief Configures ownership.
     * No provided outside allocator to allocate or deallocate
     *
     * @param is_owner Boolean indicating if this dataset is owned.
     * @return DatasetPtr A shared pointer to the configured dataset instance.
     */
    DatasetPtr
    Owner(bool is_owner) {
        return Owner(is_owner, nullptr);
    }

public:
    /**
     * @brief Sets the number of elements in the dataset.
     *
     * @param num_elements The number of elements.
     * @return DatasetPtr A shared pointer to the dataset with updated number of elements.
     */
    virtual DatasetPtr
    NumElements(int64_t num_elements) = 0;

    /**
     * @brief Retrieves the number of elements in the dataset.
     *
     * @return int64_t The number of elements.
     */
    virtual int64_t
    GetNumElements() const = 0;

    /**
     * @brief Sets the dimensionality of the dataset.
     *
     * @param dim The dimensionality value.
     * @return DatasetPtr A shared pointer to the dataset with updated dimensionality.
     */
    virtual DatasetPtr
    Dim(int64_t dim) = 0;

    /**
     * @brief Retrieves the dimensionality of the dataset.
     *
     * @return int64_t The dimensionality.
     */
    virtual int64_t
    GetDim() const = 0;

    /**
     * @brief Sets the ID array for the dataset.
     *
     * @param ids Pointer to the array of IDs.
     * @return DatasetPtr A shared pointer to the dataset with set IDs.
     */
    virtual DatasetPtr
    Ids(const int64_t* ids) = 0;

    /**
     * @brief Retrieves the ID array of the dataset.
     *
     * @return const int64_t* pointer to the array of IDs.
     */
    virtual const int64_t*
    GetIds() const = 0;

    /**
     * @brief Sets the distances array for the dataset.
     *
     * @param dists Pointer to the array of distances.
     * @return DatasetPtr A shared pointer to the dataset with updated distances.
     */
    virtual DatasetPtr
    Distances(const float* dists) = 0;

    /**
     * @brief Retrieves the distances array of the dataset.
     *
     * @return const float* Pointer to the array of distances.
     */
    virtual const float*
    GetDistances() const = 0;

    /**
     * @brief Sets the int8 vector array for the dataset.
     *
     * @param vectors Pointer to the array of int8 vectors.
     * @return DatasetPtr A shared pointer to the dataset with updated int8 vectors.
     */
    virtual DatasetPtr
    Int8Vectors(const int8_t* vectors) = 0;

    /**
     * @brief Retrieves the int8 vector array of the dataset.
     *
     * @return const int8_t* Pointer to the array of int8 vectors.
     */
    virtual const int8_t*
    GetInt8Vectors() const = 0;

    /**
     * @brief Sets the float32 vector array for the dataset.
     *
     * @param vectors Pointer to the array of float32 vectors.
     * @return DatasetPtr A shared pointer to the dataset with updated float32 vectors.
     */
    virtual DatasetPtr
    Float32Vectors(const float* vectors) = 0;

    /**
     * @brief Retrieves the float32 vector array of the dataset.
     *
     * @return const float* Pointer to the array of float32 vectors.
     */
    virtual const float*
    GetFloat32Vectors() const = 0;

    /**
     * @brief Sets the paths array for the dataset.
     *
     * @param paths Pointer to the array of paths.
     * @return DatasetPtr A shared pointer to the dataset with updated paths.
     */
    virtual DatasetPtr
    Paths(const std::string* paths) = 0;

    /**
     * @brief Retrieves the paths array of the dataset.
     *
     * @return const std::string* Pointer to the array of paths.
     */
    virtual const std::string*
    GetPaths() const = 0;
};

};  // namespace vsag
