
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

class IteratorContext {
public:
    virtual ~IteratorContext() = default;

    /**
      * @brief Write to the discard queue
      *
      * @param dis vector distance
      * @param id vector id
      */
    virtual void
    AddDiscardNode(float dis, uint32_t id){};

    /**
      * @brief Get the ID corresponding to the maximum discard distance
      */
    virtual uint32_t
    GetTopID() {
        return 0;
    };

    /**
      * @brief Get the maximum discard distance
      */
    virtual float
    GetTopDist() {
        return 0;
    };

    /**
      * @brief Discard elements out of the queue
      */
    virtual void
    PopDiscard(){};

    /**
      * @brief Determine whether the discard queue is empty
      *
      * @return true if vector is valid, otherwise false
      */
    virtual bool
    Empty() {
        return true;
    };

    /**
      * @brief When using ctx for the first time, you need to initialize it
      * to determine whether it is the first time to use it
      *
      * @return Returns whether ctx is used for the first time
      */
    virtual bool
    IsFirstUsed() {
        return true;
    };

    /**
      * @brief After use, set is_first_used to false
      */
    virtual void
    SetOFFFirstUsed(){};

    /**
      * @brief Mark the id of the result that has been returned
      *
      * @param id vector id
      */
    virtual void
    SetPoint(uint32_t id){};

    /**
      * @brief Check if it has been marked returned
      *
      * @param id vector id
      * @return Returns true if not marked.
      */
    virtual bool
    CheckPoint(uint32_t id) {
        return false;
    };

    /**
      * @brief Get the number of elements in discard
      *
      * @return return the number of elements in discard
      */
    virtual int64_t
    GetDiscardElementNum() {
        return false;
    };
};

using IteratorContextPtr = std::shared_ptr<IteratorContext>;

};  // namespace vsag