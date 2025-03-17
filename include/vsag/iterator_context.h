
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
    virtual void
    AddDiscardNode(float dis, uint32_t id) {};
    virtual uint32_t
    GetTopID() {
        return 0;
    };
    virtual float
    GetTopDist() {
        return 0;
    };
    virtual void
    PopDiscard() {};
    virtual bool
    Empty() {
        return true;
    };
    virtual bool
    IsFirstUsed() {
        return true;
    };
    virtual void
    SetOFFFirstUsed() {};
    virtual void
    SetPoint(uint32_t id) {};
    virtual bool
    CheckPoint(uint32_t id) {
        return false;
    };
    virtual int64_t
    GetDiscardElementNum() {
        return false;
    };
};

using IteratorContextPtr = std::shared_ptr<IteratorContext>;

};  // namespace vsag