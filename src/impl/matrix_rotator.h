
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

#include "stream_reader.h"
#include "stream_writer.h"

class MatrixRotator {
public:
    MatrixRotator() = default;
    virtual ~MatrixRotator() = default;

    virtual void
    Transform(const float* original_vec, float* transformed_vec) const = 0;

    virtual void
    InverseTransform(const float* transformed_vec, float* original_vec) const = 0;

    virtual bool
    Build() = 0;

    virtual void
    Serialize(StreamWriter& writer) = 0;

    virtual void
    Deserialize(StreamReader& reader) = 0;
};
