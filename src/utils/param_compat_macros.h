
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

#include "impl/logger/logger.h"

// Cast `other` to the target parameter type; log and return false on type mismatch.
#define PARAM_CAST_OR_RETURN(Type, var, other)                      \
    auto var = std::dynamic_pointer_cast<Type>((other));            \
    if (not var) {                                                  \
        logger::error(#Type "::CheckCompatibility: type mismatch"); \
        return false;                                               \
    }

// Compare a scalar field; log the field name and return false on mismatch.
#define CHECK_FIELD_EQ(self, other, field) \
    if ((self).field != (other).field) {   \
        logger::error(#field " mismatch"); \
        return false;                      \
    }

// Recursively check a sub-parameter; log the field name and return false on incompatibility.
// Safely handles null pointers: both null is compatible, one null is incompatible.
#define CHECK_SUB_PARAM(self, other, field)                                   \
    if (((self).field == nullptr) != ((other).field == nullptr)) {            \
        logger::error(#field " incompatible (null mismatch)");                \
        return false;                                                         \
    }                                                                         \
    if ((self).field && not(self).field->CheckCompatibility((other).field)) { \
        logger::error(#field " incompatible");                                \
        return false;                                                         \
    }
