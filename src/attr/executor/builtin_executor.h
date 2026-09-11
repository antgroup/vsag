// Copyright 2024-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include "executor.h"

namespace vsag {
// Internal typed-AST compatibility adapter. Returns nullptr for unregistered expression types.
// Registrations are compile-linked; this is not an external callback or runtime plugin API.
ExecutorPtr
CreateBuiltinExecutor(Allocator* allocator,
                      const ExprPtr& expression,
                      const AttrInvertedInterfacePtr& attr_index);
}  // namespace vsag
