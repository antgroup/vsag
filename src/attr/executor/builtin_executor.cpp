// Copyright 2024-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include "builtin_executor.h"

#include "executor_factory_registry.h"
#include "region_filter_executor.h"

namespace vsag {
namespace {
ExecutorPtr
make_region(Allocator* allocator,
            const ExprPtr& expression,
            const AttrInvertedInterfacePtr& attr_index) {
    return std::make_shared<RegionFilterExecutor>(allocator, expression, attr_index);
}
}  // namespace

const ExecutorFactoryRegistry&
ExecutorFactoryRegistry::Builtins() {
    static const ExecutorFactoryRegistry registry{{"region_filter", make_region}};
    return registry;
}

ExecutorPtr
CreateBuiltinExecutor(Allocator* allocator,
                      const ExprPtr& expression,
                      const AttrInvertedInterfacePtr& attr_index) {
    // Keep the existing grammar and typed arguments; generic FUNCTION remains unsupported.
    if (std::dynamic_pointer_cast<RegionFilterExpression>(expression)) {
        return ExecutorFactoryRegistry::Builtins().Create(
            "region_filter", allocator, expression, attr_index);
    }
    return nullptr;
}
}  // namespace vsag
