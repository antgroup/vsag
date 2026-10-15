// Copyright 2024-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#pragma once

#include <initializer_list>
#include <string>
#include <unordered_map>

#include "executor.h"

namespace vsag {
// Internal immutable named factories. Each invocation creates an independently owned executor.
// The allocator must outlive its executors; each executor retains its expression and index.
// There is no runtime registration or public registry injection into search.
class ExecutorFactoryRegistry {
public:
    using Factory = ExecutorPtr (*)(Allocator*, const ExprPtr&, const AttrInvertedInterfacePtr&);
    using Entry = std::pair<std::string, Factory>;

    explicit ExecutorFactoryRegistry(std::initializer_list<Entry> entries);

    ExecutorPtr
    Create(const std::string& name,
           Allocator* allocator,
           const ExprPtr& expression,
           const AttrInvertedInterfacePtr& attr_index) const;

    static const ExecutorFactoryRegistry&
    Builtins();

private:
    const std::unordered_map<std::string, Factory> factories_;
};
}  // namespace vsag
