// Copyright 2024-present the vsag project
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#include "executor_factory_registry.h"

namespace vsag {

ExecutorFactoryRegistry::ExecutorFactoryRegistry(std::initializer_list<Entry> entries)
    : factories_([&entries] {
          std::unordered_map<std::string, Factory> factories;
          for (const auto& entry : entries) {
              if (entry.first.empty() || entry.second == nullptr ||
                  !factories.emplace(entry.first, entry.second).second) {
                  throw VsagException(ErrorType::INVALID_ARGUMENT,
                                      "invalid or duplicate executor factory registration");
              }
          }
          return factories;
      }()) {
}

ExecutorPtr
ExecutorFactoryRegistry::Create(const std::string& name,
                                Allocator* allocator,
                                const ExprPtr& expression,
                                const AttrInvertedInterfacePtr& attr_index) const {
    auto factory = factories_.find(name);
    if (factory == factories_.end()) {
        throw VsagException(ErrorType::UNSUPPORTED_INDEX_OPERATION,
                            "unknown executor factory: " + name);
    }
    return factory->second(allocator, expression, attr_index);
}
}  // namespace vsag
