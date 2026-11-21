// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include "vsag/vsag.h"

namespace {
// The aggregate test binary links vsag.cpp through other test modules. Retain
// that translation unit's normal library initialization and ISA diagnostics in
// the scoped coverage executable too, without calling init() a second time.
[[maybe_unused]] const auto linked_version = vsag::version();
}  // namespace
