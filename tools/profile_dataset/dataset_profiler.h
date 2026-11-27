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

#include <string>
#include <vector>

#include "profile_types.h"
#include "split_profiler.h"

namespace vsag::dataset_profile {

/// Report sections that may differ between runs with identical inputs: timing, peak memory
/// and the optional AI analysis. Everything else is covered by `determinism.metrics_sha256`.
const std::vector<std::string>&
non_deterministic_sections();

/// Profiles a VSAG-compatible HDF5 file and returns the versioned JSON report (without AI
/// analysis). Throws ProfileError for invalid input or options.
JsonType
profile_dataset(const ProfileOptions& options, const ProgressFn& progress);

/// SHA-256 of the canonical dump of every deterministic section.
std::string
metrics_digest(const JsonType& report);

/// Looks up a dotted path ("splits.train.norms.summary.p50"); numeric segments index arrays.
const JsonType*
find_path(const JsonType& root, const std::string& path);

/// Rule-based observations with fixed, documented thresholds. Each signal cites the report
/// keys it was derived from; this works without any AI provider.
JsonType
derive_signals(const JsonType& report);

}  // namespace vsag::dataset_profile
