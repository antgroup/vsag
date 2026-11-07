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

#include "profile_types.h"

namespace vsag::dataset_profile {

/// Content-Security-Policy of the report: no network fetches of any kind, and the only script
/// allowed to run is the report's own inline script, pinned by its SHA-256 hash. Inline event
/// handlers are therefore blocked even if markup were injected.
std::string
html_content_security_policy();

/// Renders the JSON report as one self-contained HTML page: inline CSS, inline SVG charts and
/// a small inline script, no external resources. Every chart states its metric key,
/// definition, scope (full split or sample) and population or sample size. AI output, when
/// present, is HTML-escaped and shown separately from the deterministic metrics.
std::string
render_html_report(const JsonType& report);

}  // namespace vsag::dataset_profile
