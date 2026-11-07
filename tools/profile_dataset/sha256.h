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

#include <array>
#include <cstdint>
#include <string>

namespace vsag::dataset_profile {

/// SHA-256 (FIPS 180-4) digest of a byte string. Used for the report digest, for auditing AI
/// request payloads and for the Content-Security-Policy hash of the report's inline script.
std::array<uint8_t, 32>
sha256_digest(const std::string& data);

/// Lowercase hex of sha256_digest().
std::string
sha256_hex(const std::string& data);

/// Standard base64 (RFC 4648, padded) of sha256_digest(), the form CSP script hashes use.
std::string
sha256_base64(const std::string& data);

}  // namespace vsag::dataset_profile
