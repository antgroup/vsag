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

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "profile_types.h"

namespace vsag::dataset_profile {

// Optional AI-assisted interpretation. The deterministic report is the only input: the
// request carries an allow-listed set of aggregate metrics (never vectors, row ids,
// projections, term ids or label values), every reply is validated against a schema and
// against the evidence keys that were sent, and any failure degrades to a warning.

constexpr const char* AI_PROMPT_VERSION = "1";
constexpr const char* DEFAULT_API_KEY_ENV = "VSAG_PROFILE_AI_API_KEY";

struct AiOptions {
    /// "none", "openai" (OpenAI-compatible Chat Completions) or "anthropic" (Messages API).
    std::string provider{"none"};
    /// Base URL ("https://api.openai.com/v1"); a URL ending in /chat/completions or
    /// /v1/messages is used as is. Defaults depend on the provider.
    std::string endpoint;
    std::string model;
    /// Name of the environment variable holding the API key. Keys are never accepted on the
    /// command line, so they cannot leak through shell history or the process list.
    std::string api_key_env{DEFAULT_API_KEY_ENV};
    /// Header carrying the key: "Authorization" sends "Bearer <key>", any other name (for
    /// example Azure's "api-key") sends the raw key. Anthropic always uses x-api-key.
    std::string auth_header{"Authorization"};
    double timeout_seconds{60.0};
    uint64_t max_retries{1};
    double retry_delay_seconds{1.0};
    uint64_t max_output_tokens{2048};
    /// Language of the generated text: "en" or "zh".
    std::string language{"en"};
    /// Send the dataset file name (off by default; paths are never sent).
    bool include_file_name{false};
    /// Build and save the request without sending it.
    bool dry_run{false};
    /// Optional path for the exact request body (credentials are headers, never in the body).
    std::string request_output;

    [[nodiscard]] bool
    Enabled() const {
        return provider != "none";
    }
};

/// Throws ProfileError for an unknown provider or language.
void
validate_ai_options(const AiOptions& options);

struct HttpRequest {
    std::string url;
    std::vector<std::pair<std::string, std::string>> headers;
    std::string body;
    double timeout_seconds{60.0};
};

struct HttpResponse {
    int status{0};
    std::string body;
    /// Non-empty when no HTTP response was received (DNS, connect, TLS, timeout).
    std::string error;
};

/// Network seam: production uses cpp-httplib, tests inject failures and canned replies.
class HttpTransport {
public:
    virtual ~HttpTransport() = default;

    virtual HttpResponse
    Post(const HttpRequest& request) = 0;
};

/// cpp-httplib transport. HTTPS requires a build with OpenSSL 3; HTTP_PROXY/HTTPS_PROXY and
/// NO_PROXY are honored.
std::unique_ptr<HttpTransport>
make_http_transport();

/// Whether this build can reach https:// endpoints.
bool
transport_supports_https();

/// Flattens the allow-listed aggregate metrics of a report into {"dotted.key": value}.
/// Numbers keep 6 significant digits and the result is bounded in size.
JsonType
build_evidence(const JsonType& report, bool include_file_name);

/// Provider-neutral prompt: {"system": ..., "user": ..., "response_schema": ...}.
JsonType
build_prompt(const JsonType& evidence, const std::string& language);

struct AiValidation {
    bool ok{false};
    /// Validated analysis containing only schema fields and known evidence keys.
    JsonType analysis;
    /// Items dropped by validation: [{"reason": ..., "item": ...}].
    JsonType rejected = JsonType::array();
    std::vector<std::string> errors;
};

/// Parses and validates a model reply (JSON, optionally inside a Markdown code fence).
AiValidation
validate_ai_response(const std::string& reply, const JsonType& evidence);

/// Runs the AI analysis for a finished report and returns the `ai_analysis` section.
/// Never throws: provider, network or validation failures produce a section with a failed
/// status and a warning on `warnings`.
JsonType
run_ai_analysis(const JsonType& report,
                const AiOptions& options,
                HttpTransport& transport,
                std::ostream& warnings);

/// Replaces every occurrence of `secret` in `text` (used for all error messages).
std::string
redact(const std::string& text, const std::string& secret);

}  // namespace vsag::dataset_profile
