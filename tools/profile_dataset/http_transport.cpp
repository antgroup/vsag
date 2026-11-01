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

// cpp-httplib's TLS code calls its own deprecated helpers; keep -Werror for our code only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <httplib.h>
#pragma GCC diagnostic pop

#include <cstdlib>
#include <sstream>

#include "ai_analysis.h"

namespace vsag::dataset_profile {

namespace {

struct Endpoint {
    std::string scheme;
    std::string host;
    int port{0};
    std::string target;
};

bool
parse_endpoint(const std::string& url, Endpoint& endpoint, std::string& error) {
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        error = "malformed URL";
        return false;
    }
    endpoint.scheme = url.substr(0, scheme_end);
    const std::string rest = url.substr(scheme_end + 3);
    const auto slash = rest.find('/');
    std::string authority = rest.substr(0, slash);
    endpoint.target = slash == std::string::npos ? "/" : rest.substr(slash);
    endpoint.port = endpoint.scheme == "https" ? 443 : 80;
    const auto colon = authority.rfind(':');
    if (colon != std::string::npos && authority.find(']') == std::string::npos) {
        try {
            endpoint.port = std::stoi(authority.substr(colon + 1));
        } catch (const std::exception&) {
            error = "malformed port";
            return false;
        }
        authority = authority.substr(0, colon);
    }
    endpoint.host = authority;
    return !endpoint.host.empty();
}

const char*
environment(const char* upper, const char* lower) {
    const char* value = std::getenv(upper);
    if (value == nullptr || *value == '\0') {
        value = std::getenv(lower);
    }
    return value != nullptr && *value != '\0' ? value : nullptr;
}

bool
bypass_proxy(const std::string& host) {
    // Local providers (Ollama, vLLM, test servers) are never reached through a proxy.
    if (host == "localhost" || host.starts_with("127.") || host == "[::1]") {
        return true;
    }
    const char* no_proxy = environment("NO_PROXY", "no_proxy");
    if (no_proxy == nullptr) {
        return false;
    }
    std::stringstream entries(no_proxy);
    std::string entry;
    while (std::getline(entries, entry, ',')) {
        while (!entry.empty() && entry.front() == ' ') {
            entry.erase(entry.begin());
        }
        while (!entry.empty() && entry.back() == ' ') {
            entry.pop_back();
        }
        if (!entry.empty() && entry.front() == '.') {
            entry.erase(entry.begin());
        }
        if (entry == "*" || host == entry ||
            (host.size() > entry.size() && host.ends_with(entry) &&
             host[host.size() - entry.size() - 1] == '.')) {
            return true;
        }
    }
    return false;
}

void
configure_proxy(httplib::Client& client, const Endpoint& endpoint) {
    if (bypass_proxy(endpoint.host)) {
        return;
    }
    const char* proxy = endpoint.scheme == "https" ? environment("HTTPS_PROXY", "https_proxy")
                                                   : environment("HTTP_PROXY", "http_proxy");
    if (proxy == nullptr) {
        return;
    }
    std::string url = proxy;
    if (url.find("://") == std::string::npos) {
        url = "http://" + url;
    }
    std::string credentials;
    const auto scheme_end = url.find("://");
    const auto at = url.find('@', scheme_end + 3);
    if (at != std::string::npos) {
        credentials = url.substr(scheme_end + 3, at - scheme_end - 3);
        url = url.substr(0, scheme_end + 3) + url.substr(at + 1);
    }
    Endpoint proxy_endpoint;
    std::string error;
    if (!parse_endpoint(url, proxy_endpoint, error)) {
        return;
    }
    client.set_proxy(proxy_endpoint.host, proxy_endpoint.port);
    if (!credentials.empty()) {
        const auto colon = credentials.find(':');
        client.set_proxy_basic_auth(
            credentials.substr(0, colon),
            colon == std::string::npos ? "" : credentials.substr(colon + 1));
    }
}

class HttplibTransport : public HttpTransport {
public:
    HttpResponse
    Post(const HttpRequest& request) override {
        HttpResponse response;
        Endpoint endpoint;
        if (!parse_endpoint(request.url, endpoint, response.error)) {
            return response;
        }
        if (endpoint.scheme == "https" && !transport_supports_https()) {
            response.error =
                "this profile_dataset build has no TLS support (OpenSSL 3 was not found at build "
                "time); use "
                "an http:// endpoint such as a local gateway, or rebuild with OpenSSL";
            return response;
        }
        try {
            httplib::Client client(endpoint.scheme + "://" + endpoint.host + ":" +
                                   std::to_string(endpoint.port));
            const auto seconds = static_cast<time_t>(request.timeout_seconds);
            const auto microseconds =
                static_cast<time_t>((request.timeout_seconds - static_cast<double>(seconds)) * 1e6);
            client.set_connection_timeout(seconds, microseconds);
            client.set_read_timeout(seconds, microseconds);
            client.set_write_timeout(seconds, microseconds);
            configure_proxy(client, endpoint);
            httplib::Headers headers;
            std::string content_type = "application/json";
            for (const auto& [name, value] : request.headers) {
                if (name == "Content-Type") {
                    content_type = value;
                } else {
                    headers.emplace(name, value);
                }
            }
            auto result = client.Post(endpoint.target, headers, request.body, content_type);
            if (!result) {
                response.error = httplib::to_string(result.error());
                return response;
            }
            response.status = result->status;
            response.body = result->body;
        } catch (const std::exception& error) {
            response.error = error.what();
        }
        return response;
    }
};

}  // namespace

bool
transport_supports_https() {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
    return true;
#else
    return false;
#endif
}

std::unique_ptr<HttpTransport>
make_http_transport() {
    return std::make_unique<HttplibTransport>();
}

}  // namespace vsag::dataset_profile
