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

// AI layer: privacy of the request, validation of replies, provider failures, and an
// end-to-end run against a local mock provider.

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <httplib.h>
#pragma GCC diagnostic pop

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

#include "ai_analysis.h"
#include "dataset_profiler.h"
#include "html_report.h"
#include "profile_cli.h"
#include "profile_dataset_test_utils.h"

using namespace vsag::dataset_profile;
using namespace vsag::dataset_profile::testing;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr const char* TEST_KEY = "sk-test-0123456789abcdef";
constexpr const char* TEST_KEY_ENV = "VSAG_PROFILE_TEST_API_KEY";

/// Sets an environment variable for the lifetime of the object.
class ScopedEnv {
public:
    ScopedEnv(const char* name, const char* value) : name_(name) {
        setenv(name, value, 1);
    }
    ~ScopedEnv() {
        unsetenv(name_);
    }
    ScopedEnv(const ScopedEnv&) = delete;
    ScopedEnv&
    operator=(const ScopedEnv&) = delete;

private:
    const char* name_;
};

class MockTransport : public HttpTransport {
public:
    std::vector<HttpRequest> requests;
    std::vector<HttpResponse> responses;

    HttpResponse
    Post(const HttpRequest& request) override {
        requests.push_back(request);
        if (responses.empty()) {
            return HttpResponse{0, "", "no scripted response"};
        }
        HttpResponse response = responses.front();
        if (responses.size() > 1) {
            responses.erase(responses.begin());
        }
        return response;
    }
};

const JsonType&
dense_report() {
    static TempDir dir;
    static const JsonType report = [] {
        const auto path = dir.File("secret_customer_vectors.hdf5");
        make_dense_fixture(path);
        ProfileOptions options;
        options.data_path = path;
        options.sample_size = 800;
        options.knn_queries = 100;
        options.pair_count = 3000;
        options.quantization_base = 500;
        options.quantization_queries = 40;
        return profile_dataset(options, nullptr);
    }();
    return report;
}

std::string
valid_reply(const std::string& evidence_key,
            const std::string& summary = "The data has invalid rows.") {
    return JsonType{
        {"summary", summary},
        {"findings",
         JsonType::array({JsonType{{"title", "Invalid values"},
                                   {"severity", "critical"},
                                   {"detail", "Three rows contain NaN or infinity."},
                                   {"evidence", JsonType::array({evidence_key})}}})},
        {"recommendations",
         JsonType::array(
             {JsonType{{"title", "Drop invalid rows"},
                       {"category", "data_quality"},
                       {"hypothesis", "Removing the rows avoids undefined distances."},
                       {"rationale", "The profile counts invalid values."},
                       {"validation", "Rebuild and compare recall with eval_performance."},
                       {"confidence", "high"},
                       {"evidence", JsonType::array({evidence_key})}}})},
        {"uncertainties", JsonType::array({"Sampled metrics have sampling error."})},
        {"additional_measurements", JsonType::array({"Benchmark recall at the target ef_search."})}}
        .dump();
}

std::string
openai_envelope(const std::string& content) {
    return JsonType{{"choices",
                     JsonType::array({JsonType{
                         {"message", JsonType{{"role", "assistant"}, {"content", content}}}}})}}
        .dump();
}

AiOptions
openai_options() {
    AiOptions options;
    options.provider = "openai";
    options.model = "mock-model";
    options.endpoint = "https://llm.example.com/v1";
    options.api_key_env = TEST_KEY_ENV;
    options.retry_delay_seconds = 0.0;
    return options;
}

void
check_private(const std::string& body) {
    CHECK(body.find(TEST_KEY) == std::string::npos);
    CHECK(body.find("secret_customer_vectors") == std::string::npos);
    for (const char* forbidden :
         {"per_dimension", "\"points\"", "examples", "top_terms", "histogram"}) {
        INFO(forbidden);
        CHECK(body.find(forbidden) == std::string::npos);
    }
    for (uint64_t d = 0; d < 16; ++d) {
        char text[32];
        std::snprintf(text, sizeof(text), "%.6g", static_cast<double>(canary_value(d)));
        INFO(text);
        CHECK(body.find(text) == std::string::npos);
    }
}

std::string
read_text(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

int
cli(const std::vector<std::string>& arguments, std::string& out, std::string& err) {
    std::vector<std::string> storage = {"profile_dataset"};
    storage.insert(storage.end(), arguments.begin(), arguments.end());
    std::vector<char*> argv;
    argv.reserve(storage.size());
    for (auto& argument : storage) {
        argv.push_back(argument.data());
    }
    std::ostringstream out_stream;
    std::ostringstream err_stream;
    const int code = run_cli(static_cast<int>(argv.size()), argv.data(), out_stream, err_stream);
    out = out_stream.str();
    err = err_stream.str();
    return code;
}

}  // namespace

TEST_CASE("Evidence contains only allow-listed aggregate metrics", "[profile][ai]") {
    const JsonType& report = dense_report();
    const JsonType evidence = build_evidence(report, false);
    REQUIRE(evidence.is_object());
    CHECK(evidence.contains("splits.train.norms.summary.p50"));
    CHECK(evidence.contains("splits.train.invalid_values.total"));
    CHECK(evidence.contains("config.metric"));
    CHECK(evidence.contains("signals.0.id"));
    CHECK_FALSE(evidence.contains("input.file_name"));
    for (const auto& item : evidence.items()) {
        INFO(item.key());
        CHECK_FALSE(item.value().is_object());
        CHECK_FALSE(item.value().is_array());
        for (const char* denied : {"per_dimension",
                                   "histogram",
                                   "points",
                                   "examples",
                                   "top_terms",
                                   "rule",
                                   "evidence",
                                   "run.",
                                   "determinism",
                                   "min_label",
                                   "max_label"}) {
            CHECK(item.key().find(denied) == std::string::npos);
        }
    }
    check_private(evidence.dump());
    CHECK(evidence.dump().size() < 48ULL * 1024ULL);
    CHECK(build_evidence(report, true).contains("input.file_name"));
    // Evidence size does not grow with the row count: only aggregates are included.
    CHECK(evidence.size() < 1500);
}

TEST_CASE("Prompt carries rules, schema and language", "[profile][ai]") {
    const JsonType evidence = build_evidence(dense_report(), false);
    const JsonType prompt = build_prompt(evidence, "en");
    CHECK_THAT(prompt["system"].get<std::string>(), ContainsSubstring("hypotheses"));
    CHECK_THAT(prompt["system"].get<std::string>(), ContainsSubstring("RESPONSE_SCHEMA"));
    CHECK_THAT(prompt["user"].get<std::string>(),
               ContainsSubstring("splits.train.norms.summary.p50"));
    CHECK_THAT(build_prompt(evidence, "zh")["system"].get<std::string>(),
               ContainsSubstring("Simplified Chinese"));
}

TEST_CASE("Replies are validated against the schema and the evidence keys", "[profile][ai]") {
    const JsonType evidence = build_evidence(dense_report(), false);
    const std::string key = "splits.train.invalid_values.total";
    AiValidation valid = validate_ai_response(valid_reply(key), evidence);
    REQUIRE(valid.ok);
    CHECK(valid.analysis["findings"].size() == 1);
    CHECK(valid.analysis["recommendations"].size() == 1);
    CHECK(valid.rejected.empty());
    CHECK(validate_ai_response("```json\n" + valid_reply(key) + "\n```", evidence).ok);
    CHECK(validate_ai_response("Here is the analysis:\n" + valid_reply(key) + "\nThanks.", evidence)
              .ok);

    CHECK_FALSE(validate_ai_response("{not json", evidence).ok);
    CHECK_FALSE(validate_ai_response("[1, 2, 3]", evidence).ok);
    CHECK_FALSE(validate_ai_response("", evidence).ok);
    CHECK_FALSE(
        validate_ai_response(std::string(300ULL * 1024ULL, ' ') + valid_reply(key), evidence).ok);
    JsonType reply = JsonType::parse(valid_reply(key));
    reply.erase("summary");
    CHECK_FALSE(validate_ai_response(reply.dump(), evidence).ok);
    reply = JsonType::parse(valid_reply(key));
    reply.erase("uncertainties");
    CHECK_FALSE(validate_ai_response(reply.dump(), evidence).ok);

    reply = JsonType::parse(valid_reply(key));
    reply["findings"][0]["evidence"] = JsonType::array({"splits.train.made_up_metric"});
    reply["recommendations"][0]["evidence"] = JsonType::array({key, "splits.train.made_up_metric"});
    AiValidation partial = validate_ai_response(reply.dump(), evidence);
    REQUIRE(partial.ok);
    CHECK(partial.analysis["findings"].empty());
    CHECK(partial.analysis["recommendations"][0]["evidence"] == JsonType::array({key}));
    REQUIRE(partial.rejected.size() == 1);
    CHECK_THAT(partial.rejected[0]["reason"].get<std::string>(), ContainsSubstring("unknown"));

    reply = JsonType::parse(valid_reply(key));
    reply["recommendations"][0]["hypothesis"] = "HGraph with sq8 is guaranteed to be optimal.";
    reply["findings"][0].erase("evidence");
    AiValidation rejected = validate_ai_response(reply.dump(), evidence);
    CHECK_FALSE(rejected.ok);
    REQUIRE(rejected.rejected.size() == 2);
    CHECK_THAT(rejected.rejected.dump(), ContainsSubstring("missing evidence"));
    CHECK_THAT(rejected.rejected.dump(), ContainsSubstring("guarantee"));

    reply = JsonType::parse(valid_reply(key));
    reply["recommendations"][0].erase("validation");
    reply["findings"][0]["severity"] = "catastrophic";
    reply["recommendations"].push_back(reply["recommendations"][0]);
    reply["recommendations"][1]["validation"] = "Compare with eval_performance.";
    reply["recommendations"][1]["category"] = "magic";
    AiValidation malformed = validate_ai_response(reply.dump(), evidence);
    CHECK_FALSE(malformed.ok);
    CHECK(malformed.rejected.size() == 3);
}

TEST_CASE("Requests carry no vectors or credentials and replies are validated", "[profile][ai]") {
    ScopedEnv key(TEST_KEY_ENV, TEST_KEY);
    MockTransport transport;
    transport.responses = {
        HttpResponse{200, openai_envelope(valid_reply("splits.train.invalid_values.total")), ""}};
    std::ostringstream warnings;
    const JsonType section = run_ai_analysis(dense_report(), openai_options(), transport, warnings);
    REQUIRE(section["status"] == "ok");
    CHECK(warnings.str().empty());
    REQUIRE(transport.requests.size() == 1);
    const HttpRequest& request = transport.requests[0];
    CHECK(request.url == "https://llm.example.com/v1/chat/completions");
    bool authorized = false;
    for (const auto& [name, value] : request.headers) {
        authorized =
            authorized || (name == "Authorization" && value == std::string("Bearer ") + TEST_KEY);
    }
    CHECK(authorized);
    check_private(request.body);
    const JsonType body = JsonType::parse(request.body);
    CHECK(body["model"] == "mock-model");
    CHECK(body["messages"].size() == 2);
    CHECK(section["request_sha256"].get<std::string>().size() == 64);
    CHECK(section["raw_vectors_sent"] == false);
    CHECK(section.dump().find(TEST_KEY) == std::string::npos);

    // Untrusted text from the model is escaped in the HTML report.
    MockTransport hostile;
    hostile.responses = {
        HttpResponse{200,
                     openai_envelope(valid_reply("splits.train.invalid_values.total",
                                                 "<img src=x onerror=alert(1)>")),
                     ""}};
    JsonType report = dense_report();
    report["ai_analysis"] = run_ai_analysis(report, openai_options(), hostile, warnings);
    const std::string html = render_html_report(report);
    CHECK(html.find("<img src=x") == std::string::npos);
    CHECK_THAT(html, ContainsSubstring("&lt;img src=x onerror=alert(1)&gt;"));
    CHECK_THAT(html, ContainsSubstring("Recommendations to validate"));
}

TEST_CASE("Anthropic provider uses the Messages API", "[profile][ai]") {
    ScopedEnv key(TEST_KEY_ENV, TEST_KEY);
    MockTransport transport;
    const JsonType envelope = JsonType{
        {"content",
         JsonType::array({JsonType{{"type", "text"},
                                   {"text", valid_reply("splits.train.norms.summary.p50")}}})}};
    transport.responses = {HttpResponse{200, envelope.dump(), ""}};
    AiOptions options = openai_options();
    options.provider = "anthropic";
    options.endpoint = "";
    std::ostringstream warnings;
    const JsonType section = run_ai_analysis(dense_report(), options, transport, warnings);
    CHECK(section["status"] == "ok");
    REQUIRE(transport.requests.size() == 1);
    CHECK(transport.requests[0].url == "https://api.anthropic.com/v1/messages");
    bool has_key = false;
    bool has_version = false;
    for (const auto& [name, value] : transport.requests[0].headers) {
        has_key = has_key || (name == "x-api-key" && value == TEST_KEY);
        has_version = has_version || name == "anthropic-version";
    }
    CHECK(has_key);
    CHECK(has_version);
    check_private(transport.requests[0].body);
}

TEST_CASE("Provider failures degrade to warnings and never leak the key", "[profile][ai]") {
    ScopedEnv key(TEST_KEY_ENV, TEST_KEY);
    std::ostringstream warnings;
    MockTransport unreachable;
    unreachable.responses = {
        HttpResponse{0, "", std::string("connection refused while sending ") + TEST_KEY}};
    AiOptions options = openai_options();
    options.max_retries = 2;
    JsonType section = run_ai_analysis(dense_report(), options, unreachable, warnings);
    CHECK(section["status"] == "failed");
    CHECK(unreachable.requests.size() == 3);
    CHECK_THAT(warnings.str(), ContainsSubstring("AI analysis failed"));
    CHECK_THAT(warnings.str(), ContainsSubstring("deterministic JSON and HTML reports"));
    CHECK(warnings.str().find(TEST_KEY) == std::string::npos);
    CHECK(section.dump().find(TEST_KEY) == std::string::npos);
    CHECK_THAT(section["message"].get<std::string>(), ContainsSubstring("[REDACTED]"));

    MockTransport flaky;
    flaky.responses = {
        HttpResponse{503, "busy", ""},
        HttpResponse{429, "slow down", ""},
        HttpResponse{200, openai_envelope(valid_reply("splits.train.invalid_values.total")), ""}};
    section = run_ai_analysis(dense_report(), options, flaky, warnings);
    CHECK(section["status"] == "ok");
    CHECK(section["attempts"] == 3);

    MockTransport unauthorized;
    unauthorized.responses = {
        HttpResponse{401, std::string(R"({"error": "bad key )") + TEST_KEY + "\"}", ""}};
    section = run_ai_analysis(dense_report(), options, unauthorized, warnings);
    CHECK(section["status"] == "failed");
    CHECK(unauthorized.requests.size() == 1);
    CHECK_THAT(section["message"].get<std::string>(), ContainsSubstring("HTTP 401"));
    CHECK(section.dump().find(TEST_KEY) == std::string::npos);

    // A key echoed across the 300-byte truncation boundary must not survive in part either.
    const std::string key_prefix = std::string(TEST_KEY).substr(0, 12);
    MockTransport cut_body;
    cut_body.responses = {HttpResponse{401, std::string(280, 'x') + TEST_KEY, ""}};
    section = run_ai_analysis(dense_report(), options, cut_body, warnings);
    CHECK(section["status"] == "failed");
    CHECK(section.dump().find(key_prefix) == std::string::npos);
    CHECK_THAT(section["message"].get<std::string>(), ContainsSubstring("[REDACTED]"));
    MockTransport cut_error;
    cut_error.responses = {HttpResponse{0, "", std::string(280, 'y') + TEST_KEY}};
    section = run_ai_analysis(dense_report(), options, cut_error, warnings);
    CHECK(section["status"] == "failed");
    CHECK(section.dump().find(key_prefix) == std::string::npos);
    CHECK(warnings.str().find(key_prefix) == std::string::npos);

    MockTransport garbage;
    garbage.responses = {HttpResponse{200, "<html>gateway</html>", ""}};
    CHECK(run_ai_analysis(dense_report(), options, garbage, warnings)["status"] ==
          "invalid_response");

    MockTransport unsupported;
    unsupported.responses = {
        HttpResponse{200,
                     openai_envelope(R"({"summary": "x", "findings": [], "recommendations": [],
        "uncertainties": [], "additional_measurements": []})"),
                     ""}};
    CHECK(run_ai_analysis(dense_report(), options, unsupported, warnings)["status"] ==
          "invalid_response");

    MockTransport unused;
    options.model = "";
    CHECK_THAT(
        run_ai_analysis(dense_report(), options, unused, warnings)["message"].get<std::string>(),
        ContainsSubstring("--ai-model"));
    options.model = "mock-model";
    options.endpoint = "https://user:password@llm.example.com/v1";
    CHECK_THAT(
        run_ai_analysis(dense_report(), options, unused, warnings)["message"].get<std::string>(),
        ContainsSubstring("credentials"));
    CHECK(unused.requests.empty());
    CHECK(redact("a sk-1 b sk-1", "sk-1") == "a [REDACTED] b [REDACTED]");
    CHECK(redact("unchanged", "") == "unchanged");
}

TEST_CASE("Dry runs build and save the request without sending it", "[profile][ai]") {
    ScopedEnv key(TEST_KEY_ENV, TEST_KEY);
    TempDir dir;
    MockTransport transport;
    AiOptions options = openai_options();
    options.dry_run = true;
    options.model = "";
    options.request_output = dir.File("request.json");
    std::ostringstream warnings;
    const JsonType section = run_ai_analysis(dense_report(), options, transport, warnings);
    CHECK(section["status"] == "dry_run");
    CHECK(transport.requests.empty());
    const std::string saved = read_text(options.request_output);
    CHECK_THAT(saved, ContainsSubstring("messages"));
    check_private(saved);
}

TEST_CASE("End to end: a local OpenAI-compatible provider, then an unavailable one",
          "[profile][ai][cli]") {
    ScopedEnv key(TEST_KEY_ENV, TEST_KEY);
    TempDir dir;
    const auto data = dir.File("secret_customer_vectors.hdf5");
    make_dense_fixture(data);
    std::mutex mutex;
    std::string captured_body;
    std::string captured_authorization;
    httplib::Server server;
    server.Post(
        "/v1/chat/completions", [&](const httplib::Request& request, httplib::Response& response) {
            {
                std::lock_guard<std::mutex> lock(mutex);
                captured_body = request.body;
                captured_authorization = request.get_header_value("Authorization");
            }
            response.set_content(openai_envelope(valid_reply("splits.train.invalid_values.total")),
                                 "application/json");
        });
    const int port = server.bind_to_any_port("127.0.0.1");
    REQUIRE(port > 0);
    std::thread worker([&server] { server.listen_after_bind(); });
    server.wait_until_ready();

    const auto json = dir.File("report.json");
    const auto html = dir.File("report.html");
    const std::vector<std::string> common = {"-d",
                                             data,
                                             "-o",
                                             json,
                                             "--html",
                                             html,
                                             "--sample-size",
                                             "500",
                                             "--knn-queries",
                                             "60",
                                             "--ai-provider",
                                             "openai",
                                             "--ai-model",
                                             "mock-model",
                                             "--ai-api-key-env",
                                             TEST_KEY_ENV,
                                             "--ai-timeout",
                                             "5",
                                             "--ai-endpoint",
                                             "http://127.0.0.1:" + std::to_string(port) + "/v1"};
    std::string out;
    std::string err;
    REQUIRE(cli(common, out, err) == EXIT_OK);
    server.stop();
    worker.join();
    {
        std::lock_guard<std::mutex> lock(mutex);
        CHECK(captured_authorization == std::string("Bearer ") + TEST_KEY);
        check_private(captured_body);
    }
    JsonType report = JsonType::parse(read_text(json));
    CHECK(report["ai_analysis"]["status"] == "ok");
    CHECK(report["ai_analysis"]["analysis"]["recommendations"].size() == 1);
    CHECK_THAT(out, ContainsSubstring("AI analysis: ok"));
    for (const std::string& text : {read_text(json), read_text(html), out, err}) {
        CHECK(text.find(TEST_KEY) == std::string::npos);
    }
    // The digest covers the deterministic sections only, so it ignores the AI section.
    CHECK(metrics_digest(report) == report["determinism"]["metrics_sha256"].get<std::string>());

    std::remove(json.c_str());
    std::remove(html.c_str());
    REQUIRE(cli(common, out, err) == EXIT_OK);
    CHECK_THAT(err, ContainsSubstring("warning: AI analysis failed"));
    report = JsonType::parse(read_text(json));
    CHECK(report["ai_analysis"]["status"] == "failed");
    CHECK(report.contains("splits"));
    CHECK_THAT(read_text(html), ContainsSubstring("Status: failed"));
    std::vector<std::string> required = common;
    required.emplace_back("--ai-required");
    CHECK(cli(required, out, err) == EXIT_AI_REQUIRED_FAILED);
    CHECK(JsonType::parse(read_text(json)).contains("splits"));
}
