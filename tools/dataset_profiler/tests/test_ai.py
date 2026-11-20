# Copyright 2024-present the vsag project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the optional AI interpretation adapter."""

from __future__ import annotations

import json

import pytest

from dataset_profiler import ai
from dataset_profiler.ai import (
    AIAnalysisError,
    MockProvider,
    OpenAICompatibleProvider,
    aggregate_payload,
    analyze_report,
    build_prompt,
    parse_and_validate_response,
    provider_from_env,
)
from dataset_profiler.profiler import profile_dataset
from dataset_profiler.stats import dump_json


def _report(dense_path):
    return profile_dataset(str(dense_path), sample_size=24, pairwise_sample_size=12)


def _canned_response():
    return json.dumps(
        {
            "summary": "The sample is anisotropic with a small duplicate count.",
            "findings": [
                {
                    "title": "Dominant direction",
                    "detail": "PC1 explains most variance.",
                    "evidence": ["metrics.anisotropy.top1_eigenvalue_share"],
                    "confidence": "medium",
                }
            ],
            "recommendations": [
                {
                    "title": "Consider HGraph",
                    "detail": "Hypothesis: a graph index may help recall.",
                    "evidence": ["metrics.norms.stats.p50", "metrics.pairwise_distances.stats.p50"],
                    "validation": "Compare recall and QPS in tools/eval before adopting.",
                    "confidence": "low",
                }
            ],
            "limitations": ["Sampled metrics only."],
            "additional_measurements": ["Full pairwise distance run."],
        }
    )


# ---------------------------------------------------------------------------
# Prompt / payload privacy boundary
# ---------------------------------------------------------------------------
def test_prompt_contains_only_aggregate_metrics(dense_path, dense_matrix):
    report = _report(dense_path)
    prompt = build_prompt(report)
    payload = aggregate_payload(report)
    assert set(payload) == {
        "schema_version",
        "tool",
        "split",
        "dataset",
        "config",
        "metrics",
        "warnings",
    }
    assert "Allowed evidence metric keys:" in prompt
    # no raw vector row is ever part of the prompt
    for row in dense_matrix[:3]:
        assert repr(row.tolist()) not in prompt
    assert "raw_vectors" not in prompt


def test_prompt_is_jsonable(dense_path):
    report = _report(dense_path)
    payload = aggregate_payload(report)
    json.dumps(payload, allow_nan=False, sort_keys=True)


def test_aggregate_payload_omits_arbitrary_dataset_object_names(dense_path):
    report = _report(dense_path)
    report["dataset"]["datasets"].append("customer-private-label")
    report["dataset"]["attributes"]["private_label"] = "customer-secret"

    payload = aggregate_payload(report)
    dataset = payload["dataset"]
    assert set(dataset) == {
        "vector_type",
        "dtype",
        "count",
        "dim",
        "dim_inferred",
        "distance_metric",
        "has_offsets",
    }
    assert "datasets" not in dataset
    assert "attributes" not in dataset
    prompt = build_prompt(report)
    assert "customer-private-label" not in prompt
    assert "customer-secret" not in prompt


def test_aggregate_payload_excludes_per_vector_projections(dense_path):
    report = profile_dataset(
        str(dense_path), sample_size=24, pairwise_sample_size=12, clusters=3
    )
    assert report["metrics"]["clustering"]["projection_points"]
    assert report["metrics"]["clustering"]["labels"]
    payload = aggregate_payload(report)
    clustering = payload["metrics"]["clustering"]
    assert "projection_points" not in clustering
    assert "labels" not in clustering
    # aggregate summary fields are still forwarded
    assert "cluster_sizes" in clustering
    assert clustering["n_clusters"] == 3
    prompt = build_prompt(report)
    assert "projection_points" not in prompt



def test_aggregate_payload_forwards_budget_unavailability():
    report = {
        "metrics": {
            "norms": {"available": False, "reason": "norm budget exceeded"},
            "outliers": {"available": False, "reason": "norm budget exceeded"},
            "sparsity": {
                "nnz_available": False,
                "nnz_reason": "nnz budget exceeded",
            },
        }
    }
    metrics = aggregate_payload(report)["metrics"]
    for name in ("norms", "outliers"):
        assert metrics[name]["available"] is False
        assert metrics[name]["reason"] == "norm budget exceeded"
    assert metrics["sparsity"]["nnz_available"] is False
    assert metrics["sparsity"]["nnz_reason"] == "nnz budget exceeded"


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------
def test_valid_response_is_parsed(dense_path):
    report = _report(dense_path)
    block = parse_and_validate_response(_canned_response(), report)
    assert block["summary"].startswith("The sample is anisotropic")
    assert block["findings"][0]["evidence"] == ["metrics.anisotropy.top1_eigenvalue_share"]
    assert block["recommendations"][0]["validation"].startswith("Compare recall")
    assert block["recommendations"][0]["confidence"] == "low"


def test_response_inside_markdown_fences_is_parsed(dense_path):
    report = _report(dense_path)
    fenced = "```json\n" + _canned_response() + "\n```"
    block = parse_and_validate_response(fenced, report)
    assert block["findings"]


def test_malformed_responses_are_rejected(dense_path):
    report = _report(dense_path)
    bad_cases = [
        "",
        "not json at all",
        json.dumps([1, 2, 3]),
        json.dumps({"findings": [{"title": "x", "detail": "y", "evidence": ["metrics.norms.stats.mean"]}],
                   "recommendations": []}),  # missing summary / empty recommendations
        json.dumps({"summary": "s", "findings": [], "recommendations": []}),
        json.dumps({"summary": "s", "findings": "oops", "recommendations": []}),
    ]
    for text in bad_cases:
        with pytest.raises(AIAnalysisError):
            parse_and_validate_response(text, report)


def test_evidence_must_reference_real_metric_keys(dense_path):
    report = _report(dense_path)
    response = json.loads(_canned_response())
    response["findings"][0]["evidence"] = ["metrics.does_not_exist"]
    with pytest.raises(AIAnalysisError, match="unknown evidence key"):
        parse_and_validate_response(json.dumps(response), report)

    response = json.loads(_canned_response())
    response["findings"][0]["evidence"] = []
    with pytest.raises(AIAnalysisError, match="non-empty evidence"):
        parse_and_validate_response(json.dumps(response), report)

    response = json.loads(_canned_response())
    response["recommendations"][0].pop("validation")
    with pytest.raises(AIAnalysisError, match="validation"):
        parse_and_validate_response(json.dumps(response), report)

    response = json.loads(_canned_response())
    response["findings"][0]["confidence"] = "certain"
    with pytest.raises(AIAnalysisError, match="confidence"):
        parse_and_validate_response(json.dumps(response), report)


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
def test_analyze_report_ok_path(dense_path):
    report = _report(dense_path)
    provider = MockProvider(_canned_response())
    block = analyze_report(report, provider=provider)
    assert block["status"] == "ok"
    assert block["provider"] == "mock"
    assert block["model"] == "mock-model"
    assert provider.last_prompt is not None
    # AI block never breaks JSON serialization of the full report
    report["ai_analysis"] = block
    json.loads(dump_json(report))


def test_analyze_report_degrades_on_provider_failure(dense_path):
    class FailingProvider:
        name = "failing"
        model = "m"

        def complete(self, prompt):
            raise ai.AIAnalysisError("AI provider is unreachable")

    report = _report(dense_path)
    block = analyze_report(report, provider=FailingProvider())
    assert block["status"] == "error"
    assert "unreachable" in block["error"]


def test_analyze_report_redacts_secrets_in_errors(dense_path):
    class LeakyProvider:
        name = "leaky"
        model = "m"

        def complete(self, prompt):
            raise RuntimeError("auth failed with sk-abcdefghijkl12345678")

    report = _report(dense_path)
    block = analyze_report(report, provider=LeakyProvider())
    assert block["status"] == "error"
    assert "sk-abcdefghijkl12345678" not in block["error"]
    assert "[redacted]" in block["error"]
    dump_json({"ai_analysis": block})


def test_analyze_report_rejects_malformed_provider_json(dense_path):
    provider = MockProvider("this is not json")
    block = analyze_report(_report(dense_path), provider=provider)
    assert block["status"] == "error"


def test_provider_requires_https_and_credentials():
    with pytest.raises(AIAnalysisError):
        OpenAICompatibleProvider(api_key="", model="m")
    with pytest.raises(AIAnalysisError):
        OpenAICompatibleProvider(api_key="k", model="")
    with pytest.raises(AIAnalysisError):
        OpenAICompatibleProvider(api_key="k", model="m", base_url="http://insecure.example/v1")


def test_provider_from_env_resolution():
    with pytest.raises(AIAnalysisError):
        provider_from_env(env={})
    provider = provider_from_env(
        env={"VSAG_AI_API_KEY": "k", "VSAG_AI_MODEL": "m", "VSAG_AI_BASE_URL": "https://example.com/v1"}
    )
    assert provider.model == "m"
    assert provider.base_url == "https://example.com/v1"
    fallback = provider_from_env(
        env={"OPENAI_API_KEY": "k2", "VSAG_AI_MODEL": "m2"},
        base_url="https://other.example/v1",
    )
    assert fallback.base_url == "https://other.example/v1"


def test_explicit_empty_vsag_key_does_not_use_openai_fallback():
    with pytest.raises(AIAnalysisError, match="no API key"):
        provider_from_env(
            env={
                "VSAG_AI_API_KEY": "",
                "OPENAI_API_KEY": "must-not-be-used",
                "VSAG_AI_MODEL": "m",
            }
        )


def test_provider_rejects_redirects_before_forwarding_credentials():
    provider = OpenAICompatibleProvider(
        api_key="sk-supersecretvalue123", model="m", base_url="https://api.example.com/v1"
    )
    handler = next(
        item for item in provider._opener.handlers if isinstance(item, ai._RejectRedirectHandler)
    )
    request = ai.urllib.request.Request(
        "https://api.example.com/v1/chat/completions",
        headers={"Authorization": "Bearer sk-supersecretvalue123"},
    )
    with pytest.raises(AIAnalysisError, match="redirects are disabled"):
        handler.redirect_request(
            request, None, 302, "Found", {}, "https://attacker.example/collect"
        )


def test_openai_request_body_excludes_credentials_and_vectors(monkeypatch, dense_path, dense_matrix):
    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": _canned_response()}}]}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_open(request, timeout=None):
        captured["headers"] = dict(request.headers)
        captured["body"] = request.data.decode("utf-8")
        captured["url"] = request.full_url
        return FakeResponse()

    provider = OpenAICompatibleProvider(
        api_key="sk-supersecretvalue123", model="m", base_url="https://api.example.com/v1"
    )
    monkeypatch.setattr(provider._opener, "open", fake_open)
    report = _report(dense_path)
    block = analyze_report(report, provider=provider)
    assert block["status"] == "ok"
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"].get("Authorization") == "Bearer sk-supersecretvalue123"
    assert "sk-supersecretvalue123" not in captured["body"]
    for row in dense_matrix[:3]:
        assert repr(row.tolist()) not in captured["body"]
    # the key must never reach the persisted report either
    report["ai_analysis"] = block
    assert "sk-supersecretvalue123" not in dump_json(report)


@pytest.mark.parametrize("api_key", ["abc+defghijklmnop\n", "abc/defghijkl\n"])
def test_malformed_configured_keys_are_rejected_without_leaking(dense_path, api_key):
    block = analyze_report(
        _report(dense_path),
        env={"VSAG_AI_API_KEY": api_key, "VSAG_AI_MODEL": "m"},
    )
    assert block["status"] == "error"
    assert "visible ASCII" in block["error"]
    assert api_key.strip() not in block["error"]
    assert json.dumps(api_key)[1:-1] not in block["error"]


def test_analyze_report_redacts_configured_key_in_escaped_exception(dense_path):
    api_key = "abc/defghijkl\n"

    class LeakyProvider:
        name = "leaky"
        model = "m"
        redaction_secrets = (api_key,)

        def complete(self, prompt):
            raise ValueError("invalid Authorization value %r" % ("Bearer " + api_key))

    block = analyze_report(_report(dense_path), provider=LeakyProvider())
    assert block["status"] == "error"
    assert api_key not in block["error"]
    assert api_key.strip() not in block["error"]
    assert json.dumps(api_key)[1:-1] not in block["error"]
    assert "[redacted]" in block["error"]
    dump_json({"ai_analysis": block})


@pytest.mark.parametrize("bare_string", [False, True])
def test_mock_provider_follows_redaction_secrets_contract(dense_path, bare_string):
    api_key = "custom-provider-secret"

    class LeakyMockProvider(MockProvider):
        def complete(self, prompt):
            raise ValueError("request failed with credential " + api_key)

    secrets = api_key if bare_string else (api_key,)
    provider = LeakyMockProvider("", secrets=secrets)
    block = analyze_report(_report(dense_path), provider=provider)
    assert provider.redaction_secrets == (api_key,)
    assert block["status"] == "error"
    assert api_key not in block["error"]
    assert "request failed with credential" in block["error"]
    assert "[redacted]" in block["error"]
