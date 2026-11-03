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

"""Provider-neutral AI-assisted interpretation of a dataset profile.

The adapter accepts the *structured, aggregated* profile report produced by
:func:`dataset_profiler.profiler.profile_dataset` and asks a language model to
explain it.  Raw vectors are never part of the request: the payload contains
only aggregate metrics, sampling configuration, and bounded non-sensitive
metadata.

Configuration is explicit and environment-based so that credentials never end
up in generated reports, command output, source code, or logs:

* ``VSAG_AI_API_KEY``   (fallback: ``OPENAI_API_KEY``)  -- bearer token.
* ``VSAG_AI_MODEL``     -- model name required when AI is enabled.
* ``VSAG_AI_BASE_URL``  (fallback: ``OPENAI_BASE_URL``) -- OpenAI-compatible
  endpoint root, defaulting to ``https://api.openai.com/v1``.
* ``VSAG_AI_KEY_ENV``   -- name of the environment variable holding the key
  (used when a caller wants a custom variable).

Every finding and recommendation returned by the model must reference metric
keys that exist in the report; responses that do not parse, omit evidence, or
invent evidence keys are rejected.  Provider failures never prevent the
deterministic reports from being written: :func:`analyze_report` degrades to a
``{"status": "error", ...}`` entry with a redacted message.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

AI_SCHEMA_VERSION = "1.0"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_LIST_ITEMS = 10
MAX_TEXT_LENGTH = 2000
VALID_CONFIDENCE = ("low", "medium", "high")

_ERROR_REDACTIONS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}"),
)


class AIAnalysisError(RuntimeError):
    """Raised for configuration, transport, or response-validation failures."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
_SYSTEM_INSTRUCTIONS = (
    "You are interpreting a deterministic statistical profile of a vector "
    "dataset produced by the VSAG dataset profiler. You did NOT receive raw "
    "vectors; you receive only aggregate metrics. Respond with a single JSON "
    "object and nothing else, using exactly this schema:\n"
    "{\n"
    '  "summary": "<short plain-language summary>",\n'
    '  "findings": [{"title": str, "detail": str, "evidence": [metric_key],\n'
    '               "confidence": "low"|"medium"|"high"}],\n'
    '  "recommendations": [{"title": str, "detail": str, "evidence": [metric_key],\n'
    '                      "validation": "<how to test this in a VSAG benchmark>",\n'
    '                      "confidence": "low"|"medium"|"high"}],\n'
    '  "limitations": [str],\n'
    "  \"additional_measurements\": [str]\n"
    "}\n"
    "Rules:\n"
    "1. Every finding and recommendation must cite one or more evidence keys "
    "from the provided list; never cite keys that are not listed.\n"
    "2. Distinguish full-dataset metrics from sampled metrics; sampled "
    "conclusions are hypotheses about the sample.\n"
    "3. Recommendations must be phrased as hypotheses to validate with "
    "reproducible VSAG benchmarks, never as guaranteed optimal "
    "configurations.\n"
    "4. Report uncertainty honestly in limitations and additional_measurements."
)


def evidence_metric_keys(report: Dict[str, Any]) -> List[str]:
    """Return dotted ``metrics.*`` keys that evidence references may use."""
    from .stats import metric_keys

    return metric_keys(report)


def _resolve_evidence_key(report: Dict[str, Any], key: str) -> bool:
    """Return True when ``key`` is a ``metrics.*`` path present in the report."""
    if not key.startswith("metrics."):
        return False
    node: Any = report.get("metrics", {})
    for part in key.split(".")[1:]:
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False
    return True


def aggregate_payload(report: Dict[str, Any]) -> Dict[str, Any]:
    """Build the bounded, aggregate-only request payload from a report.

    Only aggregate metrics, sampling configuration, warnings, and non-sensitive
    metadata are included.  The full file path and any per-vector raw content
    are never sent.
    """
    return {
        "schema_version": report.get("schema_version"),
        "tool": report.get("tool"),
        "split": report.get("input", {}).get("split"),
        "dataset": report.get("dataset"),
        "config": report.get("config"),
        "metrics": report.get("metrics"),
        "warnings": report.get("warnings"),
    }


def build_prompt(report: Dict[str, Any]) -> str:
    """Render the user prompt sent to the provider (JSON payload + evidence keys)."""
    payload = aggregate_payload(report)
    keys = evidence_metric_keys(report)
    parts = [
        _SYSTEM_INSTRUCTIONS,
        "",
        "Allowed evidence metric keys:",
        json.dumps(keys, sort_keys=True),
        "",
        "Aggregate profile payload:",
        json.dumps(payload, sort_keys=True, allow_nan=False),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parsing / validation
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$", re.MULTILINE)


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise AIAnalysisError("empty AI response")
    cleaned = _FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise AIAnalysisError("AI response is not a JSON object")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise AIAnalysisError("AI response is not valid JSON: %s" % exc)
    if not isinstance(parsed, dict):
        raise AIAnalysisError("AI response JSON must be an object")
    return parsed


def _clean_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AIAnalysisError("AI response field %r must be a non-empty string" % field)
    return value.strip()[:MAX_TEXT_LENGTH]


def _clean_item(
    item: Any, report: Dict[str, Any], field: str, index: int, require_validation: bool
) -> Dict[str, Any]:
    where = "%s[%d]" % (field, index)
    if not isinstance(item, dict):
        raise AIAnalysisError("AI response %s must be an object" % where)
    entry: Dict[str, Any] = {
        "title": _clean_text(item.get("title"), where + ".title"),
        "detail": _clean_text(item.get("detail"), where + ".detail"),
    }
    evidence = item.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise AIAnalysisError("AI response %s must cite non-empty evidence" % where)
    keys: List[str] = []
    for key in evidence:
        if not isinstance(key, str) or not _resolve_evidence_key(report, key):
            raise AIAnalysisError(
                "AI response %s cites unknown evidence key %r" % (where, key)
            )
        if key not in keys:
            keys.append(key)
    entry["evidence"] = keys
    confidence = item.get("confidence", "medium")
    if confidence not in VALID_CONFIDENCE:
        raise AIAnalysisError(
            "AI response %s.confidence must be one of %s"
            % (where, ", ".join(VALID_CONFIDENCE))
        )
    entry["confidence"] = confidence
    if require_validation:
        entry["validation"] = _clean_text(
            item.get("validation"), where + ".validation"
        )
    return entry


def _clean_string_list(value: Any, field: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > MAX_LIST_ITEMS:
        raise AIAnalysisError("AI response %r must be a list of at most %d items" % (field, MAX_LIST_ITEMS))
    return [_clean_text(item, "%s item" % field) for item in value]


def parse_and_validate_response(text: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Parse a raw provider response into the validated ``ai_analysis`` block.

    Raises :class:`AIAnalysisError` when the response is malformed, cites
    evidence keys that do not exist in the report, or omits required fields.
    """
    parsed = _extract_json_object(text)
    result: Dict[str, Any] = {"summary": _clean_text(parsed.get("summary"), "summary")}

    for field, require_validation in (("findings", False), ("recommendations", True)):
        items = parsed.get(field)
        if not isinstance(items, list) or not items:
            raise AIAnalysisError("AI response %r must be a non-empty list" % field)
        if len(items) > MAX_LIST_ITEMS:
            raise AIAnalysisError("AI response %r must have at most %d items" % (field, MAX_LIST_ITEMS))
        result[field] = [
            _clean_item(item, report, field, index, require_validation)
            for index, item in enumerate(items)
        ]
    result["limitations"] = _clean_string_list(parsed.get("limitations"), "limitations")
    result["additional_measurements"] = _clean_string_list(
        parsed.get("additional_measurements"), "additional_measurements"
    )
    return result


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------
class MockProvider:
    """Deterministic provider that replays a canned response (tests/demos)."""

    name = "mock"

    def __init__(self, response: str, model: str = "mock-model") -> None:
        self._response = response
        self.model = model
        self.last_prompt: Optional[str] = None

    def complete(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response


class OpenAICompatibleProvider:
    """Calls an OpenAI-compatible ``/chat/completions`` endpoint over HTTPS."""

    name = "openai-compatible"

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if not api_key:
            raise AIAnalysisError("AI analysis enabled but no API key was found in the environment")
        if not model:
            raise AIAnalysisError("AI analysis enabled but no model was configured (set VSAG_AI_MODEL or pass --ai-model)")
        if not base_url.startswith("https://"):
            raise AIAnalysisError(
                "AI base URL must use https so the API key is never sent in the clear"
            )
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def complete(self, prompt: str) -> str:
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                },
                sort_keys=True,
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise AIAnalysisError(
                "AI provider returned HTTP %s for %s" % (exc.code, self.base_url)
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AIAnalysisError("AI provider is unreachable: %s" % type(exc).__name__)
        except json.JSONDecodeError:
            raise AIAnalysisError("AI provider returned a non-JSON response body")
        try:
            return str(body["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise AIAnalysisError("AI provider response is missing choices[0].message.content")


def provider_from_env(
    env: Optional[Dict[str, str]] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    key_env: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> OpenAICompatibleProvider:
    """Build a provider from the environment; credentials are never persisted."""
    env = os.environ if env is None else env
    key_name = key_env or env.get("VSAG_AI_KEY_ENV") or "VSAG_AI_API_KEY"
    api_key = env.get(key_name) or env.get("OPENAI_API_KEY") or ""
    resolved_model = model or env.get("VSAG_AI_MODEL") or ""
    resolved_base = base_url or env.get("VSAG_AI_BASE_URL") or env.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    return OpenAICompatibleProvider(
        api_key=api_key,
        model=resolved_model,
        base_url=resolved_base,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _redact(message: str) -> str:
    for pattern in _ERROR_REDACTIONS:
        message = pattern.sub("[redacted]", message)
    return message[:500]


def analyze_report(
    report: Dict[str, Any],
    provider: Optional[Any] = None,
    env: Optional[Dict[str, str]] = None,
    **provider_kwargs: Any
) -> Dict[str, Any]:
    """Produce the ``ai_analysis`` block for ``report``.

    Never raises: configuration, transport, and validation failures are folded
    into a ``{"status": "error"}`` block so the deterministic JSON and HTML
    reports are still written.
    """
    if provider is None:
        try:
            provider = provider_from_env(env=env, **provider_kwargs)
        except AIAnalysisError as exc:
            return {"status": "error", "provider": None, "error": _redact(str(exc))}
    name = getattr(provider, "name", type(provider).__name__)
    model = getattr(provider, "model", None)
    try:
        raw = provider.complete(build_prompt(report))
        block = parse_and_validate_response(raw, report)
    except AIAnalysisError as exc:
        return {
            "status": "error",
            "provider": name,
            "model": model,
            "error": _redact(str(exc)),
        }
    except Exception as exc:  # defensive: providers may raise transport errors
        return {
            "status": "error",
            "provider": name,
            "model": model,
            "error": _redact(
                "unexpected AI failure: %s: %s" % (type(exc).__name__, exc)
            ),
        }
    block.update(
        {
            "status": "ok",
            "provider": name,
            "model": model,
            "ai_schema_version": AI_SCHEMA_VERSION,
        }
    )
    return block
