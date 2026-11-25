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

"""Result formatters and exporters, mirroring those of the ``eval_performance`` tool."""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

FORMATS = ("json", "table", "text", "line_protocol")

_TABLE_HEADER = (
    "Name",
    "Index",
    "NumVectors",
    "Dim",
    "DataType",
    "MetricType",
    "IndexParam",
    "Memory(build)",
    "BuildTime",
    "TPS",
    "SearchParam",
    "Memory(search)",
    "QPS",
    "LatencyAvg(ms)",
    "RecallAvg",
)


def _field(result, path, default, convert=str):
    value = result
    try:
        for key in path:
            value = value[key]
        if value is None:
            return default
        return convert(value)
    except (KeyError, TypeError, ValueError):
        return default


def _number(value):
    return "%f" % float(value)


def _compact_json(value):
    return json.dumps(value, separators=(",", ":"))


def _format_json(results):
    return json.dumps(results, separators=(",", ":"))


def _format_table(results):
    rows = [list(_TABLE_HEADER)]
    for name, result in results.items():
        rows.append(
            [
                name,
                _field(result, ["index"], "N/A"),
                _field(result, ["dataset_info", "base_count"], "N/A"),
                _field(result, ["dataset_info", "dim"], "N/A"),
                _field(result, ["dataset_info", "data_type"], "N/A"),
                _field(result, ["index_info", "metric_type"], "N/A"),
                _field(result, ["index_info", "index_param"], "N/A", _compact_json),
                _field(result, ["memory_peak(build)"], "N/A"),
                _field(result, ["duration(s)"], "N/A", _number),
                _field(result, ["tps"], "N/A", _number),
                _field(result, ["search_param"], "N/A"),
                _field(result, ["memory_peak(search)"], "N/A"),
                _field(result, ["qps"], "N/A", _number),
                _field(result, ["latency_avg(ms)"], "N/A", _number),
                _field(result, ["recall_avg"], "N/A", _number),
            ]
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(_TABLE_HEADER))]
    separator = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    lines = [separator]
    for row in rows:
        lines.append(
            "| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |"
        )
        lines.append(separator)
    return "\n".join(lines)


def _format_line_protocol(results):
    now = time.time_ns()
    lines = []
    for name, result in results.items():
        values = [
            ("num_vectors", _field(result, ["dataset_info", "base_count"], "N/A")),
            ("dim", _field(result, ["dataset_info", "dim"], "-1")),
            ("build_time", _field(result, ["duration(s)"], "-1", _number)),
            ("tps", _field(result, ["tps"], "-1", _number)),
            ("qps", _field(result, ["qps"], "-1", _number)),
            ("latency_avg", _field(result, ["latency_avg(ms)"], "-1", _number)),
            ("latency_p90", _field(result, ["latency_detail(ms)", "p90"], "-1", _number)),
            ("latency_p95", _field(result, ["latency_detail(ms)", "p95"], "-1", _number)),
            ("latency_p99", _field(result, ["latency_detail(ms)", "p99"], "-1", _number)),
            ("recall_avg", _field(result, ["recall_avg"], "-1", _number)),
        ]
        lines.append(
            "performance,case_name=%s,index_name=%s,dtype=%s %s %d"
            % (
                name,
                _field(result, ["index"], "-1"),
                _field(result, ["dataset_info", "data_type"], "-1"),
                ",".join("%s=%s" % pair for pair in values),
                now,
            )
        )
    return "".join(line + "\n" for line in lines)


_FORMATTERS = {
    "json": _format_json,
    "table": _format_table,
    "text": _format_table,
    "line_protocol": _format_line_protocol,
}


def format_results(results, format="table"):
    """Render ``{case_name: result}`` as ``"json"``, ``"table"`` (alias ``"text"``) or
    ``"line_protocol"`` (InfluxDB), like the C++ tool."""
    if format not in _FORMATTERS:
        raise ValueError("unknown format %r, expected one of %s" % (format, list(FORMATS)))
    return _FORMATTERS[format](results)


def _export_stdout(text, _vars):
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    sys.stdout.flush()
    return True


def _export_file(path, append):
    def export(text, _vars):
        with open(path, "a" if append else "w", encoding="utf-8") as file:
            file.write(text)
        return True

    return export


def _export_influxdb(endpoint, variables):
    if "token" not in variables:
        raise ValueError("influxdb exporter requires vars.token")
    url = "http" + endpoint[len("influxdb") :]

    def export(text, _vars):
        request = urllib.request.Request(
            url,
            data=text.encode("utf-8"),
            method="POST",
            headers={
                "Authorization": str(variables["token"]),
                "Content-Type": "text/plain; charset=utf-8",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status == 204
        except urllib.error.HTTPError as error:
            sys.stderr.write("HTTP Error: %d - %s\n" % (error.code, error.read().decode()))
        except urllib.error.URLError as error:
            sys.stderr.write("HTTP Request failed: %s\n" % error.reason)
        return False

    return export


def _create_exporter(to, variables):
    if to == "stdout":
        return _export_stdout
    if to.startswith("appendfile://"):
        return _export_file(to[len("appendfile:/") :], append=True)
    if to.startswith("file://"):
        return _export_file(to[len("file:/") :], append=False)
    if to.startswith("influxdb://"):
        return _export_influxdb(to, variables)
    raise ValueError(
        "unknown exporter destination %r, expected stdout, file://, appendfile:// or influxdb://"
        % to
    )


def export_results(results, format="table", to="stdout", vars=None):
    """Format ``results`` and send them to ``to``.

    ``to`` is ``"stdout"``, ``"file://<path>"``, ``"appendfile://<path>"`` or
    ``"influxdb://host:port/path?query"`` (needs ``vars={"token": ...}``). Returns ``True`` on
    success. Raises ``ValueError`` for unknown formats or destinations.
    """
    text = format_results(results, format)
    return _create_exporter(to, vars or {})(text, vars)
