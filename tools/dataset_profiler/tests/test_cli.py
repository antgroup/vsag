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

"""End-to-end tests for the command-line interface."""

from __future__ import annotations

import json

import pytest

from dataset_profiler import cli


def test_default_outputs_written_next_to_input(dense_path, capsys):
    rc = cli.main([str(dense_path), "--sample-size", "16", "--pairwise-sample-size", "8"], env={})
    assert rc == 0
    json_path = dense_path.with_name("dense_toy.profile.json")
    html_path = dense_path.with_name("dense_toy.profile.html")
    assert json_path.is_file() and html_path.is_file()
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == "1.0"
    assert "ai_analysis" not in report
    assert html_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    assert "wrote" in capsys.readouterr().err


def test_explicit_output_paths_and_stdout(dense_path, tmp_path, capsys):
    out_json = tmp_path / "nested" / "report.json"
    rc = cli.main(
        [str(dense_path), "--out-json", str(out_json), "--sample-size", "8"], env={}
    )
    assert rc == 0
    assert out_json.is_file()
    assert not (tmp_path / "nested" / "report.html").exists()

    rc = cli.main([str(dense_path), "--out-json", "-", "--sample-size", "8"], env={})
    assert rc == 0
    stdout = capsys.readouterr().out
    parsed = json.loads(stdout)
    assert parsed["dataset"]["count"] == 60


def test_two_cli_runs_are_byte_identical(dense_path, tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    args = ["--out-json", str(a), "--sample-size", "20", "--pairwise-sample-size", "10"]
    assert cli.main([str(dense_path)] + args, env={}) == 0
    args_b = ["--out-json", str(b), "--sample-size", "20", "--pairwise-sample-size", "10"]
    assert cli.main([str(dense_path)] + args_b, env={}) == 0
    assert a.read_bytes() == b.read_bytes()


def test_invalid_dataset_returns_error_exit_code(tmp_path):
    missing = tmp_path / "nope.hdf5"
    rc = cli.main([str(missing)], env={})
    assert rc == 1


def test_invalid_config_is_rejected(dense_path):
    rc = cli.main([str(dense_path), "--clusters", "-1"], env={})
    assert rc == 1


def test_unsupported_metric_choice_rejected(dense_path):
    with pytest.raises(SystemExit):
        cli.main([str(dense_path), "--metric", "hamming"], env={})


def test_ai_failure_still_writes_reports(dense_path, tmp_path, capsys):
    out = tmp_path / "profile"
    rc = cli.main(
        [
            str(dense_path),
            "--ai",
            "--sample-size", "8",
            "--out-json", str(out / "r.json"),
            "--out-html", str(out / "r.html"),
        ],
        env={},  # no credentials configured
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "AI analysis failed" in err
    report = json.loads((out / "r.json").read_text(encoding="utf-8"))
    assert report["ai_analysis"]["status"] == "error"
    assert (out / "r.html").read_text(encoding="utf-8").find("AI analysis unavailable") != -1


def test_ai_enabled_by_env_var_without_flag(dense_path, tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = cli.main(
        [str(dense_path), "--sample-size", "8", "--out-json", str(out)],
        env={"VSAG_DATASET_PROFILER_AI": "1"},
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ai_analysis"]["status"] == "error"  # still no credentials


def test_sparse_cli_run_produces_html(sparse_path, tmp_path):
    out = tmp_path / "s.profile.html"
    rc = cli.main(
        [
            str(sparse_path),
            "--sample-size", "12",
            "--pairwise-sample-size", "6",
            "--out-json", str(tmp_path / "s.json"),
            "--out-html", str(out),
        ],
        env={}
    )
    assert rc == 0
    assert out.is_file()
    report = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert report["dataset"]["vector_type"] == "sparse"


def test_help_mentions_key_flags():
    parser = cli.build_parser()
    text = parser.format_help()
    for flag in ("--seed", "--sample-size", "--pairwise-sample-size", "--ai", "--out-html"):
        assert flag in text
    assert "omitted when --out-json is given without --out-html" in " ".join(text.split())
