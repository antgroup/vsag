#!/usr/bin/env python3
"""Focused tests for the build-performance report parser."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "collect_build_metrics", ROOT / "scripts/collect_build_metrics.py"
)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class BuildMetricsTest(unittest.TestCase):
    def test_parse_ninja_log_classifies_edges_and_sorts_translation_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ninja_log = Path(directory) / ".ninja_log"
            ninja_log.write_text(
                "# ninja log v5\n"
                "0\t2500\t0\tsrc/CMakeFiles/vsag.dir/index.cpp.o\thash\n"
                "0\t900\t0\ttests/CMakeFiles/unittests.dir/test.cpp.o\thash\n"
                "0\t500\t0\t_deps/fmt-build/CMakeFiles/fmt.dir/format.cc.o\thash\n"
                "500\t1300\t0\thdf5-prefix/src/hdf5-stamp/hdf5-build\thash\n"
                "2500\t2800\t0\tsrc/libvsag.so\thash\n",
                encoding="utf-8",
            )

            result = METRICS.parse_ninja_log(
                ninja_log, {"index.cpp.o": "src/index.cpp", "test.cpp.o": "tests/test.cpp"}
            )

        self.assertEqual(result["categories"]["production_compile"]["cumulative_seconds"], 2.5)
        self.assertEqual(result["categories"]["test_compile"]["cumulative_seconds"], 0.9)
        self.assertEqual(result["categories"]["dependency_compile"]["cumulative_seconds"], 0.5)
        self.assertEqual(result["categories"]["dependency_build"]["cumulative_seconds"], 0.8)
        self.assertEqual(result["categories"]["link"]["cumulative_seconds"], 0.3)
        self.assertEqual(result["slowest_translation_units"][0]["source"], "src/index.cpp")

    def test_rendered_summary_uses_real_newlines(self) -> None:
        report = {
            "status": "success",
            "configuration": {
                "compiler": "gcc",
                "jobs": 3,
                "dependency_preparation_seconds": 1.25,
                "compiler_cache_key": "compiler-key",
                "dependency_cache_key": "dependency-key",
            },
            "phases": [
                {
                    "name": "configure",
                    "elapsed_seconds": 2.5,
                    "peak_rss_kib": None,
                    "ccache": {"cache_hit": 0, "cache_miss": 0},
                }
            ],
        }

        summary = METRICS.render_markdown(report, concise=True)

        self.assertIn("\n| Phase | Wall time", summary)
        self.assertNotIn("\\n", summary)
        self.assertTrue(summary.endswith("\n"))

    def test_ccache_json_supports_version_four_stats(self) -> None:
        stats = METRICS.load_ccache_stats('{"stats":{"cache_hit":7,"cache_miss":2}}')

        self.assertEqual(stats["cache_hit"], 7)
        self.assertEqual(stats["cache_miss"], 2)

    def test_ccache_machine_stats_support_ubuntu_version(self) -> None:
        stats = METRICS.load_ccache_stats(
            "direct_cache_hit\t5\npreprocessed_cache_hit\t2\ncache_miss\t3\n"
        )

        self.assertEqual(stats["cache_hit"], 7)
        self.assertEqual(stats["cache_miss"], 3)

    def test_collector_writes_json_markdown_and_summary_reports(self) -> None:
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                args = SimpleNamespace(
                    build_dir="build",
                    output_dir="metrics",
                    jobs=3,
                    base_sha="base",
                    commit_sha="commit",
                    compiler="gcc-12",
                    compiler_cache_key="compiler-key",
                    dependency_cache_key="dependency-key",
                    dependency_preparation_seconds=1.25,
                    clear_ccache=True,
                )
                collector = METRICS.Collector(args)
                collector.phases = [
                    {
                        "name": "configure",
                        "elapsed_seconds": 2.5,
                        "peak_rss_kib": 1024,
                        "ccache": {"cache_hit": 0, "cache_miss": 0},
                    }
                ]
                collector.write_reports()

                report = json.loads(Path("metrics/build-metrics.json").read_text())
                summary = Path("metrics/job-summary.md").read_text()
            finally:
                os.chdir(previous_directory)

        self.assertEqual(report["base_sha"], "base")
        self.assertIn("\n| Phase | Wall time", summary)
        self.assertTrue(summary.endswith("\n"))


if __name__ == "__main__":
    unittest.main()
