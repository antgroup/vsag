#!/usr/bin/env python3
"""Focused tests for the build-performance report parser."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "collect_build_metrics", ROOT / "scripts/collect_build_metrics.py"
)
assert SPEC is not None and SPEC.loader is not None
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


class BuildMetricsTest(unittest.TestCase):
    def test_dependency_paths_require_strict_repository_descendants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            default = root / "_deps"
            alias = root / "root-alias"
            alias.symlink_to(root, target_is_directory=True)
            for value in (str(root), str(root / "."), str(alias), str(root.parent)):
                with self.subTest(value=value):
                    self.assertIsNone(METRICS.local_dependency_path(root, value, default))
            self.assertIsNone(METRICS.local_dependency_path(root, None, root))
            self.assertEqual(METRICS.local_dependency_path(root, None, default), default)
            default.mkdir()
            self.assertEqual(METRICS.local_dependency_path(root, str(default), root), default)

    def test_missing_ninja_log_cannot_verify_noop_or_reuse_stale_capture(self) -> None:
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                for exit_code in (0, 1):
                    for stale_capture in (False, True):
                        with self.subTest(exit_code=exit_code, stale_capture=stale_capture):
                            collector = METRICS.Collector(self.collector_args())
                            captured = collector.output_dir / "noop_build.ninja_log"
                            if stale_capture:
                                captured.write_text("# ninja log v5\n", encoding="utf-8")
                            phase = {
                                "name": "noop_build", "exit_code": exit_code,
                                "elapsed_seconds": 0.1,
                            }
                            collector.run_logged = lambda *arguments: phase
                            collector.ccache = lambda *arguments: {}
                            collector.capture_build("noop_build", "no-op")
                            self.assertFalse(phase["ninja"]["available"])
                            self.assertIsNone(phase["ninja_edges"])
                            self.assertIsNone(phase["build_edges"])
                            self.assertFalse(phase["true_noop"])
                            self.assertFalse(captured.exists())
                            summary = METRICS.render_markdown({
                                "status": "ok",
                                "configuration": {
                                    "compiler": "test", "jobs": 1,
                                    "dependency_preparation_seconds": 0.0,
                                    "compiler_cache_key": "test",
                                    "dependency_cache_key": "test",
                                },
                                "phases": [phase],
                            }, concise=True)
                            self.assertNotIn("verified", summary)
                            self.assertNotIn("None", summary)
                            self.assertIn("| n/a |", summary)
            finally:
                os.chdir(previous_directory)

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

    def test_dependency_edges_cover_ci_sources_external_stages_and_multi_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ninja_log = Path(directory) / ".ninja_log"
            ninja_log.write_text(
                "# ninja log v5\n"
                "0\t700\t0\t/home/runner/vsag/.ci-fetchcontent/catch2-build/"
                "src/CMakeFiles/Catch2.dir/catch.cpp.o\tcatch-hash\n"
                "700\t1700\t0\tbuild/.vsag-build-info/antlr4-configure\tantlr-configure\n"
                "1700\t4200\t0\tbuild/.vsag-build-info/hdf5-build\thdf5-build\n"
                "1700\t4200\t0\tbuild/hdf5/install/lib/libhdf5.a\thdf5-build\n"
                "4200\t5100\t0\textern/antlr4/CMakeFiles/antlr4-autogen.dir/"
                "fc/FCLexer.cpp.o\tantlr-compile\n",
                encoding="utf-8",
            )

            result = METRICS.parse_ninja_log(ninja_log, {})

        self.assertEqual(result["raw_log_records"], 5)
        self.assertEqual(result["edge_count"], 4)
        self.assertEqual(result["deduplicated_output_records"], 1)
        self.assertEqual(result["categories"]["dependency_compile"]["edges"], 2)
        self.assertEqual(result["categories"]["dependency_build"]["edges"], 2)
        self.assertEqual(result["dependencies"]["catch2"]["stages"]["compile"]["edges"], 1)
        self.assertEqual(
            result["dependencies"]["antlr4"]["stages"]["configure"]["cumulative_seconds"],
            1.0,
        )
        self.assertEqual(result["dependencies"]["antlr4"]["stages"]["compile"]["edges"], 1)
        self.assertEqual(result["dependencies"]["hdf5"]["stages"]["build"]["edges"], 1)
        self.assertEqual(
            result["dependencies"]["hdf5"]["stages"]["build"]["cumulative_seconds"],
            2.5,
        )

    def test_deduplication_does_not_merge_nonconsecutive_matching_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ninja_log = Path(directory) / ".ninja_log"
            ninja_log.write_text(
                "# ninja log v5\n"
                "0\t100\t7\tsrc/CMakeFiles/vsag.dir/a.cpp.o\tshared-hash\n"
                "0\t100\t7\tsrc/CMakeFiles/vsag.dir/middle.cpp.o\tother-hash\n"
                "0\t100\t7\tsrc/CMakeFiles/vsag.dir/b.cpp.o\tshared-hash\n",
                encoding="utf-8",
            )

            result = METRICS.parse_ninja_log(ninja_log, {})

        self.assertEqual(result["raw_log_records"], 3)
        self.assertEqual(result["edge_count"], 3)
        self.assertEqual(result["deduplicated_output_records"], 0)

    def test_regular_dependency_build_output_does_not_override_compile_priority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ninja_log = Path(directory) / ".ninja_log"
            ninja_log.write_text(
                "# ninja log v5\n"
                "0\t100\t7\textern/example/CMakeFiles/example.dir/example.cpp.o\thash\n"
                "0\t100\t7\textern/example/libexample.a\thash\n",
                encoding="utf-8",
            )

            result = METRICS.parse_ninja_log(ninja_log, {})

        self.assertEqual(result["edge_count"], 1)
        self.assertEqual(result["categories"]["dependency_compile"]["edges"], 1)
        self.assertEqual(result["dependencies"]["example"]["stages"]["compile"]["edges"], 1)

    def test_link_outputs_are_normalized_before_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ninja_log = Path(directory) / ".ninja_log"
            ninja_log.write_text(
                "# ninja log v5\n0\t100\t7\tbin/custom-output\thash\n",
                encoding="utf-8",
            )

            result = METRICS.parse_ninja_log(
                ninja_log, {}, {"./BIN/CUSTOM-OUTPUT"}
            )

        self.assertEqual(result["categories"]["link"]["edges"], 1)

    def test_dependency_resolution_records_source_system_and_fallback(self) -> None:
        resolutions = METRICS.parse_dependency_resolutions(
            "-- Third-party override: dependency=ANTLR4, pin=4.13.2, source=pinned, "
            "variable=VSAG_THIRDPARTY_ANTLR4_4_13_2; legacy fallback "
            "VSAG_THIRDPARTY_ANTLR4 is unset; default URLs remain download fallbacks\n"
            "-- Third-party override: dependency=HDF5, pin=hdf5_1.14.4, source=default, "
            "variable=none; expected pinned variable VSAG_THIRDPARTY_HDF5_1_14_4; "
            "deprecated legacy fallback VSAG_THIRDPARTY_HDF5 is unset\n"
            "-- Using system OpenBLAS as BLAS backend\n"
        )

        self.assertEqual(resolutions["antlr4"]["resolution"], "source")
        self.assertEqual(resolutions["antlr4"]["source_origin"], "pinned")
        self.assertEqual(resolutions["antlr4"]["pin"], "4.13.2")
        self.assertEqual(resolutions["hdf5"]["source_origin"], "default")
        self.assertEqual(resolutions["openblas"]["resolution"], "system")
        self.assertIn("fallback", resolutions["antlr4"]["fallback"])

    def test_pybind11_fetchcontent_preparation_and_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for layout, preparation in (
                ("_deps", "FetchContent source tree"),
                (".ci-fetchcontent", "pinned FetchContent checkout"),
            ):
                with self.subTest(layout=layout):
                    fetchcontent_dir = root / layout
                    source_dir = fetchcontent_dir / "pybind11-src"
                    build_dir = fetchcontent_dir / "pybind11-build"
                    source_dir.mkdir(parents=True)
                    build_dir.mkdir()
                    (source_dir / "CMakeLists.txt").write_bytes(b"source fixture\n")
                    (build_dir / "cmake_install.cmake").write_bytes(b"build fixture\n")
                    ninja_log = root / ".ninja_log"
                    ninja_log.write_text(
                        "# ninja log v5\n"
                        f"0\t100\t0\t{layout}/pybind11-build/cmake_install.cmake\thash\n",
                        encoding="utf-8",
                    )
                    ninja = METRICS.parse_ninja_log(ninja_log, {})
                    report = METRICS.build_dependency_report(
                        {}, ninja["dependencies"], root / "build", fetchcontent_dir, {}
                    )

                    self.assertEqual(set(report), {"pybind11"})
                    self.assertEqual(report["pybind11"]["preparation"], preparation)
                    self.assertEqual(report["pybind11"]["source_bytes"], 15)
                    self.assertEqual(report["pybind11"]["build_bytes"], 14)

    def test_unclassified_source_dependency_is_not_reported_as_host_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = METRICS.build_dependency_report(
                {},
                {
                    "unknown_dep": {
                        "edges": 1,
                        "cumulative_seconds": 0.1,
                        "stages": {"build": {"edges": 1, "cumulative_seconds": 0.1}},
                    }
                },
                root / "build",
                None,
                {},
            )

        self.assertEqual(
            report["unknown_dep"]["preparation"], "source preparation not classified"
        )
        self.assertEqual(
            report["unknown_dep"]["fallback"],
            "classification path unknown; metrics may be incomplete",
        )

    def test_source_openblas_without_override_does_not_claim_a_stale_pin(self) -> None:
        resolutions = METRICS.parse_dependency_resolutions(
            "-- Building OpenBLAS from source\n"
        )

        self.assertEqual(resolutions["openblas"]["pin"], "unknown")

    def test_cmake_cache_parser_preserves_value_delimiters_and_skips_untyped_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "CMakeCache.txt"
            cache.write_text(
                "CMAKE_CXX_FLAGS:STRING=-DFOO=bar:https://example.test\n"
                "UNTYPED=https://example.test:value\n",
                encoding="utf-8",
            )

            values = METRICS.read_cmake_cache(cache)

        self.assertEqual(
            values["CMAKE_CXX_FLAGS"], "-DFOO=bar:https://example.test"
        )
        self.assertNotIn("UNTYPED", values)

    def test_rendered_summary_uses_real_newlines(self) -> None:
        report = {
            "status": "success",
            "configuration": {
                "compiler": "gcc",
                "jobs": 3,
                "dependency_preparation_seconds": 1.25,
                "dependency_source_preparation_seconds": 1.0,
                "dependency_cache_restore_seconds": 0.25,
                "dependency_cache_state": "fallback-hit",
                "dependency_cache_write_policy": "restore-only",
                "dependency_storage": {
                    "fetchcontent_bytes": 1048576,
                    "archive_bytes": 524288,
                    "external_build_bytes": 0,
                },
                "compiler_cache_key": "compiler-key",
                "dependency_cache_key": "dependency-key",
                "dependency_cache_matched_key": "dependency-old-key",
            },
            "phases": [
                {
                    "name": "configure",
                    "elapsed_seconds": 2.5,
                    "peak_rss_kib": None,
                    "ccache": {"cache_hit": 0, "cache_miss": 0},
                }
            ],
            "dependencies": {
                "antlr4": {
                    "display_name": "ANTLR4",
                    "resolution": "source",
                    "pin": "4.13.2",
                    "source_origin": "default",
                    "fallback": "default upstream URLs",
                    "source_bytes": 1024,
                    "build_bytes": 2048,
                    "stages": {"build": {"edges": 1, "cumulative_seconds": 3.0}},
                },
                "openblas": {
                    "display_name": "OpenBLAS",
                    "resolution": "system",
                    "pin": "system",
                    "fallback": "bundled source fallback available",
                    "source_bytes": 0,
                    "build_bytes": 0,
                    "stages": {},
                },
            },
        }

        summary = METRICS.render_markdown(report, concise=True)

        self.assertIn("\n| Phase | Wall time", summary)
        self.assertIn("fallback hit", summary)
        self.assertIn("restore only", summary)
        self.assertIn("| ANTLR4 | `4.13.2` | source", summary)
        self.assertIn("| OpenBLAS | `system` | system", summary)
        self.assertNotIn("\\n", summary)
        self.assertTrue(summary.endswith("\n"))

    @staticmethod
    def write_build_fixture(root: Path) -> None:
        shutil.copyfile(ROOT / "Makefile", root / "Makefile")
        (root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.18)\n"
            "project(metrics_fixture NONE)\n"
            "add_custom_command(OUTPUT artifact.txt\n"
            "  COMMAND ${CMAKE_COMMAND} -E touch artifact.txt VERBATIM)\n"
            "add_custom_target(artifact ALL DEPENDS artifact.txt)\n",
            encoding="utf-8",
        )

    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("make", "cmake", "ninja")),
        "requires Make, CMake, and Ninja",
    )
    def test_make_recipes_accept_default_and_raw_build_directory_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_build_fixture(root)
            for build_dir in ("build", "custom build"):
                with self.subTest(build_dir=build_dir):
                    override = [] if build_dir == "build" else [f"DEBUG_BUILD_DIR={root / build_dir}"]
                    for target in ("configure-asan", "build-asan"):
                        result = subprocess.run(
                            ["make", target, "CMAKE_GENERATOR=Ninja", "COMPILE_JOBS=2", *override],
                            cwd=root, text=True, capture_output=True, check=False,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue((root / build_dir / "CMakeCache.txt").is_file())
                    self.assertTrue((root / build_dir / "artifact.txt").is_file())

    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("make", "cmake", "ninja")),
        "requires Make, CMake, and Ninja",
    )
    def test_collection_builds_and_cleans_directory_with_spaces_end_to_end(self) -> None:
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.write_build_fixture(Path(directory))
                args = self.collector_args()
                args.build_dir = "custom build"
                collector = METRICS.Collector(args)
                collector.ccache = lambda *arguments: {}
                collector.write_reports = lambda: None
                with patch.dict(os.environ, {"CMAKE_GENERATOR": "Ninja"}):
                    self.assertEqual(collector.collect(), 0)
                self.assertTrue((collector.build_dir / "artifact.txt").is_file())
                self.assertEqual(
                    [phase["name"] for phase in collector.phases],
                    ["configure", "clean_build", "prepare_warm_cache_build", "warm_cache_build", "noop_build"],
                )
                for phase in collector.phases:
                    self.assertEqual(phase["exit_code"], 0)
                    if phase["command"][0] == "make":
                        self.assertIn(f"DEBUG_BUILD_DIR={collector.build_dir}", phase["command"])
                for index in (1, 3):
                    self.assertGreater(collector.phases[index]["build_edges"], 0)
                self.assertTrue(collector.phases[-1]["true_noop"])
            finally:
                os.chdir(previous_directory)

    def test_collection_orders_clean_warm_cache_and_true_noop_measurements(self) -> None:
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                args = self.collector_args()
                args.build_dir = "custom build"
                collector = METRICS.Collector(args)
                calls = []

                def run_logged(name, command):
                    calls.append(("command", name))
                    if name == "configure":
                        self.assertIn(
                            f"DEBUG_BUILD_DIR={collector.build_dir}", command
                        )
                    phase = {"name": name, "exit_code": 0}
                    collector.phases.append(phase)
                    return phase

                def capture_build(name, measurement_mode):
                    calls.append(("build", name, measurement_mode))

                collector.run_logged = run_logged
                collector.capture_build = capture_build
                collector.ccache = lambda *arguments: {"available": True}
                collector.write_reports = lambda: None

                result = collector.collect()
            finally:
                os.chdir(previous_directory)

        self.assertEqual(result, 0)
        self.assertEqual(
            [call for call in calls if call[0] == "build"],
            [
                ("build", "clean_build", "clean"),
                ("build", "warm_cache_build", "warm-cache"),
                ("build", "noop_build", "no-op"),
            ],
        )
        self.assertLess(calls.index(("build", "warm_cache_build", "warm-cache")), calls.index(("build", "noop_build", "no-op")))

    def test_noop_measurement_is_verified_from_zero_ninja_edges(self) -> None:
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                args = self.collector_args()
                args.build_dir = "custom build"
                collector = METRICS.Collector(args)
                collector.build_dir.mkdir()
                ninja_log = collector.build_dir / ".ninja_log"
                ninja_log.write_text(
                    "# ninja log v7\n0\t100\t0\tsrc/unchanged.cpp.o\told-hash\n",
                    encoding="utf-8",
                )

                def run_logged(name, command):
                    self.assertIn(
                        f"DEBUG_BUILD_DIR={collector.build_dir}", command
                    )
                    phase = {
                        "name": name,
                        "command": command,
                        "elapsed_seconds": 0.1,
                        "exit_code": 0,
                        "peak_rss_kib": None,
                        "ninja_stats": [],
                    }
                    collector.phases.append(phase)
                    return phase

                collector.run_logged = run_logged
                collector.ccache = lambda *arguments: {"available": True}

                collector.capture_build("noop_build", "no-op")
                preserved_ninja_log = ninja_log.read_text(encoding="utf-8")
                captured_ninja_log = Path("metrics/noop_build.ninja_log").read_text(
                    encoding="utf-8"
                )
            finally:
                os.chdir(previous_directory)

        phase = collector.phases[0]
        self.assertEqual(phase["measurement_mode"], "no-op")
        self.assertEqual(phase["ninja_edges"], 0)
        self.assertTrue(phase["true_noop"])
        self.assertIn("src/unchanged.cpp.o", preserved_ninja_log)
        self.assertEqual(captured_ninja_log, "# ninja log v7\n")

    def test_ninja_log_delta_handles_append_and_compaction(self) -> None:
        old = "0\t100\t0\tsrc/old.cpp.o\told-hash"
        new = "0\t200\t0\tsrc/new.cpp.o\tnew-hash"

        self.assertEqual(METRICS.ninja_log_delta([old], [old, new]), [new])
        self.assertEqual(METRICS.ninja_log_delta([old], [new]), [new])

    def test_ninja_housekeeping_is_not_counted_as_noop_build_work(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ninja_log = Path(directory) / ".ninja_log"
            ninja_log.write_text(
                "# ninja log v5\n"
                "0\t10\t0\tbuild/CMakeFiles/cmake.verify_globs\tglob-hash\n"
                "10\t20\t0\tCMakeFiles/version\tversion-hash\n"
                "10\t20\t0\tbuild/CMakeFiles/version\tversion-hash\n",
                encoding="utf-8",
            )

            result = METRICS.parse_ninja_log(ninja_log, {})

        self.assertEqual(result["edge_count"], 2)
        self.assertEqual(result["build_edge_count"], 0)
        self.assertEqual(result["categories"]["build_maintenance"]["edges"], 2)

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
                args = self.collector_args()
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

    @staticmethod
    def collector_args() -> SimpleNamespace:
        return SimpleNamespace(
            build_dir="build",
            output_dir="metrics",
            jobs=3,
            base_sha="base",
            commit_sha="commit",
            compiler="gcc-12",
            compiler_cache_key="compiler-key",
            dependency_cache_key="dependency-key",
            dependency_cache_matched_key="dependency-old-key",
            dependency_cache_state="fallback-hit",
            dependency_cache_write_policy="restore-only",
            dependency_preparation_seconds=1.25,
            dependency_source_preparation_seconds=1.0,
            dependency_cache_restore_seconds=0.25,
            clear_ccache=True,
        )


if __name__ == "__main__":
    unittest.main()
