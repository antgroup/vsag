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
    def test_normalize_output_preserves_path_components(self) -> None:
        for path, expected in (
            ("./SRC/File.cpp.o", "src/file.cpp.o"),
            (".\\SRC\\File.cpp.o", "src/file.cpp.o"),
            ("../SRC/File.cpp.o", "../src/file.cpp.o"),
            ("/SRC/File.cpp.o", "/src/file.cpp.o"),
            (".ci-fetchcontent/File.o", ".ci-fetchcontent/file.o"),
            ("", ""),
        ):
            with self.subTest(path=path):
                self.assertEqual(METRICS.normalize_output(path), expected)

    def test_nested_module_and_auxiliary_classification(self) -> None:
        cases = {
            "src/datacell/CMakeFiles/datacell.dir/flatten_datacell_factory_ip.cpp.o": "production_compile",
            "src/datacell/CMakeFiles/datacell.dir/multi_vector_datacell_factory.cpp.o": "production_compile",
            "src/datacell/CMakeFiles/datacell.dir/flatten_datacell_factory_cosine.cpp.o": "production_compile",
            "src/other/deep/CMakeFiles/module.dir/file.cpp.o": "production_compile",
            "src/module/CMakeFiles/module_test.dir/file.cpp.o": "test_compile",
            "tests/module/CMakeFiles/fixture.dir/file.cpp.o": "test_compile",
            ".ci-fetchcontent/catch2-build/src/CMakeFiles/Catch2.dir/file.cpp.o": "dependency_compile",
            "tools/eval/CMakeFiles/eval.dir/src/file.cpp.o": "tools_compile",
            "examples/cpp/CMakeFiles/example.dir/src/file.cpp.o": "examples_compile",
            "unknown/CMakeFiles/unknown.dir/file.cpp.o": "other_compile",
        }
        for path, category in cases.items():
            for prefix in ("", "./", "/workspace/project/build/"):
                with self.subTest(path=prefix + path):
                    self.assertEqual(METRICS.classify_edge(prefix + path), category)

    def test_cache_reasons_and_rates_do_not_imply_overall_coverage(self) -> None:
        for text in (
            "direct_cache_hit\t590\npreprocessed_cache_hit\t2\ncache_miss\t0\n"
            "could_not_use_precompiled_header\t166\ncalled_for_link\t10\n",
            json.dumps({"stats": {"cache_hit_direct": 590, "cache_hit_preprocessed": 2,
                                  "cache_miss": 0, "could_not_use_precompiled_header": 166,
                                  "called_for_link": 10}}),
        ):
            stats = METRICS.load_ccache_stats(text)
            self.assertEqual(stats["cache_hit"], 592)
            self.assertEqual(stats["cacheable_request_hit_rate"], 1.0)
            self.assertEqual(stats["uncacheable_reasons"]["could_not_use_precompiled_header"], 166)
            self.assertIsNone(stats["uncacheable_reasons"]["could_not_use_modules"])
            self.assertIsNone(stats["overall_cache_coverage"])
        for text in ('{"cache_hit":0,"cache_miss":0}', '{"cache_hit_direct":7}',
                     '{"files_in_cache":3}'):
            self.assertIsNone(METRICS.load_ccache_stats(text)["cacheable_request_hit_rate"])
        for text in ("unavailable", "[]", '{"stats":null}'):
            self.assertFalse(METRICS.load_ccache_stats(text)["available"])

    def test_every_category_is_visible_and_missing_telemetry_is_not_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".ninja_log"
            path.write_text("# ninja log v5\n0\t120\t0\tunknown.cpp.o\ta\n"
                            "0\t200\t0\tgenerated.txt\tb\n")
            ninja = METRICS.parse_ninja_log(path, {})
        self.assertEqual(set(ninja["categories"]), set(METRICS.BUILD_CATEGORIES))
        report = {
            "status": "success",
            "configuration": {"compiler": "fixture", "jobs": 1,
                              "dependency_preparation_seconds": 0,
                              "compiler_cache_key": "test", "dependency_cache_key": "test"},
            "phases": [{"name": "clean_build", "elapsed_seconds": 1,
                        "ninja": ninja, "ccache": METRICS.load_ccache_stats(
                            '{"cache_hit":592,"cache_miss":0,"could_not_use_precompiled_header":166}')},
                       {"name": "noop_build", "elapsed_seconds": 0.1,
                        "ninja": {"available": False}}],
        }
        for concise in (False, True):
            rendered = METRICS.render_markdown(report, concise)
            for category in METRICS.BUILD_CATEGORIES:
                self.assertIn(f"| clean_build | {category} |", rendered)
            self.assertIn("| other_compile | 1 | 0.120 s |", rendered)
            self.assertIn("| noop_build | unavailable | n/a | n/a |", rendered)
            self.assertIn("unverified", rendered)
            self.assertIn("could_not_use_precompiled_header: 166", rendered)
            self.assertIn("100.00% | n/a", rendered)
            self.assertIn("not aggregate concurrent build memory", rendered)
            self.assertIn("not wall time", rendered)
            self.assertNotIn("None", rendered)

    def test_cache_reason_rendering_distinguishes_unavailable_partial_and_zero(self) -> None:
        cases = (
            (METRICS.load_ccache_stats("ccache: error"), "n/a (ccache statistics unavailable)"),
            ({"available": False, "uncacheable_reasons": dict.fromkeys(METRICS.UNCACHEABLE_REASONS, 0)},
             "n/a (ccache statistics unavailable)"),
            (METRICS.load_ccache_stats('{"cache_hit":0,"cache_miss":0}'),
             "could_not_use_precompiled_header: n/a"),
            (METRICS.load_ccache_stats(json.dumps(dict.fromkeys(METRICS.UNCACHEABLE_REASONS, 0))),
             "all reported reasons: 0"),
        )
        for cache, expected in cases:
            for concise in (False, True):
                with self.subTest(cache=cache, concise=concise):
                    report = {
                        "status": "success",
                        "configuration": {"compiler": "fixture", "jobs": 1,
                                          "dependency_preparation_seconds": 0,
                                          "compiler_cache_key": "test", "dependency_cache_key": "test"},
                        "phases": [{"name": "configure", "elapsed_seconds": 0, "ccache": cache}],
                    }
                    rendered = METRICS.render_markdown(report, concise)
                    self.assertIn(expected, rendered)
                    if expected != "all reported reasons: 0":
                        self.assertNotIn("all reported reasons: 0", rendered)

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

    def test_system_policy_normalization_matches_cmake(self) -> None:
        for global_policy, override, resolution, expected in (
            ("off", None, "source", "OFF"),
            ("On", "", "system", "ON"),
            ("off", "oN", "system", "ON"),
            ("ON", "oFf", "source", "OFF"),
            ("aUtO", None, "source", "AUTO"),
        ):
            with self.subTest(global_policy=global_policy, override=override):
                cache = {"VSAG_USE_SYSTEM_DEPS": global_policy}
                if override is not None:
                    cache["VSAG_USE_SYSTEM_FMT"] = override
                report = METRICS.build_dependency_report(
                    {"fmt": {"resolution": resolution}}, {}, ROOT / "build", None, cache
                )["fmt"]
                self.assertEqual(report["system_policy"], expected)
                if expected == "ON":
                    self.assertEqual(report["fallback"], "none; system dependency required")
                elif expected == "OFF":
                    self.assertEqual(
                        report["fallback"],
                        "system dependency disabled; source hash validation required",
                    )

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
                            self.assertIn("unverified", summary)
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
                    self.assertEqual(report["pybind11"]["local_bytes"], 29)

    def test_external_project_local_bytes_include_in_source_builds_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_dir = Path(directory)
            for dependency, artifact in (
                ("hdf5", "source/src/hdf5.o"),
                ("openblas", "source/kernel/kernel.o"),
                ("antlr4", "source/runtime/Cpp/CMakeFiles/antlr4.o"),
            ):
                with self.subTest(dependency=dependency):
                    dependency_dir = build_dir / dependency
                    files = {
                        "source/README": b"source",
                        artifact: b"object bytes",
                        "install/lib/library.a": b"installed library",
                        "src/stamp/build": b"stamp",
                    }
                    for relative, content in files.items():
                        path = dependency_dir / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(content)
                    shared = build_dir / ".vsag-build-info"
                    shared.mkdir(exist_ok=True)
                    (shared / f"{dependency}-cfgcmd.txt").write_bytes(b"shared metadata")
                    details = METRICS.build_dependency_report(
                        {dependency: {"resolution": "source"}}, {}, build_dir, None, {}
                    )[dependency]

                    self.assertEqual(details["local_bytes"], 40)
                    storage = METRICS.dependency_storage(build_dir, None, None)
                    self.assertEqual(storage["external_build_bytes"],
                                     sum(METRICS.directory_size(build_dir / name)
                                         for name in ("hdf5", "openblas", "antlr4")))
                    self.assertNotIn("source_bytes", details)
                    self.assertNotIn("build_bytes", details)

                    # A system selection must not count a leftover source build.
                    system = METRICS.build_dependency_report(
                        {dependency: {"resolution": "system"}}, {}, build_dir, None, {}
                    )[dependency]
                    self.assertEqual(system["local_bytes"], 0)

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
                    "local_bytes": 3072,
                    "stages": {"build": {"edges": 1, "cumulative_seconds": 3.0}},
                },
                "openblas": {
                    "display_name": "OpenBLAS",
                    "resolution": "system",
                    "pin": "system",
                    "fallback": "bundled source fallback available",
                    "local_bytes": 0,
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
        self.assertIn("| 3.0 KiB |", summary)
        self.assertIn("in-source builds cannot be split reliably", summary)
        self.assertIn("Shared BUILD_INFO_DIR files", summary)
        self.assertIn("are excluded from both Local size", summary)
        self.assertNotIn("\\n", summary)
        self.assertTrue(summary.endswith("\n"))

    @staticmethod
    def write_build_fixture(root: Path) -> None:
        shutil.copyfile(ROOT / "Makefile", root / "Makefile")
        (root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.18)\n"
            "project(metrics_fixture CXX)\n"
            "set(CMAKE_CXX_COMPILER_LAUNCHER ${CMAKE_SOURCE_DIR}/record-command.py)\n"
            "set(CMAKE_CXX_LINKER_LAUNCHER ${CMAKE_SOURCE_DIR}/record-command.py)\n"
            "add_executable(fixture main.cpp)\n"
            "add_custom_command(OUTPUT artifact.txt\n"
            "  COMMAND ${CMAKE_COMMAND} -E touch artifact.txt VERBATIM)\n"
            "add_custom_target(artifact ALL DEPENDS artifact.txt)\n",
            encoding="utf-8",
        )

        (root / "main.cpp").write_text("int main() { return 0; }\n")
        launcher = root / "record-command.py"
        launcher.write_text(
            f"#!{os.sys.executable}\n"
            "import json, pathlib, subprocess, sys\n"
            "with (pathlib.Path(__file__).parent / 'commands.jsonl').open('a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "sys.exit(subprocess.call(sys.argv[1:]))\n"
        )
        launcher.chmod(0o755)

    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("make", "cmake", "ninja")),
        "requires Make, CMake, and Ninja",
    )
    def test_archive_storage_matches_cmake_default_and_ci_override(self) -> None:
        previous_directory = Path.cwd()
        for override in ("default", "environment", "cache"):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                self.write_build_fixture(root)
                shutil.copyfile(ROOT / "cmake/VSAGOptions.cmake", root / "VSAGOptions.cmake")
                with (root / "CMakeLists.txt").open("a", encoding="utf-8") as cmake:
                    cmake.write('include("${CMAKE_SOURCE_DIR}/VSAGOptions.cmake")\ninclude(FetchContent)\n')
                env = dict(os.environ)
                env.pop("VSAG_THIRDPARTY_DOWNLOAD_DIR", None)
                env.pop("EXTRA_DEFINED", None)
                expected = root / ("build/.vsag-downloads" if override == "default" else ".ci-downloads")
                env.pop("VSAG_FETCHCONTENT_BASE_DIR", None)
                expected_fetch = root / "build/_deps"
                if override == "environment":
                    env["VSAG_THIRDPARTY_DOWNLOAD_DIR"] = str(expected)
                elif override == "cache":
                    expected_fetch = root / "configured-deps"
                    env["VSAG_THIRDPARTY_DOWNLOAD_DIR"] = str(root / "unused-archives")
                    env["EXTRA_DEFINED"] = (
                        f"-DDOWNLOAD_DIR={expected} -DFETCHCONTENT_BASE_DIR={expected_fetch}"
                    )
                (expected_fetch / "fmt-src").mkdir(parents=True)
                (expected_fetch / "fmt-src/source").write_bytes(b"source bytes")
                expected.mkdir(parents=True)
                (expected / "restored.tar.gz").write_bytes(b"restored archive")
                try:
                    os.chdir(root)
                    with patch.dict(os.environ, env, clear=True):
                        collector = METRICS.Collector(self.collector_args())
                        result = collector.run_logged(
                            "configure", ["make", "configure-asan", "COMPILE_JOBS=2",
                                          "CMAKE_GENERATOR=Ninja"],
                        )
                        self.assertEqual(collector.failure_code, 0, result)
                        cache = METRICS.read_cmake_cache(root / "build/CMakeCache.txt")
                        self.assertEqual(Path(cache["DOWNLOAD_DIR"]), expected)
                        (Path(cache["DOWNLOAD_DIR"]) / "downloaded.tar.gz").write_bytes(
                            b"downloaded archive"
                        )
                        self.assertEqual(Path(cache["FETCHCONTENT_BASE_DIR"]), expected_fetch)
                        if override != "default":
                            unused = root / "build/.vsag-downloads"
                            unused.mkdir()
                            (unused / "unrelated").write_bytes(b"must not be counted")
                        collector.write_reports()
                        report = json.loads((root / "metrics/build-metrics.json").read_text())
                        self.assertEqual(
                            report["configuration"]["dependency_storage"]["fetchcontent_bytes"],
                            len(b"source bytes"),
                        )
                        self.assertEqual(
                            report["configuration"]["dependency_storage"]["archive_bytes"],
                            len(b"restored archive") + len(b"downloaded archive"),
                        )
                finally:
                    os.chdir(previous_directory)

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
        all(shutil.which(tool) for tool in ("make", "cmake")),
        "requires Make and CMake",
    )
    def test_make_test_runs_binaries_from_configured_build_directory(self) -> None:
        for build_dir in (None, "custom-build", "relative build", "absolute build"):
            with self.subTest(build_dir=build_dir), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                self.write_build_fixture(root)
                binary = root / "test-fixture.sh"
                binary.write_text(
                    '#!/bin/sh\n'
                    'printf "%s\\n" "$0" "$@" >> executed-tests.txt\n',
                    encoding="utf-8",
                )
                binary.chmod(0o755)
                with (root / "CMakeLists.txt").open("a", encoding="utf-8") as cmake:
                    cmake.write(
                        'foreach(name unittests functests eval_monitor_test)\n'
                        '  configure_file(test-fixture.sh "${CMAKE_BINARY_DIR}/tests/${name}" COPYONLY)\n'
                        'endforeach()\n'
                    )
                configured = root / (build_dir or "build")
                override = [] if build_dir is None else [
                    f"DEBUG_BUILD_DIR={configured if build_dir == 'absolute build' else build_dir}"
                ]
                result = subprocess.run(
                    ["make", "test", "CMAKE_GENERATOR=Unix Makefiles", "COMPILE_JOBS=2",
                     'CASE="[fixture]"', "SHARD=--shard-count 2 --shard-index 1", *override],
                    cwd=root, text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((configured / "artifact.txt").is_file())
                lines = (root / "executed-tests.txt").read_text(encoding="utf-8").splitlines()
                arguments = ["-d", "yes", "[fixture]", "--allow-running-no-tests",
                             "--shard-count", "2", "--shard-index", "1"]
                self.assertEqual(len(lines), 3 * (1 + len(arguments)))
                for index, name in enumerate(("unittests", "functests", "eval_monitor_test")):
                    start = index * (1 + len(arguments))
                    self.assertEqual((root / lines[start]).resolve(), configured / "tests" / name)
                    self.assertEqual(lines[start + 1:start + 1 + len(arguments)], arguments)

    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("make", "cmake")),
        "requires Make and CMake",
    )
    def test_make_test_module_runs_binary_from_configured_build_directory(self) -> None:
        for build_dir in (None, "custom-build", "relative build", "absolute build"):
            with self.subTest(build_dir=build_dir), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                self.write_build_fixture(root)
                binary = root / "test-fixture.sh"
                binary.write_text(
                    '#!/bin/sh\n'
                    'printf "%s\\n" "$0" "$@" >> executed-tests.txt\n',
                    encoding="utf-8",
                )
                binary.chmod(0o755)
                with (root / "CMakeLists.txt").open("a", encoding="utf-8") as cmake:
                    cmake.write(
                        'foreach(name unittests_datacell)\n'
                        '  configure_file(test-fixture.sh "${CMAKE_BINARY_DIR}/tests/${name}" COPYONLY)\n'
                        'endforeach()\n'
                        'add_custom_target(unittests_datacell DEPENDS artifact)\n'
                    )
                configured = root / (build_dir or "build")
                override = [] if build_dir is None else [
                    f"DEBUG_BUILD_DIR={configured if build_dir == 'absolute build' else build_dir}"
                ]
                result = subprocess.run(
                    ["make", "test-module", "MODULE=datacell", "CMAKE_GENERATOR=Unix Makefiles", "COMPILE_JOBS=2",
                     'CASE="[fixture]"', "SHARD=--shard-count 2 --shard-index 1", *override],
                    cwd=root, text=True, capture_output=True, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((configured / "artifact.txt").is_file())
                lines = (root / "executed-tests.txt").read_text(encoding="utf-8").splitlines()
                arguments = ["-d", "yes", "[fixture]", "--allow-running-no-tests",
                             "--shard-count", "2", "--shard-index", "1"]
                self.assertEqual(len(lines), 1 + len(arguments))
                for index, name in enumerate(("unittests_datacell",)):
                    start = index * (1 + len(arguments))
                    self.assertEqual((root / lines[start]).resolve(), configured / "tests" / name)
                    self.assertEqual(lines[start + 1:start + 1 + len(arguments)], arguments)

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
                capture = collector.capture_build
                observed = {}

                def capture_with_command_evidence(name, mode):
                    command_log = Path(directory) / "commands.jsonl"
                    before = command_log.read_text() if command_log.exists() else ""
                    capture(name, mode)
                    after = command_log.read_text()
                    observed[name] = after[len(before):].splitlines()

                collector.capture_build = capture_with_command_evidence
                with patch.dict(os.environ, {"CMAKE_GENERATOR": "Ninja"}):
                    self.assertEqual(collector.collect(), 0)
                self.assertEqual(len(observed["clean_build"]), 2)
                self.assertEqual(len(observed["warm_cache_build"]), 2)
                self.assertEqual(observed["noop_build"], [])
                self.assertEqual(collector.phases[-1]["build_edges"], 0)
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
                self.assertIn("CMAKE_BUILD_ARGS=-d stats", collector.phases[-1]["command"])
                self.assertTrue(collector.phases[-1]["true_noop"])
            finally:
                os.chdir(previous_directory)

    @unittest.skipUnless(
        all(shutil.which(tool) for tool in ("make", "cmake")),
        "requires Make and CMake",
    )
    def test_collection_supports_unix_makefiles_fallback(self) -> None:
        previous_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                self.write_build_fixture(Path(directory))
                # Simulate an unavailable Ninja without hiding Make/CMake.
                bindir = Path(directory) / "bin"
                bindir.mkdir()
                ninja = bindir / "ninja"
                ninja.write_text("#!/bin/sh\nexit 1\n")
                ninja.chmod(0o755)
                args = self.collector_args()
                args.build_dir = "custom build"
                collector = METRICS.Collector(args)
                collector.ccache = lambda *arguments: {}
                collector.write_reports = lambda: None
                environment = dict(os.environ)
                environment.pop("CMAKE_GENERATOR", None)
                environment["PATH"] = str(bindir) + os.pathsep + environment["PATH"]
                with patch.dict(os.environ, environment, clear=True):
                    self.assertEqual(collector.collect(), 0)
                self.assertEqual(METRICS.read_cmake_cache(
                    collector.build_dir / "CMakeCache.txt")["CMAKE_GENERATOR"], "Unix Makefiles")
                self.assertTrue((collector.build_dir / "artifact.txt").is_file())
                for phase in collector.phases:
                    self.assertEqual(phase["exit_code"], 0)
                    if phase["name"] in ("clean_build", "warm_cache_build", "noop_build"):
                        self.assertIn("CMAKE_BUILD_ARGS=", phase["command"])
                        self.assertFalse(phase["ninja"]["available"])
                self.assertFalse(collector.phases[-1]["true_noop"])
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
