"""Source preflight and the real Pyramid regression that blocked all Daily lanes."""
import itertools
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/testing"))
import daily_parameters as parameters
import daily_plan as dp


class DailyParameterPreflightTests(unittest.TestCase):
    def fixture(self, root, source):
        for relative in parameters.PARAMETER_SOURCES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def test_repository_preflight_and_real_regression_cartesian_product(self):
        cases = parameters.preflight(ROOT)
        name = "Pyramid adaptive PiPNN build search Add and restore"
        case = next(c for c in cases if c["name"] == name)
        self.assertEqual(case["configurations"], 12)
        source = (ROOT / case["source_file"]).read_text()
        axes, choices = parameters.configurations(source, case["source_line"])
        self.assertEqual([a["choices"] for a in axes], [2, 2, 3])
        self.assertEqual([a["expression"] for a in axes],
                         ['std::string("single_layer"), std::string("multi_layer")',
                          '1, 4', '0, 1, 2'])
        observed = [tuple(c[a["line"]] for a in axes) for c in choices]
        self.assertEqual(observed, list(itertools.product(range(2), range(2), range(3))))
        # Reintroduce the exact defect: preflight must reject it before a build.
        original = source.replace('const auto threads = DAILY_GENERATE(1, 4);',
                                  'const auto threads = GENERATE(1, 4);')
        self.assertNotEqual(original, source)
        with self.assertRaisesRegex(dp.PlanError, name + r".*Unadapted GENERATE at line"):
            parameters.configurations(original, case["source_line"],
                                      source_file=case["source_file"], case_name=name)
        units = [{"key": str(i), "owner": "pyramid", "mode": "rotation"} for i in range(12)]
        ctx = dp.context("a" * 40, "2026-11-06T08:40:39Z", "2026-11-06")
        plan = dp.build_plan(ctx, "lane/Linux/x86_64", units[:1], units[1:])
        rotated = [u["key"] for day in plan["rotation"] for u in day]
        self.assertEqual(sorted(rotated), sorted(u["key"] for u in units[1:]))
        self.assertEqual(len(set(rotated)), 11)
        selected = dp.sealed({"schema_version": 1, "selected": units})
        shards = [dp.shard_plan(selected, i, 4) for i in range(4)]
        self.assertEqual(sorted(u["key"] for p in shards for u in p["selected"]),
                         sorted(u["key"] for u in units))

    def test_new_unadapted_case_is_rejected_with_filename_name_and_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = 'TEST_CASE("new case", "[pyramid]") {\n auto x = GENERATE(1, 2);\n}\n'
            self.fixture(root, source)
            with self.assertRaises(dp.PlanError) as caught:
                parameters.preflight(root)
            for relative in parameters.PARAMETER_SOURCES:
                self.assertIn(relative + ':1: new case: Unadapted GENERATE at line 2', str(caught.exception))
            proc = subprocess.run([sys.executable, str(ROOT / 'scripts/testing/daily_parameters.py'),
                                   '--root', str(root)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn('new case', proc.stderr)

    def test_multiline_fixture_and_strings_do_not_corrupt_case_boundaries(self):
        source = ('// TEST_CASE("fake", "") { GENERATE(1); }\n'
                  'TEST_CASE_PERSISTENT_FIXTURE(Fixture,\n "first", "[sindi]") {\n'
                  ' const char* s = R"tag({ GENERATE(1) })tag";\n'
                  ' auto x = DAILY_GENERATE(1,\n 2);\n}\n'
                  'TEST_CASE("second", "[sindi]") { auto x = DAILY_GENERATE(false, true); }\n')
        self.assertEqual([c['name'] for c in parameters.source_cases(source)], ['first', 'second'])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, source)
            self.assertEqual([c['configurations'] for c in parameters.preflight(root)], [2, 2, 2, 2])
        for line in (2, 3):  # GCC opening and Clang closing declaration locations.
            axes, choices = parameters.configurations(source, line)
            self.assertEqual(len(choices), 2)
            self.assertEqual(parameters.runtime_choices(axes, choices[1]), {'5': 1, '6': 1})

    def test_empty_adapter_arguments_fail_preflight_with_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for expression in ("", "/* comment */", ",", ",1", "1,", "1,,2"):
                self.fixture(root, 'TEST_CASE("empty axis", "[pyramid]") {\n'
                             ' auto x = DAILY_GENERATE(' + expression + ');\n}\n')
                with self.assertRaises(dp.PlanError) as caught:
                    parameters.preflight(root)
                self.assertIn('tests/test_pyramid.cpp:1: empty axis: Empty DAILY_GENERATE value at line 2',
                              str(caught.exception))
            self.fixture(root, 'TEST_CASE("nested", "[pyramid]") { auto x = '
                         'DAILY_GENERATE(std::string("a,b"), make(1, 2)); }')
            self.assertEqual([x['configurations'] for x in parameters.preflight(root)], [2, 2])

    def test_invalid_names_and_encoding_report_clean_cli_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # Valid C++ escape, outside this deliberately JSON-compatible name parser.
            self.fixture(root, r'TEST_CASE("bad\x41", "[pyramid]") {}')
            proc = subprocess.run([sys.executable, str(ROOT / 'scripts/testing/daily_parameters.py'),
                                   '--root', str(root)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn('tests/test_pyramid.cpp: Invalid literal test name at line 1', proc.stderr)
            self.assertNotIn('Traceback', proc.stderr)
            (root / parameters.PARAMETER_SOURCES[0]).write_bytes(b'\xff')
            proc = subprocess.run([sys.executable, str(ROOT / 'scripts/testing/daily_parameters.py'),
                                   '--root', str(root)], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 1)
            self.assertIn('tests/test_pyramid.cpp:', proc.stderr)
            self.assertNotIn('Traceback', proc.stderr)

    def test_missing_empty_unsupported_and_nonfinite_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(dp.PlanError): parameters.preflight(root)
            for source in ('', 'TEST_CASE', 'TEST_CASE_METHOD(F, "x", "") {}',
                           'TEST_CASE("x", "") { auto x = DAILY_GENERATE(range(1, 5)); }'):
                self.fixture(root, source)
                with self.assertRaises(dp.PlanError): parameters.preflight(root)


if __name__ == "__main__":
    unittest.main()
