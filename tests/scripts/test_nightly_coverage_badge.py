import json
from pathlib import Path
import tempfile
import unittest

from scripts.testing.nightly_coverage_badge import write_badge


class NightlyCoverageBadgeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.run_path = root / "run.json"
        self.report_path = root / "job-report.json"
        self.output = root / "badge"
        self.output.mkdir()
        sha = "c1820f122d004a9ae829d6db40029494af166e59"
        self.run = {
            "id": 45067476925,
            "run_number": 514,
            "run_attempt": 1,
            "head_sha": sha,
            "created_at": "2026-11-22T20:37:25Z",
            "conclusion": "failure",
        }
        self.report = {
            "schema_version": 1,
            "job": "full-coverage-coverage",
            "source_sha": sha,
            "run_id": str(self.run["id"]),
            "attempt": "1",
            "status_before_upload": "success",
            "coverage": {"available": True, "lines_hit": 60872, "lines_found": 70464},
        }

    def publish(self):
        self.run_path.write_text(json.dumps(self.run))
        self.report_path.write_text(json.dumps(self.report))
        return write_badge(self.run_path, self.report_path, self.output)

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.output.iterdir()}

    def test_uses_lcov_lines_from_successful_coverage_in_failed_nightly(self):
        self.assertTrue(self.publish())
        badge = json.loads((self.output / "badge.json").read_text())
        self.assertEqual(badge["schemaVersion"], 1)
        self.assertEqual(badge["label"], "Nightly C++ line coverage")
        self.assertEqual(badge["message"], "86.39%")
        landing = (self.output / "README.md").read_text()
        self.assertIn("60,872 / 70,464 lines covered by LCOV", landing)
        self.assertIn("2026-11-23 (Beijing time)", landing)
        self.assertIn(self.run["head_sha"], landing)
        self.assertIn("https://github.com/antgroup/vsag/actions/runs/45067476925/attempts/1", landing)

    def test_repeat_and_older_run_cannot_replace_newer_coverage(self):
        self.assertTrue(self.publish())
        original = self.snapshot()

        for run_number, attempt in ((514, 1), (513, 4)):
            with self.subTest(run_number=run_number, attempt=attempt):
                self.run.update(run_number=run_number, run_attempt=attempt)
                self.report["attempt"] = str(attempt)
                self.report["coverage"]["lines_hit"] = 100
                self.assertFalse(self.publish())
                self.assertEqual(self.snapshot(), original)

    def test_new_attempt_can_replace_previous_coverage(self):
        self.assertTrue(self.publish())
        self.run["run_attempt"] = 2
        self.report["attempt"] = "2"
        self.report["coverage"]["lines_hit"] = 70464
        self.assertTrue(self.publish())
        self.assertEqual(json.loads((self.output / "badge.json").read_text())["message"], "100.00%")

    def test_mismatched_or_unsuccessful_reports_preserve_the_badge(self):
        self.assertTrue(self.publish())
        original = self.snapshot()
        self.run.update(id=45099999999, run_number=515, run_attempt=2)
        matching = dict(self.report, run_id="45099999999", attempt="2")
        for key, value in (
            ("source_sha", "0" * 40),
            ("run_id", "45067476925"),
            ("attempt", "1"),
            ("job", "summary"),
            ("status_before_upload", "failure"),
        ):
            with self.subTest(key=key):
                self.report = dict(matching, **{key: value})
                with self.assertRaisesRegex(ValueError, "does not match"):
                    self.publish()
                self.assertEqual(self.snapshot(), original)

    def test_invalid_coverage_does_not_publish(self):
        for coverage in (
            {"available": False, "lines_hit": 1, "lines_found": 1},
            {"available": True, "lines_hit": 0, "lines_found": 0},
            {"available": True, "lines_hit": 2, "lines_found": 1},
            {"available": True, "lines_hit": True, "lines_found": 1},
        ):
            with self.subTest(coverage=coverage):
                self.report["coverage"] = coverage
                with self.assertRaisesRegex(ValueError, "invalid LCOV"):
                    self.publish()
                self.assertEqual(self.snapshot(), {})

    def test_missing_report_does_not_publish(self):
        self.run_path.write_text(json.dumps(self.run))
        with self.assertRaises(FileNotFoundError):
            write_badge(self.run_path, self.report_path, self.output)
        self.assertEqual(self.snapshot(), {})
