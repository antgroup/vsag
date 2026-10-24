"""Run the compatibility shell contract against an observable mock executable."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CompatibilityInputsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="compatibility inputs ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = self.root / "fixtures with spaces"
        self.inputs.mkdir()
        self.build = self.root / "build"
        self.tool = self.build / "tools/check_compatibility/check_compatibility"
        self.tool.parent.mkdir(parents=True)
        self.log = self.root / "calls.jsonl"
        self.tool.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['MOCK_CALLS'], 'a') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "sys.exit(int(os.environ.get('MOCK_EXIT', '0')))\n"
        )
        self.tool.chmod(0o755)
        self.vectors = self.inputs / "random_512d_10K.bin"
        with self.vectors.open("wb") as stream:
            stream.truncate(20480000)
        self.fixture("v1.0.0_hgraph")

    def fixture(self, name):
        (self.inputs / (name + ".index")).write_bytes(b"mock index")
        for suffix in ("_build.json", "_search.json"):
            (self.inputs / (name + suffix)).write_text("{}\n")

    def run_script(self, **extra):
        env = dict(os.environ, COMPATIBILITY_INDEX_DIR=str(self.inputs),
                   BUILD_DIR=str(self.build), MOCK_CALLS=str(self.log), **extra)
        return subprocess.run(["bash", str(ROOT / "scripts/check_compatibility.sh")],
                              env=env, cwd=self.root, text=True, capture_output=True)

    def assert_rejected_before_execution(self):
        result = self.run_script()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.log.exists(), result.stdout + result.stderr)

    def test_explicit_directory_and_every_maintained_index(self):
        self.fixture("v0.9.0_pyramid")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(calls, [["v0.9.0_pyramid", str(self.inputs)],
                                 ["v1.0.0_hgraph", str(self.inputs)]])

    def test_missing_sidecar_rejects_entire_set_before_execution(self):
        self.fixture("v9.0.0_hgraph")
        (self.inputs / "v9.0.0_hgraph_search.json").unlink()
        self.assert_rejected_before_execution()

    def test_release_orphan_pair_reported_without_inventing_a_case(self):
        # compatibility-indexes contains these parameters but no v0.12.0 index.
        for suffix in ("_build.json", "_search.json"):
            (self.inputs / ("v0.12.0_hgraph" + suffix)).write_text("{}")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Orphan compatibility parameters without a published index: "
                      "v0.12.0_hgraph", result.stderr)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(calls, [["v1.0.0_hgraph", str(self.inputs)]])

    def test_single_orphan_sidecar_does_not_create_a_case(self):
        (self.inputs / "v9.0.0_hgraph_build.json").write_text("{}")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("v9.0.0_hgraph", result.stderr)
        self.assertEqual(len(self.log.read_text().splitlines()), 1)

    def test_orphan_sidecars_only_cannot_pass(self):
        (self.inputs / "v1.0.0_hgraph.index").unlink()
        self.assert_rejected_before_execution()

    def test_empty_index_rejected(self):
        (self.inputs / "v1.0.0_hgraph.index").write_bytes(b"")
        self.assert_rejected_before_execution()

    def test_invalid_json_rejected(self):
        (self.inputs / "v1.0.0_hgraph_search.json").write_text("{broken")
        self.assert_rejected_before_execution()

    def test_non_object_json_rejected(self):
        (self.inputs / "v1.0.0_hgraph_build.json").write_text("[]")
        self.assert_rejected_before_execution()

    def test_missing_vectors_rejected(self):
        self.vectors.unlink()
        self.assert_rejected_before_execution()

    def test_truncated_vectors_rejected_even_if_500_queries_remain(self):
        with self.vectors.open("wb") as stream:
            stream.truncate(500 * 512 * 4)
        self.assert_rejected_before_execution()

    def test_oversized_vectors_rejected(self):
        with self.vectors.open("ab") as stream:
            stream.write(b"x")
        self.assert_rejected_before_execution()

    def test_removed_indexes_are_still_skipped(self):
        for name in ("v0.5_hnsw", "v0.5_fresh_hnsw", "v0.5_diskann", "v0.5_HNSW"):
            (self.inputs / (name + ".index")).write_bytes(b"unused")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(self.log.read_text().splitlines()), 1)

    def test_removed_only_cannot_pass(self):
        for path in self.inputs.glob("v1*"):
            path.unlink()
        self.fixture("v0.5_hnsw")
        self.assert_rejected_before_execution()

    def test_empty_directory_cannot_pass(self):
        for path in self.inputs.iterdir():
            path.unlink()
        self.assert_rejected_before_execution()

    def test_failed_tool_stops_without_success(self):
        self.fixture("v9.0.0_hgraph")
        result = self.run_script(MOCK_EXIT="2")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("All compatibility checks passed", result.stdout)
        self.assertEqual(len(self.log.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
