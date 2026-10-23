"""Read-only compatibility release manifest verification contracts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ci/verify_compatibility_cache.py"
SPEC = importlib.util.spec_from_file_location("compatibility_cache", SCRIPT)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class CompatibilityCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve()
        self.manifest = self.directory / "manifest.json"
        self.document = dict(schema_version=1, repository="antgroup/vsag",
                             release_tag="compatibility-indexes", release_id=123, assets=[])
        for index, name in enumerate(("v1_hgraph.index", "v1_hgraph_build.json",
                                      "v1_hgraph_search.json", "random_512d_10K.bin"), 1):
            content = b"fixture " + name.encode()
            (self.directory / name).write_bytes(content)
            self.document["assets"].append(dict(name=name, id=index, size=len(content),
                                                 sha256=hashlib.sha256(content).hexdigest()))
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.document))

    def rejected(self):
        with self.assertRaises((ValueError, OSError, TypeError)):
            cache.verify(self.directory, self.manifest)

    def test_complete_readonly_snapshot(self):
        before = {path.name: path.read_bytes() for path in self.directory.iterdir()}
        for path in self.directory.iterdir():
            path.chmod(0o444)
        self.directory.chmod(0o555)
        try:
            result = cache.verify(self.directory, self.manifest)
            self.assertEqual(result["assets"], 4)
            self.assertEqual(result["release_id"], 123)
            self.assertGreaterEqual(result["verify_seconds"], 0)
            self.assertEqual(before, {path.name: path.read_bytes() for path in self.directory.iterdir()})
        finally:
            self.directory.chmod(0o755)
            for path in self.directory.iterdir():
                path.chmod(0o644)

    def test_missing_whole_triplet(self):
        for path in self.directory.glob("v1_*"):
            path.unlink()
        self.rejected()

    def test_missing_file(self):
        (self.directory / "v1_hgraph_build.json").unlink()
        self.rejected()

    def test_extra_file(self):
        (self.directory / "leftover").write_bytes(b"x")
        self.rejected()

    def test_same_size_corruption(self):
        path = self.directory / "v1_hgraph.index"
        path.write_bytes(b"x" * path.stat().st_size)
        self.rejected()

    def test_truncated_file(self):
        (self.directory / "v1_hgraph.index").write_bytes(b"x")
        self.rejected()

    def test_missing_manifest(self):
        self.manifest.unlink()
        self.rejected()

    def test_empty_assets(self):
        self.document["assets"] = []
        self.save()
        self.rejected()

    def test_duplicate_asset(self):
        self.document["assets"].append(dict(self.document["assets"][0]))
        self.save()
        self.rejected()

    def test_duplicate_id(self):
        self.document["assets"][1]["id"] = 1
        self.save()
        self.rejected()

    def test_invalid_asset_names(self):
        for name in ("../escape", "/absolute", "a/b", "a\\b", "manifest.json", ".", "", "x\0y"):
            with self.subTest(name=name):
                self.document["assets"][0]["name"] = name
                self.save()
                self.rejected()

    def test_symlink_asset(self):
        path = self.directory / "v1_hgraph.index"
        path.unlink()
        path.symlink_to(self.directory / "v1_hgraph_build.json")
        self.rejected()

    def test_symlink_manifest(self):
        self.manifest.unlink()
        self.manifest.symlink_to(self.directory / "v1_hgraph_build.json")
        self.rejected()

    def test_directory_asset(self):
        path = self.directory / "v1_hgraph.index"
        path.unlink()
        path.mkdir()
        self.rejected()

    def test_metadata_types_and_identity(self):
        for key, value in (("schema_version", True), ("schema_version", 2), ("repository", "fork/vsag"),
                           ("release_tag", "latest"), ("release_id", 0), ("release_id", True)):
            with self.subTest(key=key, value=value):
                old = self.document[key]
                self.document[key] = value
                self.save()
                self.rejected()
                self.document[key] = old

    def test_asset_numeric_and_digest_validation(self):
        for key, value in (("id", 0), ("id", True), ("size", -1), ("size", False), ("sha256", "bad")):
            with self.subTest(key=key, value=value):
                old = self.document["assets"][0][key]
                self.document["assets"][0][key] = value
                self.save()
                self.rejected()
                self.document["assets"][0][key] = old

    def test_duplicate_json_keys(self):
        self.manifest.write_text('{"assets": [], "assets": []}')
        self.rejected()

    def test_cli_success_and_failure(self):
        command = [sys.executable, str(SCRIPT), "--directory", str(self.directory),
                   "--manifest", str(self.manifest)]
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["ok"])
        (self.directory / "v1_hgraph.index").unlink()
        result = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
