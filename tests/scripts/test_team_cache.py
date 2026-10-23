"""Filesystem regression tests for immutable cache validation and private restores."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ci/team_cache.py"
SPEC = importlib.util.spec_from_file_location("team_cache", SCRIPT)
cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache)


class TeamCacheTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.source = self.root / "snapshot"
        (self.source / "a/b").mkdir(parents=True)
        (self.source / "a/b/object").write_bytes(b"compiler output")
        (self.source / "ccache.conf").write_text("compression = true\n")
        self.manifest = self.root / "manifest.json"
        self.destination = self.root / "private"
        self.namespace = dict(image="ghcr.io/antgroup/vsag@sha256:" + "a" * 64,
                              compiler="gcc-11.4", arch="x86_64", sanitizer="asan",
                              build_options="system-openblas;examples;tools",
                              path_scheme="ccache-base-dir-v1")
        cache.generate(self.source, self.manifest, self.namespace, "1" * 40)

    def mutate_manifest(self, mutation):
        value = json.loads(self.manifest.read_text())
        mutation(value)
        self.manifest.write_text(json.dumps(value))

    def restore(self):
        return cache.restore(self.source, self.manifest, self.destination, self.namespace)

    def rejected(self):
        with self.assertRaises((ValueError, OSError)):
            self.restore()
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.root.glob(".team-cache-*")), [])

    def test_readonly_source_and_writable_independent_copy(self):
        before = cache.inventory(self.source)
        for path in self.source.rglob("*"):
            path.chmod(0o555 if path.is_dir() else 0o444)
        self.source.chmod(0o555)
        try:
            result = self.restore()
            (self.destination / "a/b/object").write_bytes(b"PR private mutation")
            self.assertEqual(cache.inventory(self.source), before)
            self.assertEqual(result["files"], 2)
            self.assertGreaterEqual(result["total_seconds"], result["copy_seconds"])
            self.assertGreaterEqual(result["copy_seconds"], 0)
        finally:
            self.source.chmod(0o755)
            for path in self.source.rglob("*"):
                path.chmod(0o755 if path.is_dir() else 0o644)

    def test_namespace_not_publisher_sha(self):
        original = cache.namespace_id(self.namespace)
        self.mutate_manifest(lambda m: m.update(publisher_main_sha="2" * 40))
        self.assertEqual(self.restore()["namespace_id"], original)
        for key in self.namespace:
            changed = dict(self.namespace)
            changed[key] += "different"
            if key in ("sanitizer", "image"):
                with self.assertRaises(ValueError):
                    cache.namespace_id(changed)
            else:
                self.assertNotEqual(cache.namespace_id(changed), original)

    def test_pr_sha_is_not_a_namespace_field(self):
        with self.assertRaises(ValueError):
            cache.namespace_id(dict(self.namespace, pr_sha="a" * 40))

    def test_wrong_namespace(self):
        self.namespace["compiler"] = "gcc-12"
        self.rejected()

    def test_missing_manifest(self):
        self.manifest.unlink()
        self.rejected()

    def test_invalid_manifest(self):
        self.manifest.write_text("{partial")
        self.rejected()

    def test_missing_cached_file(self):
        (self.source / "a/b/object").unlink()
        self.rejected()

    def test_extra_cached_file(self):
        (self.source / "unexpected").write_bytes(b"x")
        self.rejected()

    def test_same_size_corruption(self):
        (self.source / "a/b/object").write_bytes(b"COMPILER OUTPUT")
        self.rejected()

    def test_truncated_file(self):
        (self.source / "a/b/object").write_bytes(b"x")
        self.rejected()

    def test_path_traversal_and_noncanonical_paths(self):
        original = self.manifest.read_text()
        for name in ("../escape", "/absolute", "a/../../escape", "a//b", "a/./b", "a\\b", "."):
            with self.subTest(name=name):
                self.manifest.write_text(original)
                self.mutate_manifest(lambda m: m["files"][0].update(path=name))
                self.rejected()

    def test_duplicate_manifest_entries(self):
        self.mutate_manifest(lambda m: m["files"].append(copy.deepcopy(m["files"][0])))
        self.rejected()

    def test_source_file_symlink(self):
        target = self.source / "a/b/object"
        target.unlink()
        target.symlink_to(self.source / "ccache.conf")
        self.rejected()

    def test_source_directory_symlink(self):
        (self.source / "alias").symlink_to(self.source / "a", target_is_directory=True)
        self.rejected()

    def test_symlink_source_parent(self):
        alias = self.root / "alias"
        alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            cache.restore(alias, self.manifest, self.destination, self.namespace)

    def test_manifest_symlink(self):
        alias = self.root / "manifest-alias"
        alias.symlink_to(self.manifest)
        with self.assertRaises(ValueError):
            cache.restore(self.source, alias, self.destination, self.namespace)

    def test_source_destination_overlap(self):
        for destination in (self.source, self.source / "private", self.root):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                cache.restore(self.source, self.manifest, destination, self.namespace)

    def test_existing_destination_preserved(self):
        self.destination.mkdir()
        marker = self.destination / "mine"
        marker.write_text("keep")
        with self.assertRaises(ValueError):
            self.restore()
        self.assertEqual(marker.read_text(), "keep")

    def test_mid_copy_failure_leaves_no_partial_cache(self):
        with mock.patch.object(cache.shutil, "copyfileobj", side_effect=OSError("disk full")):
            self.rejected()

    def test_copy_corruption_leaves_no_partial_cache(self):
        with mock.patch.object(cache.shutil, "copyfileobj", side_effect=lambda src, dst, length: dst.write(b"bad")):
            self.rejected()

    def test_manifest_generation_never_writes_source(self):
        before = cache.inventory(self.source)
        with self.assertRaises(ValueError):
            cache.generate(self.source, self.source / "manifest", self.namespace)
        with self.assertRaises(ValueError):
            cache.generate(self.source, self.manifest, self.namespace)
        self.assertEqual(cache.inventory(self.source), before)

    def test_source_change_mid_copy_rejected(self):
        original_copy = cache.shutil.copyfileobj

        def change_source(src, dst, length):
            original_copy(src, dst, length)
            (self.source / "unexpected").write_bytes(b"new file")

        with mock.patch.object(cache.shutil, "copyfileobj", side_effect=change_source):
            self.rejected()

    def test_destination_symlink_rejected(self):
        self.destination.symlink_to(self.root / "elsewhere", target_is_directory=True)
        with self.assertRaises(ValueError):
            self.restore()
        self.assertTrue(self.destination.is_symlink())
        self.assertFalse((self.root / "elsewhere").exists())

    def test_empty_manifest_inventory_rejected(self):
        self.mutate_manifest(lambda m: m.update(files=[]))
        self.rejected()

    def test_cli_manifest_verify_restore(self):
        namespace = self.root / "namespace.json"
        namespace.write_text(json.dumps(self.namespace))
        self.manifest.unlink()
        for command in ("manifest", "verify", "restore"):
            args = [sys.executable, str(SCRIPT), command, "--source", str(self.source),
                    "--manifest", str(self.manifest), "--namespace-file", str(namespace)]
            if command == "restore":
                args.extend(["--destination", str(self.destination)])
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(result.stdout)["ok"])
        self.assertEqual(cache.inventory(self.source), cache.inventory(self.destination))

    def test_cli_failure_explicit_nonzero_json(self):
        namespace = self.root / "namespace.json"
        namespace.write_text(json.dumps(self.namespace))
        self.manifest.unlink()
        result = subprocess.run([sys.executable, str(SCRIPT), "restore", "--source", str(self.source),
                                 "--manifest", str(self.manifest), "--destination", str(self.destination),
                                 "--namespace-file", str(namespace)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)["ok"])
        self.assertFalse(self.destination.exists())


class TestHistoryTest(unittest.TestCase):
    mutate_manifest = TeamCacheTest.mutate_manifest
    restore = TeamCacheTest.restore
    def setUp(self):
        TeamCacheTest.setUp(self)
        self.history = self.root / "history.json"
        self.value = {"schema": 1, "source_main_sha": "1" * 40, "inventory_digest": "2" * 64,
                      "duration_history": {"a" * 64: 0.001, "b" * 64: 512.4}}
        self.history.write_text(json.dumps(self.value))

    def publish(self):
        self.manifest.unlink()
        return cache.generate(self.source, self.manifest, self.namespace, "1" * 40, self.history)

    def test_auxiliary_history_never_enters_private_compiler_cache(self):
        self.publish()
        self.assertEqual(cache.verify_history(self.history, self.manifest, self.namespace)["cases"], 2)
        self.restore()
        self.assertFalse((self.destination / "history.json").exists())
        self.assertEqual(cache.inventory(self.source), cache.inventory(self.destination))

    def test_missing_history_does_not_break_compiler_cache(self):
        self.publish()
        self.history.unlink()
        self.restore()
        with self.assertRaises(OSError):
            cache.verify_history(self.history, self.manifest, self.namespace)

    def test_corrupt_history_does_not_break_compiler_cache(self):
        self.publish()
        self.history.write_text(json.dumps(dict(self.value, duration_history={"a" * 64: 42})))
        self.restore()
        with self.assertRaisesRegex(ValueError, "digest|size"):
            cache.verify_history(self.history, self.manifest, self.namespace)

    def test_old_manifest_has_no_history_and_still_restores(self):
        with self.assertRaisesRegex(ValueError, "No published"):
            cache.verify_history(self.history, self.manifest, self.namespace)
        self.restore()

    def test_bad_history_provenance_rejected(self):
        self.value["source_main_sha"] = "3" * 40
        self.history.write_text(json.dumps(self.value))
        with self.assertRaisesRegex(ValueError, "publisher"):
            self.publish()

    def test_invalid_hint_values_and_schema(self):
        for value in (0, -1, True, "1", float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=str(value)[:30]):
                self.value["duration_history"]["a" * 64] = value
                self.history.write_text(json.dumps(self.value))
                with self.assertRaises(ValueError):
                    cache.read_history(self.history)
        for value in (dict(self.value, schema=True), dict(self.value, schema=2),
                      dict(self.value, duration_history={}), dict(self.value, unexpected=True)):
            self.history.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                cache.read_history(self.history)

    def test_duplicate_json_keys_rejected(self):
        self.history.write_text('{"schema":1,"schema":1}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            cache.read_history(self.history)

    def test_links_and_oversize_rejected(self):
        alias = self.root / "alias.json"
        alias.symlink_to(self.history)
        with self.assertRaises(ValueError):
            cache.read_history(alias)
        import os
        link = self.root / "hardlink.json"
        os.link(self.history, link)
        with self.assertRaises(ValueError):
            cache.read_history(self.history)
        link.unlink()
        self.history.write_bytes(b"x" * (cache.MAX_HISTORY_BYTES + 1))
        with self.assertRaises(ValueError):
            cache.read_history(self.history)

    def test_history_cannot_be_a_compiler_entry(self):
        path = self.source / "history.json"
        path.write_text(json.dumps(self.value))
        self.manifest.unlink()
        with self.assertRaisesRegex(ValueError, "outside"):
            cache.generate(self.source, self.manifest, self.namespace, "1" * 40, path)

    def test_cli_verifies_history_and_rejects_hash_tamper(self):
        self.publish()
        namespace = self.root / "namespace.json"
        namespace.write_text(json.dumps(self.namespace))
        args = [sys.executable, str(SCRIPT), "verify-history", "--history", str(self.history),
                "--manifest", str(self.manifest), "--namespace-file", str(namespace)]
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["cases"], 2)
        self.mutate_manifest(lambda m: m["auxiliary"]["test_history"].update(sha256="0" * 64))
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(json.loads(result.stdout)["ok"])


if __name__ == "__main__":
    unittest.main()
