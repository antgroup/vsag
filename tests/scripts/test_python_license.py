#!/usr/bin/env python3

from pathlib import Path
import configparser
import os
import runpy
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


class PythonLicenseTest(unittest.TestCase):
    def metadata(self, platform):
        # Execute setup.py's platform selection without setuptools or a native build.
        captured = {}
        modules = {}
        for name, attribute in (("setuptools", "Extension"),
                                ("setuptools.command.build_ext", "build_ext"),
                                ("setuptools.command.bdist_wheel", "bdist_wheel"),
                                ("setuptools.dist", "Distribution")):
            modules[name] = ModuleType(name)
            setattr(modules[name], attribute, object)
        modules["setuptools"].setup = lambda **kwargs: captured.update(kwargs)
        modules["setuptools_scm"] = ModuleType("setuptools_scm")
        modules["setuptools_scm"].get_version = lambda **kwargs: "1.0"
        # CMakeExtension is instantiated during setup, so it needs the normal constructor shape.
        modules["setuptools"].Extension = type("Extension", (), {
            "__init__": lambda self, *args, **kwargs: None})
        with patch.dict(sys.modules, modules), patch.object(sys, "platform", platform), patch.dict(os.environ):
            runpy.run_path(str(ROOT / "python/setup.py"))
        return captured

    def test_linux_license_metadata_stays_apache_only(self):
        metadata = self.metadata("linux")
        self.assertEqual(metadata["license_files"], ["LICENSE"])
        self.assertEqual(metadata["license_expression"], "Apache-2.0")
        config = configparser.ConfigParser()
        config.read(ROOT / "python/setup.cfg")
        self.assertNotIn("license_files", config["metadata"])

    def test_darwin_includes_runtime_license_and_notice(self):
        metadata = self.metadata("darwin")
        self.assertEqual(metadata["license_files"], ["LICENSE", "NOTICE", "licenses/LLVM-OpenMP.txt"])
        self.assertEqual(metadata["license_expression"], "Apache-2.0 AND Apache-2.0 WITH LLVM-exception")

    def test_python_license_is_a_regular_copy_of_root_license(self):
        copy = ROOT / "python/LICENSE"
        # A symlink outside python/ breaks source packaging in CI.
        self.assertFalse(copy.is_symlink())
        self.assertEqual(copy.read_bytes(), (ROOT / "LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main()
