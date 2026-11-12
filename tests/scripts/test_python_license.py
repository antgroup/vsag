#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PythonLicenseTest(unittest.TestCase):
    def test_python_license_is_a_regular_copy_of_root_license(self):
        copy = ROOT / "python/LICENSE"
        # A symlink outside python/ breaks source packaging in CI.
        self.assertFalse(copy.is_symlink())
        self.assertEqual(copy.read_bytes(), (ROOT / "LICENSE").read_bytes())


if __name__ == "__main__":
    unittest.main()
