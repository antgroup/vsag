# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest

from verify_pinned_full_acceptance import verify


class FullAuditTests(unittest.TestCase):
    def fixture(self, directory, ef=512, digest='full'):
        root = Path(directory)
        (root / 'measurements').mkdir()
        (root / 'provenance.json').write_text(json.dumps(
            dict(full_library_sha256='full', lite_library_sha256='lite')))
        (root / 'manifest.json').write_text('{}')
        (root / 'measurements/identity.json').write_text(json.dumps(dict(
            arguments=dict(ef=ef, full_ef=512), reference_library_sha256=digest,
            lite_library_sha256='lite')))
        return root

    def test_unequal_query_budget_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, ef=128)
            with self.assertRaises(AssertionError):
                verify(root)

    def test_wrong_reference_library_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory, digest='old-lite')
            with self.assertRaises(AssertionError):
                verify(root)

    def test_manifest_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            (root / 'manifest.json').write_text(json.dumps({'provenance.json': 'incorrect'}))
            with self.assertRaises(AssertionError):
                verify(root)


if __name__ == '__main__':
    unittest.main()
