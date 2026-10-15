# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest

from verify_pinned_full_acceptance import verify
from run_pinned_full_acceptance import validate_fresh_receipt


class FullAuditTests(unittest.TestCase):
    def test_fresh_receipt_guards(self):
        receipt = dict(full_source_head='feature', fresh_core=True, ccache=False,
            prior_object_count=0, core_object_count=100, install='/tmp/full-reference',
            library_sha256='a' * 64, full_headers_sha256={'vsag.h': 'b' * 64})
        validate_fresh_receipt(receipt, 'feature', Path('/tmp/full-reference'))
        for field, invalid in [('full_source_head', 'historical'), ('fresh_core', False),
                ('ccache', True), ('prior_object_count', 1), ('core_object_count', 1),
                ('install', '/tmp/another-reference'), ('library_sha256', 'short'),
                ('full_headers_sha256', {})]:
            with self.subTest(field=field):
                altered = dict(receipt, **{field: invalid})
                with self.assertRaises(AssertionError):
                    validate_fresh_receipt(altered, 'feature', Path('/tmp/full-reference'))

    def test_nonhex_receipt_digest_rejected(self):
        with self.assertRaises(ValueError):
            validate_fresh_receipt(dict(full_source_head='feature', fresh_core=True,
                ccache=False, prior_object_count=0, core_object_count=100,
                install='/tmp/full-reference', library_sha256='z' * 64), 'feature', Path('/tmp/full-reference'))

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
