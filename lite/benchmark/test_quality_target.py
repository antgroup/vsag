# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
import tempfile
import unittest
from run_quality_target import audit_query
from verify_quality_target import quantile, verify


class QualityTargetTests(unittest.TestCase):
    def fixture(self, directory, corruption=None):
        stem=Path(directory)/'sample'
        Path(str(stem)+'.queries.csv').write_text('query,latency_us\n0,1\n1,2\n')
        lines=['query,rank,id,distance']
        for query in range(2):
            for rank in range(10):
                identifier=rank
                distance=float(rank).hex()
                if corruption=='duplicate' and rank==9: identifier=0
                if corruption=='nonfinite' and rank==9: distance='nan'
                if corruption=='ordering' and rank==9: distance='0x0p+0'
                lines.append(f'{query},{rank},{identifier},{distance}')
        Path(str(stem)+'.neighbors.csv').write_text('\n'.join(lines)+'\n')
        return stem,[set(range(10)),set(range(10))]

    def test_valid_raw_hits(self):
        with tempfile.TemporaryDirectory() as directory:
            stem,truths=self.fixture(directory)
            self.assertEqual(audit_query(stem,truths,0,2)['hits'],20)

    def test_raw_corruptions_rejected(self):
        for corruption in ['duplicate','nonfinite','ordering']:
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                stem,truths=self.fixture(directory,corruption)
                with self.assertRaises(AssertionError): audit_query(stem,truths,0,2)

    def test_query_interval_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stem,truths=self.fixture(directory)
            with self.assertRaises(AssertionError): audit_query(stem,truths,1,2)

    def test_quantile_nearest_rank(self):
        self.assertEqual(quantile(list(range(1,901)),.99),891)
        self.assertEqual(quantile(list(range(1,901)),.5),450)

    def test_manifest_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'protocol.json').write_text('{}')
            (root/'manifest.json').write_text(json.dumps({'protocol.json':'invalid'}))
            with self.assertRaises(AssertionError): verify(root)

    def test_duplicate_result_query_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            stem,truths=self.fixture(directory)
            path=Path(str(stem)+'.neighbors.csv')
            path.write_text(path.read_text().replace('1,0,0,','0,0,0,'))
            with self.assertRaises(AssertionError): audit_query(stem,truths,0,2)

if __name__=='__main__': unittest.main()
