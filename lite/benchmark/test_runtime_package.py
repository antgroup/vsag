# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest

from measure_runtime_package import archive_bundle, inventory, resolved_dependencies, safe_bundle_unpack
from run_mixed_cpu_acceptance import sha


class RuntimePackageTests(unittest.TestCase):
    def test_dependency_aliases_and_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'real.so').write_bytes(b'elf')
            (root/'alias.so').symlink_to('real.so')
            parsed = resolved_dependencies('alias.so => '+str(root/'alias.so')+' (0x123)\n')
            self.assertEqual(parsed, {'alias.so':(root/'real.so').resolve()})
            with self.assertRaises(ValueError):
                resolved_dependencies('unknown.so => not found')

    def test_deterministic_pack_and_safe_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/'bundle'
            (source/'lib').mkdir(parents=True)
            (source/'lib/real.so').write_bytes(b'elf')
            (source/'lib/alias.so').symlink_to('real.so')
            archive_bundle(source,root/'first.gz')
            os.utime(source/'lib/real.so',(20,20))
            archive_bundle(source,root/'second.gz')
            self.assertEqual(sha(root/'first.gz'),sha(root/'second.gz'))
            safe_bundle_unpack(root/'first.gz',root/'relocated')
            self.assertEqual(inventory(source),inventory(root/'relocated'))

    def test_unsafe_members_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name,kind,target in [('../escape',tarfile.REGTYPE,''),
                                     ('lib/alias',tarfile.SYMTYPE,'../../escape'),
                                     ('lib/alias',tarfile.SYMTYPE,'/outside')]:
                with tarfile.open(root/'bad.gz','w:gz') as stream:
                    member=tarfile.TarInfo(name)
                    member.type=kind
                    member.linkname=target
                    stream.addfile(member,io.BytesIO(b''))
                with self.assertRaises(ValueError):
                    safe_bundle_unpack(root/'bad.gz',root/'out')


if __name__ == '__main__':
    unittest.main()
