# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from pack_mixed_cpu_acceptance import unpack


class ArchiveTests(unittest.TestCase):
    def test_regular_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with tarfile.open(root / 'raw.tar.gz', 'w:gz') as stream:
                member = tarfile.TarInfo('nested/test.txt')
                member.size = 3
                stream.addfile(member, io.BytesIO(b'raw'))
            unpack(root / 'raw.tar.gz', root / 'extracted')
            self.assertEqual((root / 'extracted/nested/test.txt').read_bytes(), b'raw')

    def test_unsafe_paths_and_links_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, kind in [('../escape', tarfile.REGTYPE),
                               ('/absolute', tarfile.REGTYPE),
                               ('link', tarfile.SYMTYPE), ('hardlink', tarfile.LNKTYPE)]:
                with tarfile.open(root / 'raw.tar.gz', 'w:gz') as stream:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    stream.addfile(member)
                with self.assertRaises(ValueError):
                    unpack(root / 'raw.tar.gz', root / 'extracted')


if __name__ == '__main__':
    unittest.main()
