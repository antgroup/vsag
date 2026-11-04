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

    def test_existing_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / 'extracted'
            destination.mkdir()
            (destination / 'kept').write_bytes(b'original')
            with tarfile.open(root / 'raw.tar.gz', 'w:gz') as stream:
                member = tarfile.TarInfo('kept')
                member.size = 3
                stream.addfile(member, io.BytesIO(b'new'))
            with self.assertRaises(FileExistsError):
                unpack(root / 'raw.tar.gz', destination)
            self.assertEqual((destination / 'kept').read_bytes(), b'original')

    def test_duplicate_members_rejected_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with tarfile.open(root / 'raw.tar.gz', 'w:gz') as stream:
                for value in [b'first', b'other']:
                    member = tarfile.TarInfo('same')
                    member.size = len(value)
                    stream.addfile(member, io.BytesIO(value))
            with self.assertRaises(ValueError):
                unpack(root / 'raw.tar.gz', root / 'extracted')
            self.assertFalse((root / 'extracted').exists())

    def test_late_unsafe_member_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with tarfile.open(root / 'raw.tar.gz', 'w:gz') as stream:
                for name in ['valid', '../unsafe']:
                    member = tarfile.TarInfo(name)
                    member.size = 3
                    stream.addfile(member, io.BytesIO(b'raw'))
            with self.assertRaises(ValueError):
                unpack(root / 'raw.tar.gz', root / 'extracted')
            self.assertFalse((root / 'extracted').exists())


if __name__ == '__main__':
    unittest.main()
