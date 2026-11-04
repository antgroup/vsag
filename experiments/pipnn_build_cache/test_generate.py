# Copyright 2024-present the vsag project
# SPDX-License-Identifier: Apache-2.0
import tempfile
from pathlib import Path
import unittest

import generate


class GenerationTests(unittest.TestCase):
    def test_formatted_main_boundary(self):
        source = 'void helper() {}\nint\nmain(int argc, char** argv) {}\n'
        self.assertEqual(generate.before_main(source), 'void helper() {}\n')

    def test_missing_main_fails(self):
        with self.assertRaisesRegex(ValueError, 'main not found'):
            generate.before_main('void helper() {}')

    def test_generation_keeps_production_source_unchanged(self):
        repo = Path(__file__).resolve().parents[2]
        source = repo / 'src/impl/pipnn/pipnn_graph_builder.cpp'
        original = source.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            generate.generate(repo, output)
            self.assertIn('::capture_sink', (output / 'builder_exposed.cpp').read_text())
            self.assertIn('++::prune_distance_calls', (output / 'final_reuse_builder.cpp').read_text())
            self.assertEqual((output / 'search_qps_driver.cpp').read_text().count('benchmark_pair('), 1)
        self.assertEqual(source.read_bytes(), original)

    def test_unsupported_builder_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            source = repo / 'src/impl/pipnn/pipnn_graph_builder.cpp'
            source.parent.mkdir(parents=True)
            source.write_text('unsupported builder\n')
            with self.assertRaisesRegex(ValueError, 'unsupported builder revision'):
                generate.generate(repo, repo / 'generated')


if __name__ == '__main__':
    unittest.main()
