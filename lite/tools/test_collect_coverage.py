import unittest
from pathlib import Path

from collect_coverage import merge_report, summarize


class CoverageTest(unittest.TestCase):
    def test_union_deduplicates_header_lines_and_excludes_tests_dependencies(self):
        root = Path('/example/repository')
        union = {}
        for count in (0, 3):
            merge_report({'files': [
                {'file': str(root / 'src/lite/query_heap.h'),
                 'lines': [{'line_number': 1, 'count': count}, {'line_number': 2, 'count': 0}]},
                {'file': str(root / 'src/lite/index_test.cpp'), 'lines': [{'line_number': 1, 'count': 4}]},
                {'file': '/dependency/other.h', 'lines': [{'line_number': 1, 'count': 4}]},
            ]}, root, union)
        self.assertEqual(summarize(union)['total'], 2)
        self.assertEqual(summarize(union)['covered'], 1)
        self.assertEqual(summarize(union)['percent'], 50)

    def test_relative_paths_and_empty_report(self):
        root = Path('/example/repository')
        union = {}
        merge_report({'current_working_directory': str(root), 'files': [
            {'file': 'include/vsag/lite/index.h', 'lines': [{'line_number': 7, 'count': 2}]}
        ]}, root, union)
        self.assertEqual(summarize(union)['percent'], 100)
        with self.assertRaises(ValueError):
            summarize({})


if __name__ == '__main__':
    unittest.main()
