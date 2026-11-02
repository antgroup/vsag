import fnmatch
import re
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


class StandaloneWorkflowTest(unittest.TestCase):
    def test_repository_dependencies_trigger_lite_ci(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / '.github/workflows/lite.yml').read_text()
        path_block = workflow.split('    paths:\n', 1)[1].split('\n\n', 1)[0]
        patterns = [
            line.strip()[2:].strip().strip('"').strip("'")
            for line in path_block.splitlines() if line.strip().startswith('- ')
        ]
        cmake = (root / 'lite/CMakeLists.txt').read_text()
        dependencies = {
            name for name in re.findall(r'VSAG_ROOT\}/([^"]+)', cmake)
            if (root / name).is_file()
        }
        dependencies.update({
            'include/vsag/lite/index.h',
            'cmake/VSAGThirdPartyOverride.cmake',
            'extern/catch2/catch2.cmake',
            '.clang-format',
            '.clang-tidy',
        })
        pending = list(dependencies)
        while pending:
            name = pending.pop()
            if Path(name).suffix not in {'.h', '.hpp', '.cpp'}:
                continue
            source = (root / name).read_text()
            for include in re.findall(r'#include\s*[<"]([^>"]+)[>"]', source):
                candidates = [
                    Path(name).parent / include, Path('src') / include, Path('include') / include
                ]
                dependency = next(
                    (candidate.as_posix() for candidate in candidates
                     if (root / candidate).is_file()), None
                )
                if dependency is not None and dependency not in dependencies:
                    dependencies.add(dependency)
                    pending.append(dependency)
        missing = sorted(
            name for name in dependencies
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        )
        self.assertFalse(missing, f'Lite dependencies missing from CI triggers: {missing}')


if __name__ == '__main__':
    unittest.main()
