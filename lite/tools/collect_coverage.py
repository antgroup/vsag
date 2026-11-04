#!/usr/bin/env python3
"""Collect gcov JSON line union across Lite sources and inline-header test units."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import tempfile


def merge_report(report, root, union):
    for entry in report['files']:
        path = Path(entry['file'])
        if not path.is_absolute():
            path = Path(report['current_working_directory']) / path
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        if not relative.startswith(('src/lite/', 'include/vsag/lite/')):
            continue
        if relative.endswith('_test.cpp'):
            continue
        lines = union.setdefault(relative, {})
        for line in entry['lines']:
            number = line['line_number']
            lines[number] = max(lines.get(number, 0), line['count'])


def summarize(union):
    files = {path: {'covered': sum(count > 0 for count in lines.values()),
                    'total': len(lines)} for path, lines in sorted(union.items())}
    total = sum(entry['total'] for entry in files.values())
    covered = sum(entry['covered'] for entry in files.values())
    if total == 0:
        raise ValueError('No instrumented production Lite lines found')
    return {'scope': 'production Lite source/header executable-line union, not Full or branch coverage',
            'covered': covered, 'total': total, 'percent': covered * 100 / total, 'files': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fail-under', type=float, default=90)
    args = parser.parse_args()
    counters = sorted(args.build.resolve().rglob('*.gcda'))
    if not counters:
        parser.error('No gcda counters: build with ENABLE_COVERAGE and run tests first')
    union = {}
    with tempfile.TemporaryDirectory(prefix='lite-gcov-') as temporary:
        for counter in counters:
            subprocess.run(['gcov', '--json-format', str(counter)], cwd=temporary,
                           check=True, stdout=subprocess.DEVNULL)
            reports = list(Path(temporary).glob('*.gcov.json.gz'))
            if not reports:
                raise ValueError(f'gcov produced no JSON for {counter}')
            for report in reports:
                merge_report(json.loads(gzip.decompress(report.read_bytes())), args.root, union)
                report.unlink()
    result = summarize(union)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['percent'] < args.fail_under:
        raise SystemExit('Production Lite line coverage is below the configured threshold')


if __name__ == '__main__':
    main()
