#!/usr/bin/env python3
"""Linux wheel identity, supported test matrix and honest pytest reporting."""
import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]


def environments():
    # NumPy 1.26 supports CPython through 3.12. Freeze actual resolved versions in reports.
    result = []
    for minor in range(10, 15):
        for numpy in (['1.26.4', '2.*'] if minor <= 12 else ['2.*']):
            result.append(dict(python=f'3.{minor}', numpy=numpy, mode='ordinary',
                               key=f'py3{minor}-np{numpy[0]}'))
    result.append(dict(python='3.12', numpy='2.*', mode='openmp', key='py312-np2-openmp'))
    return result


def lanes():
    return [f'full-python-test-python-{arch}-{env["key"]}-pytest'
            for arch in ('x86', 'aarch64') for env in environments()]


def wheel_manifest(directory, sha, arch):
    if not re.fullmatch(r'[a-f0-9]{40}', sha) or arch not in ('x86_64', 'aarch64'):
        raise ValueError('Invalid source or architecture')
    wheels = list(Path(directory).glob('*.whl'))
    if len(wheels) != 1 or not wheels[0].name.endswith(f'_{arch}.whl'):
        raise ValueError('Expected exactly one wheel of the requested architecture')
    wheel = wheels[0]
    return dict(schema=1, source_sha=sha, arch=arch, filename=wheel.name,
                sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())


def verify(directory, manifest, sha, arch):
    recorded = json.loads(Path(manifest).read_text())
    actual = wheel_manifest(directory, sha, arch)
    if actual != recorded:
        raise ValueError('Wheel source/architecture/hash does not match manifest')
    return (Path(directory) / actual['filename']).resolve()


def junit_counts(path, mode):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    counts = dict(collected=len(cases), passed=0, failed=0, skipped=0)
    for case in cases:
        key = ('failed' if case.find('failure') is not None or case.find('error') is not None
               else 'skipped' if case.find('skipped') is not None else 'passed')
        counts[key] += 1
    if not cases or (mode == 'openmp' and (len(cases) < 6 or counts['skipped'])):
        raise ValueError('Required test inventory empty or OpenMP cases missing/skipped')
    return counts


def run_tests(mode, output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(mode=mode, lane=os.environ.get('VSAG_DAILY_LANE'),
        source_sha=os.environ.get('VSAG_CI_SOURCE_SHA') or subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        python=platform.python_version(), platform=platform.platform(), status='failed')
    start = time.monotonic()
    code = 1
    try:
        package = importlib.import_module('pyvsag')
        report['import_path'] = str(Path(package.__file__).resolve())
        if (ROOT / 'python') in Path(package.__file__).resolve().parents:
            raise ValueError('Imported source tree instead of installed wheel')
        modules = ['numpy', 'cffi'] + (['torch', 'sklearn'] if mode == 'openmp' else [])
        report['dependencies'] = {m: importlib.import_module(m).__version__ for m in modules}
        manifest = Path('wheel-manifest.json')
        if manifest.exists():
            report['wheel'] = json.loads(manifest.read_text())
            if report['wheel']['source_sha'] != report['source_sha']:
                raise ValueError('Test checkout differs from wheel source')
        xml = output.with_suffix('.xml')
        command = [sys.executable, '-m', 'pytest', '-v', '--tb=short', f'--junitxml={xml}']
        command += (['test_openmp_coexistence.py'] if mode == 'openmp' else
                    ['.', '--ignore=test_openmp_coexistence.py'])
        code = subprocess.run(command, cwd=ROOT / 'tests/python').returncode
        report.update(junit_counts(xml, mode))
        if code == 0 and report['failed'] == 0 and report['passed'] > 0:
            report['status'] = 'passed'
    except Exception as error:
        report['error'] = str(error)
    finally:
        report['seconds'] = time.monotonic() - start
        output.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'passed' else (code or 1)


def load_reports(directory, sha):
    reports = []
    for path in Path(directory).rglob('results.json'):
        report = json.loads(path.read_text())
        if report.get('source_sha') != sha or report.get('status') != 'passed':
            raise ValueError('Missing or unsuccessful Python test evidence')
        wheel = report.get('wheel', {})
        if wheel.get('source_sha') != sha or not re.fullmatch('[a-f0-9]{64}', wheel.get('sha256', '')):
            raise ValueError('Python report lacks matching wheel provenance')
        reports.append(report)
    expected = {lane.removesuffix('-pytest') for lane in lanes()}
    if len(reports) != len(expected) or {r.get('lane') for r in reports} != expected:
        raise ValueError('Missing, duplicate or unexpected Python environment report')
    for arch, tag in [('x86_64', 'x86'), ('aarch64', 'aarch64')]:
        subset = [r for r in reports if r['lane'].startswith(f'full-python-test-python-{tag}-')]
        if any(r['wheel']['arch'] != arch for r in subset) or len({r['wheel']['sha256'] for r in subset}) != 1:
            raise ValueError('Python environments did not reuse the same platform wheel')
        for env in environments():
            report = next(r for r in subset if r['lane'].endswith('-' + env['key']))
            if (not report['python'].startswith(env['python'] + '.')
                    or report['dependencies']['numpy'].split('.')[0] != env['numpy'][0]
                    or (env['numpy'] == '1.26.4' and report['dependencies']['numpy'] != '1.26.4')
                    or report['mode'] != env['mode'] or report['collected'] < 1 or report['passed'] < 1
                    or report['failed'] or report['passed'] + report['skipped'] != report['collected']):
                raise ValueError('Python environment/count mismatch')
            if env['mode'] == 'openmp' and (report['skipped'] or report['collected'] < 6):
                raise ValueError('Required OpenMP cases skipped/missing')
    return sorted(reports, key=lambda r: r['lane'])


def timing(jobs):
    """Occupied runner time and active span; neither invents dependency queue time."""
    selected = [j for j in jobs if j['name'].rsplit(' / ', 1)[-1].startswith(('Python Wheel Build ', 'Python x86 ', 'Python aarch64 '))]
    starts, ends, rows = [], [], []
    for job in selected:
        if not job.get('started_at') or not job.get('completed_at'):
            continue
        start = dt.datetime.fromisoformat(job['started_at'].replace('Z', '+00:00'))
        end = dt.datetime.fromisoformat(job['completed_at'].replace('Z', '+00:00'))
        if start.tzinfo is None or end.tzinfo is None or end < start:
            raise ValueError('Invalid Python job timestamps')
        starts.append(start)
        ends.append(end)
        rows.append(dict(name=job['name'], seconds=(end-start).total_seconds()))
    return dict(jobs=rows, expected_jobs=len(lanes()) + 2, complete=len(rows) == len(lanes()) + 2,
                wall_seconds=(max(ends)-min(starts)).total_seconds() if rows else None,
                runner_seconds=sum(r['seconds'] for r in rows) if rows else None)


def markdown(reports, job_metrics=None):
    if not reports:
        return []
    cost = timing(job_metrics or [])
    lines = ['### Python installed-wheel tests', '']
    if cost['jobs']:
        lines += [f"Active span: {cost['wall_seconds']/60:.1f} min; runner occupancy: {cost['runner_seconds']/60:.1f} runner-min ({len(cost['jobs'])}/{cost['expected_jobs']} completed job timestamps).", '',
                  'Active span includes gaps between Python jobs; pre-start queue time cannot be isolated from dependency waits. No CPU-core quota is inferred.', '']
    else:
        lines += ['Python job timing evidence unavailable.', '']
    lines += ['<details><summary>Python environment results</summary>', '',
             '| Environment | Python / NumPy | Passed / failed / skipped | Test min |',
             '| --- | --- | ---: | ---: |']
    for r in reports:
        lines.append(f"| {r['lane'].removeprefix('full-python-test-python-')} | {r['python']} / {r['dependencies']['numpy']} | "
                     f"{r['passed']} / {r['failed']} / {r['skipped']} | {r['seconds']/60:.2f} |")
    lines += ['', '</details>', '', 'Linux: eight ordinary environments and one Python 3.12 OpenMP lane per architecture. '
              'macOS Python is deferred to the separate packaging adaptation. '
              'Required Python runs before optional C++ top-up; detailed wheel hashes, imports and dependencies are in artifacts.', '']
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['matrix', 'manifest', 'verify', 'test'])
    parser.add_argument('--arch', choices=['x86_64', 'aarch64'])
    parser.add_argument('--source')
    parser.add_argument('--directory', default='wheelhouse')
    parser.add_argument('--manifest', default='wheel-manifest.json')
    parser.add_argument('--mode', choices=['ordinary', 'openmp'], default='ordinary')
    parser.add_argument('--output', default='python-test-report/results.json')
    args = parser.parse_args()
    if args.command == 'matrix':
        print(json.dumps({'include': environments()}))
    elif args.command == 'manifest':
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        if actual != args.source:
            raise ValueError('Build checkout differs from expected source')
        Path(args.manifest).write_text(json.dumps(wheel_manifest(args.directory, args.source, args.arch), indent=2) + '\n')
    elif args.command == 'verify':
        print(verify(args.directory, args.manifest, args.source, args.arch))
    else:
        return run_tests(args.mode, args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
