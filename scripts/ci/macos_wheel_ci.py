#!/usr/bin/env python3
"""macOS installed-wheel matrix and verified TestPyPI round trip."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# Also support direct execution from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import python_ci as ci

# Update deliberately after validating both NumPy OpenMP lanes; see ci-routing.md.
OPENMP_PACKAGES = ('torch==2.15.0', 'scikit-learn==1.9.1')
DEPLOYMENT_TARGET = '15.0'
PLATFORM_TAG = f"macosx_{DEPLOYMENT_TARGET.replace('.', '_')}_arm64"


def environments():
    return ci.environments() + [dict(python='3.12', numpy='1.26.4', mode='openmp',
                                     key='py312-np1-openmp')]


def published_version(manifest, source):
    record = json.loads(Path(manifest).read_text())
    return manifest_version(record, source)


def manifest_version(record, source):
    match = re.fullmatch(rf'pyvsag-([A-Za-z0-9_.!+]+)-py3-none-{PLATFORM_TAG}\.whl',
                         record.get('filename', ''))
    if not match:
        raise ValueError(f'Expected wheel platform {PLATFORM_TAG} '
                         f'(MACOSX_DEPLOYMENT_TARGET={DEPLOYMENT_TARGET}); '
                         f"got {record.get('filename', '<missing>')}")
    if (record.get('schema') != 1 or record.get('source_sha') != source
            or not re.fullmatch(r'[a-f0-9]{40}', source) or record.get('arch') != 'arm64'
            or not re.fullmatch(r'[a-f0-9]{64}', record.get('sha256', ''))):
        raise ValueError('Expected a macOS arm64 wheel manifest for the selected source')
    return match.group(1)


def write_manifest(directory, manifest, source):
    record = ci.wheel_manifest(directory, source, 'arm64')
    manifest_version(record, source)  # Fail the build before uploading an unexpected platform.
    Path(manifest).write_text(json.dumps(record, indent=2) + '\n')


def require_unpublished(manifest, source):
    """Fail before upload if an immutable same-name wheel already exists."""
    version = published_version(manifest, source)
    filename = json.loads(Path(manifest).read_text())['filename']
    for attempt in range(1, 4):
        try:
            with urlopen(f'https://test.pypi.org/pypi/pyvsag/{version}/json', timeout=30) as response:
                payload = json.load(response)
            files = payload.get('urls') if isinstance(payload, dict) else None
            if not isinstance(files, list) or any(
                    not isinstance(file, dict) or not isinstance(file.get('filename'), str)
                    for file in files):
                raise ValueError('Invalid TestPyPI file list')
        except HTTPError as error:
            if error.code == 404:
                return
            if error.code != 429 and error.code < 500:
                raise RuntimeError(f'TestPyPI publication check failed: HTTP {error.code}; '
                                   'publication blocked') from None
            diagnostic = f'HTTP {error.code}'
        except (URLError, TimeoutError, ValueError) as error:
            diagnostic = f'{type(error).__name__}: {error}'
        else:
            if any(file['filename'] == filename for file in files):
                raise ValueError(
                    f'{filename} already exists on TestPyPI; refusing to compare a rebuild with an '
                    'immutable upload. Rerun only the failed verification job in the original upload '
                    'run to reuse its manifest, or publish a new version from a new main commit.')
            return
        print(f'TestPyPI publication check attempt {attempt}/3 failed: {diagnostic}', flush=True)
        if attempt < 3:
            time.sleep(5 * attempt)
    raise RuntimeError('Cannot confirm TestPyPI filename availability after 3 attempts; '
                       'publication blocked') from None


def download(manifest, source, directory, window=600):
    """Only TestPyPI, no dependencies/cache; allow ten minutes for index propagation."""
    version = published_version(manifest, source)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.iterdir()):
        raise ValueError('Published-wheel destination must be empty')
    deadline = time.monotonic() + window
    attempt = 0
    while (remaining := deadline - time.monotonic()) > 0:
        attempt += 1
        print(f'TestPyPI download attempt {attempt}; {remaining:.0f}s remaining', flush=True)
        with tempfile.TemporaryDirectory(prefix='testpypi-') as tmp:
            try:
                result = subprocess.run([
                    sys.executable, '-m', 'pip', '--isolated', 'download',
                    '--index-url', 'https://test.pypi.org/simple/', '--no-deps',
                    '--only-binary=:all:', '--no-cache-dir', '--timeout', '30', '--retries', '0',
                    '--dest', tmp, f'pyvsag=={version}',
                ], timeout=min(60, remaining))
            except subprocess.TimeoutExpired:
                print('TestPyPI download attempt timed out', flush=True)
            else:
                if result.returncode == 0:
                    wheel = ci.verify(tmp, manifest, source, 'arm64')
                    destination = directory / wheel.name
                    shutil.copyfile(wheel, destination)
                    return destination
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(15 * 2 ** min(attempt - 1, 2), remaining))
    raise RuntimeError(f'Exact wheel is not available from TestPyPI after {window}s retry window')


def test_wheel(directory, manifest, source, numpy, mode, output):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('macOS wheel tests require a native Apple Silicon interpreter')
    if numpy not in ('1.26.4', '2.*') or mode not in ('ordinary', 'openmp'):
        raise ValueError('Unsupported NumPy requirement or test mode')
    published_version(manifest, source)
    checkout = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ci.ROOT, text=True).strip()
    if checkout != source:
        raise ValueError('Test checkout differs from wheel source')
    wheel = ci.verify(directory, manifest, source, 'arm64')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, VSAG_CI_SOURCE_SHA=source, KMP_DUPLICATE_LIB_OK='FALSE')
    for name in ('PYVSAG_LIBRARY', 'PYTHONPATH', 'PYTHONHOME'):
        env.pop(name, None)
    with tempfile.TemporaryDirectory(prefix='pyvsag-macos-test-') as tmp:
        tmp = Path(tmp)
        subprocess.run([sys.executable, '-m', 'venv', str(tmp / 'venv')], check=True, env=env)
        python = str(tmp / 'venv/bin/python')
        # One resolver transaction keeps Torch/sklearn from upgrading the selected NumPy.
        packages = [str(wheel), f'numpy=={numpy}', 'pytest']
        if mode == 'openmp':
            packages += OPENMP_PACKAGES
        subprocess.run([python, '-m', 'pip', '--isolated', 'install',
                        '--index-url', 'https://pypi.org/simple/', '--only-binary=:all:',
                        '--report', str(output / 'install.json'), *packages], check=True, env=env)
        subprocess.run([python, '-m', 'pip', 'check'], check=True, env=env)
        shutil.copyfile(manifest, tmp / 'wheel-manifest.json')
        subprocess.run([python, str(ci.ROOT / 'scripts/ci/python_ci.py'), 'test',
                        '--mode', mode, '--output', str(output / 'results.json')],
                       cwd=tmp, check=True, env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['matrix', 'test', 'download', 'check-unpublished', 'manifest'])
    parser.add_argument('--source')
    parser.add_argument('--manifest', default='wheel-manifest.json')
    parser.add_argument('--directory', default='wheelhouse')
    parser.add_argument('--numpy', choices=['1.26.4', '2.*'], default='2.*')
    parser.add_argument('--mode', choices=['ordinary', 'openmp'], default='ordinary')
    parser.add_argument('--output', default='python-test-report')
    args = parser.parse_args()
    if args.command == 'matrix':
        print(json.dumps({'include': environments()}))
    elif args.command == 'manifest':
        write_manifest(args.directory, args.manifest, args.source)
    elif args.command == 'check-unpublished':
        require_unpublished(args.manifest, args.source)
    elif args.command == 'download':
        print(download(args.manifest, args.source, args.directory))
    else:
        test_wheel(args.directory, args.manifest, args.source, args.numpy, args.mode, args.output)


if __name__ == '__main__':
    main()
