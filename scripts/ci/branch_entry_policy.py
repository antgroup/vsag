#!/usr/bin/env python3
"""Preserve version push and SIMD scopes when standalone entries are consolidated."""
import json
import os
from pathlib import Path
import re
import subprocess
from checks_policy import changed_paths

PROFILES = {'full-branch', 'asan', 'tsan', 'python', 'lint', 'coverage', 'compatibility', 'simd'}


def select(name, event, profile, paths):
    if profile not in PROFILES:
        raise ValueError('Unknown branch profile')
    live = name == 'push' and event.get('deleted') is not True
    ref = event.get('ref', '')
    version = live and re.fullmatch(r'refs/heads/0\.[^/]*', ref) is not None
    return {'profile': profile, 'version': version,
            'non_docs': live and any(not p.startswith('docs/') for p in paths),
            'simd': live and (version or ref == 'refs/heads/main') and
                    any(p.startswith('src/simd/') for p in paths)}


def main():
    name = os.environ['GITHUB_EVENT_NAME']
    event = json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    profile = os.environ.get('PROFILE', 'full-branch')
    try:
        paths = changed_paths(name, event)
    except (ValueError, subprocess.CalledProcessError):
        # Missing/replaced history must not silently drop version or SIMD checks.
        paths = ['src/simd/__unknown_push_range__']
        print('Push range unavailable; selecting conservative branch coverage')
    result = select(name, event, profile, paths)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
        for key, value in result.items():
            out.write(f'{key}={str(value).lower() if isinstance(value, bool) else value}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
