#!/usr/bin/env python3
"""Check all configured Lite C++ units, including optional ISA and test units."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True, type=Path)
    args = parser.parse_args()
    database = json.loads((args.build / 'compile_commands.json').read_text())
    sources = sorted({entry['file'] for entry in database
                      if '/src/lite/' in Path(entry['file']).as_posix()})
    if not sources:
        parser.error('No configured Lite translation units found')
    subprocess.run(['clang-tidy-15', *sources, '-p', str(args.build),
                    '--header-filter=.*/(src/lite|include/vsag/lite)/.*'], check=True)


if __name__ == '__main__':
    main()
