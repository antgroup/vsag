#!/usr/bin/env python3
"""Aggregate an expanded check profile without accepting missing selected jobs."""
import json
import os


def validate(results, required):
    if not isinstance(results, dict) or not isinstance(required, list) or not required:
        return False
    if not all(isinstance(key, str) for key in required) or len(set(required)) != len(required):
        return False
    if any(not isinstance(value, dict) or value.get('result') not in ('success', 'skipped')
           for value in results.values()):
        return False
    return all(results.get(key, {}).get('result') == 'success' for key in required)


def main():
    if not validate(json.loads(os.environ['RESULTS']), json.loads(os.environ['REQUIRED'])):
        raise SystemExit('Selected jobs failed, were cancelled, skipped unexpectedly, or are missing')


if __name__ == '__main__':
    main()
