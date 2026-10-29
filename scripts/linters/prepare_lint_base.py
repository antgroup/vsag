#!/usr/bin/env python3
"""Prepare an immutable lint comparison before expensive compilation."""

import os
from pathlib import Path
import random
import re
import signal
import subprocess
import sys
import time

from changed_source_filter import build_filter, get_changed_sources

ATTEMPTS = 4
TIMEOUT_SECONDS = 60
TERMINATE_GRACE_SECONDS = 3
BACKOFF_SECONDS = (2, 5, 10)
SHA = re.compile(r"[0-9a-f]{40}")
TRANSIENT = (
    "gnutls recv error", "connection reset", "connection timed out",
    "connection was reset", "failed to connect", "could not resolve host",
    "could not resolve proxy", "operation timed out", "empty reply from server",
    "tls connection was non-properly terminated", "ssl_read", "early eof",
    "http/2 stream", "requested url returned error: 408",
    "requested url returned error: 429", "requested url returned error: 500",
    "requested url returned error: 502", "requested url returned error: 503",
    "requested url returned error: 504",
)
PERMANENT = (
    "authentication failed", "could not read username", "repository not found",
    "permission denied", "not our ref", "couldn't find remote ref",
    "requested url returned error: 401", "requested url returned error: 403",
    "requested url returned error: 404", "certificate verification failed",
    "ssl certificate problem",
)


def run_with_deadline(command, environment):
    # Give Git time to clean its lockfiles, then kill stubborn transport children
    # as a group so inherited pipes cannot keep communicate() waiting forever.
    with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def network_git(arguments):
    """Retry recognized transport failures only; never print credential-bearing stderr."""
    environment = dict(os.environ, GIT_TERMINAL_PROMPT="0", LC_ALL="C")
    for attempt in range(1, ATTEMPTS + 1):
        started = time.monotonic()
        try:
            result = run_with_deadline(["git", *arguments], environment)
            if result.returncode == 0:
                print(f"Lint base {arguments[0]} attempt {attempt}/{ATTEMPTS}: success "
                      f"({time.monotonic() - started:.1f}s)", file=sys.stderr)
                return result.stdout
            diagnostic = result.stderr.lower()
            retryable = (not any(token in diagnostic for token in PERMANENT)
                         and any(token in diagnostic for token in TRANSIENT))
            reason = f"git exit {result.returncode}; " + (
                "transient transport error" if retryable else "non-retryable error")
        except subprocess.TimeoutExpired:
            retryable = True
            reason = f"exceeded {TIMEOUT_SECONDS}s deadline"
        print(f"Lint base {arguments[0]} attempt {attempt}/{ATTEMPTS}: {reason} "
              f"({time.monotonic() - started:.1f}s)", file=sys.stderr)
        if not retryable or attempt == ATTEMPTS:
            raise RuntimeError(f"Lint base preparation failed: {reason}")
        time.sleep(BACKOFF_SECONDS[attempt - 1] + random.uniform(0, 1))
    raise AssertionError("Unreachable retry state")


def prepare_base(base_sha, event_name):
    if not base_sha:
        if event_name != "workflow_dispatch":
            raise ValueError("PR lint requires the event's exact base SHA")
        # Manual runs have no PR base. Resolve main once, then fetch only that SHA.
        lines = network_git(["ls-remote", "--exit-code", "origin", "refs/heads/main"]).splitlines()
        if len(lines) != 1 or len(lines[0].split()) != 2:
            raise ValueError("Cannot resolve one unambiguous main commit")
        base_sha, ref = lines[0].split()
        if ref != "refs/heads/main":
            raise ValueError("Unexpected manual lint base ref")
    if not SHA.fullmatch(base_sha):
        raise ValueError("Lint base must be a full lowercase 40-character commit SHA")
    print(f"Preparing immutable lint base {base_sha}", file=sys.stderr)
    network_git(["fetch", "--no-tags", "--depth=1", "origin", base_sha])
    actual = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{base_sha}^{{commit}}"], text=True,
    ).strip()
    if actual != base_sha:
        raise ValueError("Fetched lint base is not the expected commit")
    return base_sha


def main():
    try:
        output_path = os.environ.get("GITHUB_OUTPUT")
        if not output_path:
            raise ValueError("GITHUB_OUTPUT is not set")
        base = prepare_base(os.environ.get("LINT_BASE_SHA", ""),
                            os.environ.get("GITHUB_EVENT_NAME", ""))
        # Compute the filter now too: invalid Git state must fail before building.
        paths = get_changed_sources(base, "HEAD")
        pattern = build_filter(paths)
        if paths and not pattern:
            raise ValueError("Changed sources produced an empty lint filter")
        print(f"Lint base ready: {base}; changed production sources: {len(paths)}")
        # Publish only after fetch, identity verification and diff all succeeded.
        with Path(output_path).open("a") as output:
            output.write(f"base_sha={base}\npattern={pattern}\n")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"::error::Lint base preparation failed ({type(error).__name__}): {error}",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
