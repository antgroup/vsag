#!/usr/bin/env python3
"""Skip redundant CircleCI tests only for positively identified main PRs.

No token is needed for this public repository. Missing/ambiguous metadata or
API failures conservatively retain the existing build. Push jobs never call us.
"""

from http.client import HTTPException
import json
import os
import re
import subprocess
import time
import urllib.request


def should_skip(env, fetch):
    urls = set(filter(None, (env.get("CIRCLE_PULL_REQUEST", "") + "," +
                            env.get("CIRCLE_PULL_REQUESTS", "")).split(",")))
    if not urls or not env.get("CIRCLE_SHA1"):
        return False
    for url in urls:
        match = re.fullmatch(r"https://github\.com/antgroup/vsag/pull/([1-9][0-9]*)", url)
        if not match:
            return False
        pr = fetch("https://api.github.com/repos/antgroup/vsag/pulls/" + match[1])
        if (pr.get("state") != "open" or pr.get("base", {}).get("ref") != "main"
                or pr.get("base", {}).get("repo", {}).get("full_name") != "antgroup/vsag"
                or env["CIRCLE_SHA1"] not in (pr.get("head", {}).get("sha"),
                                            pr.get("merge_commit_sha"))):
            return False
    return True


def fetch_pr(url):
    for attempt in range(2):
        try:
            request = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.load(response)
        except (OSError, ValueError, HTTPException):
            if attempt:
                raise
            time.sleep(1)


def main():
    try:
        skip = should_skip(os.environ, fetch_pr)
    except (OSError, ValueError, TypeError, AttributeError, HTTPException):
        print("PR target could not be confirmed; retaining CircleCI tests.")
        return
    if skip:
        print("Main-target PR: Linux tests are enforced by PR CI Gate; skipping redundant CircleCI build.")
        try:
            subprocess.run(["circleci-agent", "step", "halt"], check=True)
        except (OSError, subprocess.CalledProcessError):
            print("circleci-agent halt failed; retaining CircleCI tests.")
    else:
        print("Retaining CircleCI tests for this branch/PR.")


if __name__ == "__main__":
    main()
