#!/usr/bin/env python3
"""Fail if a selected main check did not succeed, including unexpected skips."""
import json
import os


def validate(needs):
    policy = needs.get("changes", {})
    outputs = policy.get("outputs", {})
    if policy.get("result") != "success" or outputs.get("heavy") not in ("true", "false") or outputs.get("lint") not in ("none", "changed", "full"):
        return False
    selected = {"platforms": outputs["heavy"] == "true", "python": outputs["heavy"] == "true",
                "tsan": outputs["heavy"] == "true", "lint": outputs["lint"] != "none"}
    return all(needs.get(name, {}).get("result") == ("success" if required else "skipped") for name, required in selected.items())


def main():
    needs = json.loads(os.environ["MAIN_NEEDS"])
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
        summary.write("## Main CI\n\n| Check | Result |\n| --- | --- |\n")
        for name, value in needs.items():
            summary.write(f"| {name} | {value.get('result', 'missing')} |\n")
        summary.write("\nDocs-only pushes intentionally skip heavy checks. Full regression belongs to the manually triggered Daily Build & Test workflow until scheduling is separately enabled.\n")
    if not validate(needs):
        raise SystemExit("A selected main check did not succeed, or classification was invalid")


if __name__ == "__main__":
    main()
