#!/usr/bin/env python3

"""Render a checked-in SAQ evaluation template without requiring PyYAML."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    replacements = {
        "@DATASET@": str(args.dataset.resolve()),
        "@DIM@": str(args.dim),
        "@INDEX_ROOT@": str(args.index_root.resolve()),
        "@RESULT_JSON@": str(args.result_json.resolve()),
    }
    rendered = args.template.read_text(encoding="utf-8")
    for marker, value in replacements.items():
        if marker not in rendered:
            raise ValueError(f"template does not contain {marker}")
        rendered = rendered.replace(marker, value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.index_root.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
