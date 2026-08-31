#!/usr/bin/env python3

"""Check named struct definitions in tracked project C/C++ files."""

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys


CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".inl"}
CAMEL_CASE = re.compile(r"^[A-Z][A-Za-z0-9]*$")
EXCLUDED_PREFIXES = ("extern/", "vendor/", "third_party/", "build/", "build-")
EXCLUDED_PARTS = {"generated", "hnsw", "diskann"}
EXCLUDED_FILES = {"include/vsag/expected.hpp"}
TOKEN = re.compile(
    r"""(?P<space>\s+)|(?P<line_comment>//[^\n]*)|(?P<block_comment>/\*.*?\*/)|
        (?P<raw_string>R\"(?P<delimiter>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=delimiter)\")|
        (?P<string>\"(?:\\.|[^\"\\])*\")|(?P<char>'(?:\\.|[^'\\])*')|
        (?P<identifier>[A-Za-z_][A-Za-z0-9_]*)|(?P<symbol>\[\[|\]\]|\#\#|::|.)""",
    re.DOTALL | re.VERBOSE,
)


@dataclass(frozen=True)
class Token:
    text: str
    line: int
    identifier: bool


def tokenize(source: str) -> list[Token]:
    tokens = []
    line = 1
    for match in TOKEN.finditer(source):
        text = match.group(0)
        kind = match.lastgroup
        if kind not in {"space", "line_comment", "block_comment", "raw_string", "string", "char"}:
            tokens.append(Token(text, line, kind == "identifier"))
        line += text.count("\n")
    return tokens


def skip_balanced(tokens: list[Token], index: int, opening: str, closing: str) -> int:
    depth = 0
    while index < len(tokens):
        if tokens[index].text == opening:
            depth += 1
        elif tokens[index].text == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    return index


def struct_definitions(source: str) -> list[tuple[str, int]]:
    tokens = tokenize(source)
    definitions = []
    index = 0
    while index < len(tokens):
        if tokens[index].text != "struct":
            index += 1
            continue

        cursor = index + 1
        while cursor < len(tokens):
            if tokens[cursor].text in {"alignas", "__attribute__"} and cursor + 1 < len(tokens):
                cursor = skip_balanced(tokens, cursor + 1, "(", ")")
                continue
            if tokens[cursor].text == "[[":
                cursor = skip_balanced(tokens, cursor, "[[", "]]")
                continue
            break
        if cursor >= len(tokens) or not tokens[cursor].identifier:
            index += 1
            continue

        name = tokens[cursor]
        cursor += 1
        if cursor < len(tokens) and tokens[cursor].text == "##":
            name_parts = [name.text]
            while (
                cursor + 1 < len(tokens)
                and tokens[cursor].text == "##"
                and tokens[cursor + 1].identifier
            ):
                name_parts.append(tokens[cursor + 1].text)
                cursor += 2
            name = Token("".join(name_parts), name.line, True)
        elif cursor < len(tokens) and tokens[cursor].text == "<":
            cursor = skip_balanced(tokens, cursor, "<", ">")
        elif cursor < len(tokens) and tokens[cursor].identifier and tokens[cursor].text != "final":
            index += 1
            continue

        while cursor < len(tokens) and tokens[cursor].text not in {"{", ";"}:
            cursor += 1
        if cursor < len(tokens) and tokens[cursor].text == "{":
            definitions.append((name.text, name.line))
        index += 1
    return definitions


def is_project_file(path: str) -> bool:
    if Path(path).suffix not in CPP_SUFFIXES or path in EXCLUDED_FILES:
        return False
    if path.startswith(EXCLUDED_PREFIXES):
        return False
    return not any(part.lower() in EXCLUDED_PARTS for part in Path(path).parts)


def tracked_files(root: Path) -> list[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=root, text=True)
    return [path for path in output.splitlines() if is_project_file(path)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="files to check instead of the tracked project scope")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    files = args.files or tracked_files(root)
    failures = []
    for filename in files:
        path = Path(filename)
        if not path.is_absolute():
            path = root / path
        for name, line in struct_definitions(path.read_text(errors="surrogateescape")):
            if not CAMEL_CASE.fullmatch(name):
                failures.append(f"{path.relative_to(root)}:{line}: struct '{name}' is not CamelCase")
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Checked {len(files)} C/C++ files: all named struct definitions use CamelCase.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
