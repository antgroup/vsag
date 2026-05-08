#!/usr/bin/env bash
# new-issue.sh — draft and submit a VSAG GitHub issue with the help of a
# locally installed coding agent (Claude Code / OpenCode / Codex) and `gh`.
#
# Usage:
#   tools/issue-helper/new-issue.sh [--url URL]... [--file PATH]...
#                                   [--intent "..."] [--kind KIND]
#                                   [--agent AGENT] [--include-env]
#
#   --url URL         A vsag.io (or other) doc page to attach as context.
#                     May be repeated.
#   --file PATH       A repository file to attach as context. May be repeated.
#   --intent TEXT     One-sentence description of what you want.
#   --kind KIND       bug | feature | improvement | documentation | question
#                     (informational; the agent ultimately picks the template)
#   --agent AGENT     Force a specific agent: opencode | claude | codex
#                     Default: $VSAG_ISSUE_AGENT, else first one found.
#   --include-env     Run scripts/check_environment.sh and attach the output.
#                     Implied when --kind=bug.
#   -h, --help        Show this help.
#
# The script collects the supplied context into a temporary bundle and hands
# control to the selected agent, which loads .github/agent-prompts/create-issue.md
# and produces a `gh issue create` invocation after a dry-run preview.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROMPT_FILE="$REPO_ROOT/.github/agent-prompts/create-issue.md"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "error: prompt file not found at $PROMPT_FILE" >&2
  exit 1
fi

AGENT="${VSAG_ISSUE_AGENT:-}"
INTENT=""
KIND=""
INCLUDE_ENV=0
URLS=()
FILES=()

usage() { sed -n '2,/^$/p' "$0"; }

while (($#)); do
  case "$1" in
    --url)         URLS+=("$2"); shift 2 ;;
    --file)        FILES+=("$2"); shift 2 ;;
    --intent)      INTENT="$2"; shift 2 ;;
    --kind)        KIND="$2"; shift 2 ;;
    --agent)       AGENT="$2"; shift 2 ;;
    --include-env) INCLUDE_ENV=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

[[ "$KIND" == "bug" ]] && INCLUDE_ENV=1

# --- agent autodetect ------------------------------------------------------
detect_agent() {
  for cand in opencode claude codex; do
    if command -v "$cand" >/dev/null 2>&1; then
      echo "$cand"; return 0
    fi
  done
  return 1
}

if [[ -z "$AGENT" ]]; then
  AGENT="$(detect_agent || true)"
fi
if [[ -z "$AGENT" ]]; then
  echo "error: no supported agent found (opencode/claude/codex). Install one or set --agent." >&2
  exit 1
fi
if ! command -v "$AGENT" >/dev/null 2>&1; then
  echo "error: requested agent '$AGENT' not found in PATH." >&2
  exit 1
fi
command -v gh >/dev/null 2>&1 || {
  echo "error: GitHub CLI 'gh' is required." >&2; exit 1; }

# --- gather context --------------------------------------------------------
BUNDLE_DIR="$(mktemp -d -t vsag-issue.XXXXXX)"
trap 'rm -rf "$BUNDLE_DIR"' EXIT
CTX_FILE="$BUNDLE_DIR/context.md"

{
  echo "# Context bundle for VSAG issue drafting"
  echo
  [[ -n "$INTENT" ]] && { echo "## User intent"; echo; echo "$INTENT"; echo; }
  [[ -n "$KIND" ]]   && { echo "## Suggested template"; echo; echo "\`$KIND\`"; echo; }

  for f in "${FILES[@]:-}"; do
    [[ -z "$f" ]] && continue
    # Resolve against $REPO_ROOT so users can pass repo-relative paths even
    # when invoking the script from outside the repository root.
    resolved="$f"
    if [[ ! -f "$resolved" && -f "$REPO_ROOT/$f" ]]; then
      resolved="$REPO_ROOT/$f"
    fi
    if [[ -f "$resolved" ]]; then
      echo "## File: \`$f\`"
      echo
      echo '```'
      cat -- "$resolved"
      echo '```'
      echo
    else
      echo "## File: \`$f\` (NOT FOUND)"
      echo
    fi
  done

  for u in "${URLS[@]:-}"; do
    [[ -z "$u" ]] && continue
    echo "## URL: $u"
    echo
    if command -v curl >/dev/null 2>&1; then
      echo '```html'
      curl -fsSL --max-time 30 "$u" || echo "(failed to fetch)"
      echo '```'
    else
      echo "(curl not available; agent should fetch this URL itself)"
    fi
    echo
  done

  if (( INCLUDE_ENV )); then
    echo "## Environment (\`scripts/check_environment.sh\`)"
    echo
    echo '```'
    if [[ -x "$REPO_ROOT/scripts/check_environment.sh" ]]; then
      bash "$REPO_ROOT/scripts/check_environment.sh" 2>&1 || true
    else
      echo "(scripts/check_environment.sh not found or not executable)"
    fi
    echo '```'
    echo
  fi
} > "$CTX_FILE"

USER_MSG_FILE="$BUNDLE_DIR/user-message.md"
{
  echo "Please draft a VSAG GitHub issue following the instructions in"
  echo ".github/agent-prompts/create-issue.md."
  echo
  echo "Context bundle is below. Use it as the primary source. You may also"
  echo "read additional repo files or fetch additional URLs if needed."
  echo
  cat "$CTX_FILE"
} > "$USER_MSG_FILE"

echo "→ context bundle written to: $CTX_FILE"
echo "→ launching agent: $AGENT"
echo

# --- dispatch to the chosen agent -----------------------------------------
# Pipe the user message via stdin (or pass a file path) instead of stuffing
# the entire bundle into argv: large context bundles can easily exceed ARG_MAX.
case "$AGENT" in
  opencode)
    # opencode reads a prompt from stdin when no positional arg is given.
    opencode run < "$USER_MSG_FILE"
    ;;
  claude)
    # claude --print accepts the user message on stdin; the shared system
    # prompt is appended via --append-system-prompt-file when supported,
    # otherwise inlined (size-bounded; the prompt file itself is small).
    if claude --help 2>&1 | grep -q -- '--append-system-prompt-file'; then
      claude --print --append-system-prompt-file "$PROMPT_FILE" < "$USER_MSG_FILE"
    else
      claude --print --append-system-prompt "$(cat "$PROMPT_FILE")" < "$USER_MSG_FILE"
    fi
    ;;
  codex)
    # Prefer file-based flags; fall back to stdin if codex does not recognise
    # --user-message-file on the installed version.
    codex exec --prompt-file "$PROMPT_FILE" --user-message-file "$USER_MSG_FILE" \
      || codex exec --prompt-file "$PROMPT_FILE" - < "$USER_MSG_FILE"
    ;;
  *)
    echo "error: unsupported agent '$AGENT'" >&2
    exit 1
    ;;
esac
