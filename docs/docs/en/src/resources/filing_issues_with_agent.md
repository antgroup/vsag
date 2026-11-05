# Filing Issues with an AI Agent

You can use an AI coding agent (Claude Code, OpenCode, or Codex) together with
the VSAG repository's built-in `/create-issue` slash command to draft and
submit a high-quality GitHub issue for VSAG. The agent maps your request onto
the project's issue templates, fills in the required fields, and submits the
issue through GitHub CLI.

This page walks through the end-to-end setup. The canonical workflow that the
agent itself follows lives in
[`.github/agent-prompts/create-issue.md`](https://github.com/antgroup/vsag/blob/main/.github/agent-prompts/create-issue.md);
this page focuses on the user-facing steps.

## Prerequisites

- A GitHub account.
- One of the supported AI coding agents installed and configured locally:
  [Claude Code](https://docs.claude.com/en/docs/claude-code/overview),
  [OpenCode](https://opencode.ai/), or Codex.
- `git` available on your machine.

## 1. Install and sign in to GitHub CLI (`gh`)

First, install `gh` by following the official quickstart for your platform:

<https://docs.github.com/en/github-cli/github-cli/quickstart>

Then sign in from your terminal:

```bash
gh auth login
```

Choose **GitHub.com**, pick an authentication protocol (HTTPS is fine), and
follow the browser prompts to complete sign-in.

## 2. Verify your `gh` login

```bash
gh auth status
```

Confirm that GitHub.com authentication is active before continuing.

## 3. Clone the VSAG repository

```bash
git clone https://github.com/antgroup/vsag.git
cd vsag
```

The `/create-issue` command and its prompt files live inside the repository,
so the agent must be launched from within the `vsag/` working directory to
pick them up.

## 4. Launch your agent inside the repo

From the `vsag/` directory, start one of the supported agents:

```bash
# Claude Code
claude

# OpenCode
opencode

# Codex CLI
codex
```

## 5. Run `/create-issue`

In the agent prompt, invoke the slash command and describe your need in
natural language. For example:

```
/create-issue HGraph build crashes when dim=0; want a clear error instead.
```

The agent will:

1. Pick the most appropriate template under
   [`.github/ISSUE_TEMPLATE/`](https://github.com/antgroup/vsag/tree/main/.github/ISSUE_TEMPLATE).
2. Ask follow-up questions if required fields are missing.
3. Draft the issue body with code/doc references in `path:line` form.
4. Show you the final draft for confirmation.
5. Submit the issue via `gh issue create` once you approve.

You can iterate with the agent freely — ask it to revise wording, add
reproduction steps, switch templates, or attach logs before it submits.

## Tips

- Be specific: include the index type, parameters, dataset shape, error
  message, and platform when filing a bug.
- For feature requests, describe the use case and the expected API or
  behavior. The agent will mirror this into the template's required fields.
- Issues do **not** carry `Signed-off-by:` — DCO applies only to commits.
- If you prefer to drive the workflow without an interactive agent, see the
  shell wrapper at
  [`tools/issue-helper/new-issue.sh`](https://github.com/antgroup/vsag/blob/main/tools/issue-helper/new-issue.sh).

## See also

- [Community](community.md)
- [Contributing to VSAG](../development/contributing.md)
