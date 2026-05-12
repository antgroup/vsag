<!-- agent-hints
canonical: docs/agents/contribution-workflow.md
purpose: Branching, commits, DCO, Assisted-by, PR labels, doc sync rules
key-facts:
  - Use Conventional Commits with DCO sign-off; AI agents must NOT self-sign
  - AI agents: do NOT use `git commit -s` (it appends Signed-off-by last, with a blank line before it). Write both trailers manually in the commit body, deriving the human Signed-off-by from `git config user.name`/`user.email`
  - Trailer order: Signed-off-by first, Assisted-by immediately after, no blank line between them
  - Every PR needs both a kind/* and a version/* label
related:
  - ../../CONTRIBUTING.md
  - ../../.github/pull_request_template.md
last-reviewed: 2026-05-12
-->

# Contribution Workflow

## Branching

- Before modifying code, create and switch to a working branch from an
  up-to-date `main`.
- Keep changes scoped and reviewable.

## Commit messages

- Follow Conventional Commits: `feat:`, `fix:`, `docs:`, `chore:`,
  `refactor:`, etc.
- Commits in this project are expected to carry a DCO `Signed-off-by:`
  trailer.
- **Only humans can legally certify the DCO; AI coding agents must not add
  their own `Signed-off-by` trailer.** The human submitter is responsible for
  reviewing AI-generated changes, ensuring license compliance, and taking
  full responsibility for the contribution.
- **AI agents must NOT invoke `git commit -s`.** `-s` appends a
  `Signed-off-by:` trailer at the very end of the message and inserts a
  blank line before it if other trailers are present, which separates the
  human sign-off from the `Assisted-by:` line and reverses the required
  order. Instead, write both trailers directly in the commit message body.
  Derive the human identity from the active git configuration
  (`git config user.name` and `git config user.email`) and format it as
  `Signed-off-by: <Name> <email>`.
- For changes produced with AI assistance, attribute the agent with an
  `Assisted-by:` trailer in the form `Assisted-by: AgentName:ModelVersion`
  (see the [Linux kernel AI Coding Assistants policy](https://docs.kernel.org/process/coding-assistants.html)).
  Determine `AgentName` and `ModelVersion` at execution time from the active
  agent name and model name (for example
  `Assisted-by: OpenCode:claude-opus-4.7`).
- **Trailer order and spacing**: a single blank line separates the subject
  (and optional body) from the trailer block. Inside the trailer block,
  place the human `Signed-off-by:` first and the `Assisted-by:` line
  immediately after, with **no blank line between them**. Example:

  ```
  docs: reorganize agent guidance

  Signed-off-by: Jane Doe <jane@example.com>
  Assisted-by: OpenCode:claude-opus-4.7
  ```
- When using skip-CI commit messages, follow the repository convention and
  place `[skip ci]` at the **beginning** of the subject line.

## Pull request labels

Every pull request **must** have two labels before it can be merged:

- A `kind/*` label for the type of change: `kind/bug` (bug fix),
  `kind/feature` (new feature), `kind/improvement` (refactor, chore, or minor
  improvement), or `kind/documentation` (documentation change).
- A `version/*` label for the target version (e.g. `version/1.0`,
  `version/0.18`).

Mergify enforces these via check runs. Always add both labels when creating
a PR.

## Documentation expectations

Update relevant docs when behavior changes:

- `README.md` for user-facing features or examples.
- `DEVELOPMENT.md` for build or environment changes.
- `CONTRIBUTING.md` for workflow or contribution policy changes.
- Website docs under `docs/docs/{en,zh}/src/` together with any related
  in-repo READMEs (`tools/eval/README.md`, `tools/eval/README_zh.md`, etc.).
  Keep English and Chinese versions in sync.
