# /create-issue (Codex)

Codex CLI does not have a built-in slash-command registry that mirrors Claude
Code or OpenCode. To trigger this workflow with Codex, run:

```bash
codex exec --prompt-file .github/agent-prompts/create-issue.md
```

Or, from inside an interactive `codex` session:

```
@.github/agent-prompts/create-issue.md
```

When the prompt instructs you to append the AI footer, use
`Codex:<model>` as `<AgentName>:<ModelVersion>`.

The shared workflow lives in
[`.github/agent-prompts/create-issue.md`](../../.github/agent-prompts/create-issue.md)
and is the single source of truth across all supported agents.
