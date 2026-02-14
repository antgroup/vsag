---
name: commit
description: Commit VSAG code changes to a git remote repository with formatting checks, branch management, and signed commits
disable-model-invocation: true
allowed-tools: Bash, Read
---

Commit code changes to a git remote repository (non-interactive mode):

## Pre-commit Checks

1. **Code Formatting**
   - Run `make fmt` to ensure code follows project formatting standards
   - Stage any formatting changes automatically

2. **Branch Check**
   - Run `git rev-parse --abbrev-ref HEAD` to get the current branch name
   - If the current branch is `main` or `master` and there are changes:
     * Auto-generate a branch name based on the commit type (e.g., `feat-xxx`, `fix-xxx`, `docs-xxx`)
     * Run `git checkout -b <branch-name>` to create and switch to the new branch

3. **Stage Changes**
   - Stage all changes with `git add -A`

## Commit Message Generation

4. **Analyze Changes and Generate Commit Message**
   - Analyze staged changes to determine type and scope:
     * `feat` - New feature
     * `fix` - Bug fix
     * `docs` - Documentation updates
     * `style` - Code style changes (formatting, no functional changes)
     * `refactor` - Code refactoring
     * `test` - Tests related changes
     * `chore` - Build process or tooling changes
   - Determine scope from changed files (e.g., `hgraph`, `hnsw`, `sindi`, `pyramid`, `ivf`, `quantization`, `simd`, `storage`, `io`, `tests`, `examples`, etc.)
   - Generate commit message in format: `type(scope): description`
   - Check if changes are documentation-only (only .md files or docs/ directory):
     * If yes, prepend `[skip ci]` to commit message: `[skip ci] type(scope): description`
   - Follow VSAG project commit conventions

5. **Execute Commit**
   - Use `git commit -s` to add DCO sign-off with the user's configured git identity
   - Add Co-Authored-By trailer for Claude Code: `Co-Authored-By: Claude <noreply@anthropic.com>`
   - Commit command format:
     ```
     git commit -s -m "<commit-message>" -m "Co-Authored-By: Claude <noreply@anthropic.com>"
     ```

## Push to Remote

6. **Get Remote Repository Information**
   - Run `git remote -v` to list all configured remotes
   - Prefer GitHub remotes (URLs containing github.com):
     * If there's a GitHub remote, use it
     * If multiple GitHub remotes exist, prefer the one pointing to `antgroup/vsag`
     * If no GitHub remote, use the first available remote

7. **Push Branch**
   - For a new branch: run `git push -u <remote-name> <branch-name>`
   - For an existing branch: run `git push <remote-name> <branch-name>`
   - Report the commit summary and branch name after successful push

## Branch Naming Convention

Auto-generated branch names follow this pattern based on commit type:
- `feat-<feature-name>` - New features
- `fix-<bug-description>` - Bug fixes
- `docs-<doc-topic>` - Documentation updates
- `refactor-<refactor-topic>` - Code refactoring
- `test-<test-topic>` - Test changes
- `chore-<chore-topic>` - Build/tooling changes
- `style-<style-topic>` - Code style changes

## Commit Message Format

Format: `[skip ci] <type>(<scope>): <description>`

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `style` - Code style (formatting)
- `refactor` - Code refactoring
- `test` - Tests
- `chore` - Build/tooling

Scopes (select based on changed files):
- `hgraph` - HGraph algorithm
- `hnsw` - HNSW algorithm
- `sindi` - SINDI sparse index
- `pyramid` - Pyramid index
- `ivf` - IVF index
- `quantization` - Quantization methods
- `simd` - SIMD optimizations
- `storage` - Storage layer
- `io` - I/O abstractions
- `datacell` - DataCell components
- `factory` - Factory/creation
- `tests` - Test infrastructure
- `examples` - Example code
- `tools` - Tools and scripts
- `build` - Build system
- `claude` - Claude Code configuration
- `docs` - Documentation

## Notes

- The DCO sign-off will use the user's configured git identity (`git config user.name` and `git config user.email`)
- This skill operates in non-interactive mode and makes automatic decisions
- Always verify remote repository availability before attempting to push
- If no remote is configured, report an error and exit
