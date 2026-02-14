---
name: commit
description: Commit VSAG code changes to a git remote repository with formatting checks, branch management, and signed commits
disable-model-invocation: true
allowed-tools: Bash, Read, AskUserQuestion
---

Commit code changes to a git remote repository:

## Pre-commit Checks

1. **Code Formatting**
   - Run `make fmt` to ensure code follows project formatting standards
   - If formatting produces changes, ask the user whether to include these changes in the commit

2. **Branch Check**
   - Run `git branch --show-current` to get the current branch name
   - If the current branch is `main` or `master`:
     * Run `git status` to check for uncommitted changes
     * Suggest a new branch name based on the commit type and description (e.g., `feat-xxx`, `fix-xxx`)
     * Ask the user to confirm the suggested branch name or provide their own
     * Run `git checkout -b <branch-name>` to create and switch to the new branch

3. **Stage Changes**
   - Run `git status` to view current changes
   - Based on user input `$ARGUMENTS`, determine staging strategy:
     * If `$ARGUMENTS` is empty or contains "all": run `git add -A`
     * If `$ARGUMENTS` contains specific file paths: run `git add <file-paths>`
   - Run `git diff --staged` to review staged changes

## Commit Message Generation

4. **Analyze Changes and Generate Commit Message**
   - Analyze staged changes to determine type:
     * `feat` - New feature
     * `fix` - Bug fix
     * `docs` - Documentation updates
     * `style` - Code style changes (formatting, no functional changes)
     * `refactor` - Code refactoring
     * `test` - Tests related changes
     * `chore` - Build process or tooling changes
   - Generate commit message in format: `type: description` or `type(scope): description`
   - Follow VSAG project commit conventions

5. **Execute Commit**
   - Use the git configured user name and email (do NOT override with hardcoded values)
   - Use `git commit -s` to add DCO sign-off with the user's configured git identity
   - Command format: `git commit -s -m "<commit-message>"`

## Push to Remote

6. **Get Remote Repository Information**
   - Run `git remote -v` to list all configured remotes
   - Do NOT assume the remote is always named "origin"
   - Identify GitHub remotes (URLs containing github.com)
   - If multiple remotes exist, ask the user which one to push to
   - If only one remote exists, use that one but confirm with the user

7. **Push Branch**
   - For a new branch: run `git push -u <remote-name> <branch-name>`
   - For an existing branch: run `git push <remote-name> <branch-name>`
   - Report the commit summary and branch name after successful push

## Required User Confirmations

Use `AskUserQuestion` to get user confirmation for:
- Confirming the commit message content
- Confirming the branch name (if creating a new branch from main/master)
- Selecting and confirming the remote repository to push to
- Confirming the final push operation

## Notes

- The DCO sign-off will use the user's configured git identity (`git config user.name` and `git config user.email`)
- Always verify remote repository availability before attempting to push
- If no remote is configured, prompt the user to add one before proceeding
