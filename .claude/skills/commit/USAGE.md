# Commit Skill Usage Guide

## Overview

The `commit` skill helps you commit VSAG project code changes to a git remote repository following best practices: code formatting, branch management, and signed commits with DCO (Developer Certificate of Origin).

## Installation

The skill is already placed in the project directory at `.claude/skills/commit/`. Claude Code will auto-load it when available.

## Usage

### Basic Usage

```
/commit
```

Stages and commits all current changes (equivalent to `git add -A`).

### Commit Specific Files

```
/commit src/algorithm/hgraph.cpp
```

Only stages and commits the specified file(s).

## Interactive Workflow

### 1. Code Formatting Check

Claude automatically runs `make fmt` to ensure code follows the project's formatting standards. If formatting produces changes, Claude will ask whether to include these changes in the commit.

### 2. Branch Check and Creation

- If you have changes on the `main` or `master` branch, Claude will suggest creating a new branch
- Branch names are auto-generated based on commit type (e.g., `feat-hgraph-optimize`, `fix-memory-leak`)
- You can confirm the suggested name or provide your own

### 3. Commit Message Generation

Claude analyzes your changes and generates a commit message following the Conventional Commits specification:

```
feat(hgraph): add new optimization for vector search

Signed-off-by: Your Name <your.email@example.com>
```

Available commit types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation updates
- `style` - Code style changes (formatting only)
- `refactor` - Code refactoring
- `test` - Test-related changes
- `chore` - Build process or tooling changes

### 4. Remote Repository Selection

Claude does **not** assume the remote is always named "origin". Instead:

- Lists all configured remotes using `git remote -v`
- If multiple remotes exist, asks you to select which one to push to
- If only one remote exists, confirms with you before using it
- If no remotes are configured, prompts you to add one

### 5. Confirmation and Push

Before pushing, Claude displays:
- Complete diff summary
- The generated commit message
- Selected remote and branch

You must confirm before the push is executed.

## Usage Scenarios

### Scenario 1: Committing a New Feature

```
User: /commit
Claude: You are currently on the main branch. Suggested new branch: feat-new-index. Confirm?
User: Yes
Claude: [make fmt output]
Claude: Generated commit message: feat(index): add new index implementation
Claude: Found remotes: origin (github.com), upstream (github.com)
Claude: Which remote to push to?
User: origin
Claude: Confirm commit and push to origin/feat-new-index?
User: Yes
Claude: Successfully pushed to origin/feat-new-index
```

### Scenario 2: Committing Specific Files

```
User: /commit src/algorithm/hnswlib/hnswalg.cpp
Claude: Staging changes to src/algorithm/hnswlib/hnswalg.cpp...
Claude: [proceeds with commit workflow]
```

### Scenario 3: Committing on a Feature Branch

```
User: /commit
Claude: Currently on branch feat-optimize (not main/master), committing directly to this branch.
Claude: [proceeds with commit workflow]
```

## Important Notes

### DCO Sign-off

All commits automatically include the `-s` flag to add a DCO sign-off. The sign-off uses **your configured git identity**:

```bash
git config user.name   # Used for Signed-off-by name
git config user.email  # Used for Signed-off-by email
```

The final commit will look like:

```
feat: description here

Signed-off-by: Your Actual Name <your.actual@email.com>
```

### Code Formatting

`make fmt` runs automatically before committing. Ensure:
- `clang-format` is installed
- The `Makefile` is present and properly configured

### Main Branch Protection

If changes are detected on `main` or `master`, creating a new branch is **mandatory**.

### Remote Repository Flexibility

The skill handles various remote configurations:

```bash
# Single remote - confirms before use
git remote -v
origin  https://github.com/user/vsag.git (fetch)
origin  https://github.com/user/vsag.git (push)

# Multiple remotes - asks you to select
git remote -v
origin   https://github.com/user/vsag.git (fetch)
origin   https://github.com/user/vsag.git (push)
upstream https://github.com/antgroup/vsag.git (fetch)
upstream https://github.com/antgroup/vsag.git (push)

# Named differently - adapts accordingly
git remote -v
github  https://github.com/user/vsag.git (fetch)
github  https://github.com/user/vsag.git (push)
```

## Troubleshooting

### Skill Not Loading

Check if the skill is available:
```
What skills are available?
```

Verify the file exists:
```bash
ls -la .claude/skills/commit/SKILL.md
```

### Push Failed

Common causes:
- **No remote configured**: Add a remote with `git remote add <name> <url>`
- **Permission denied**: Check SSH keys or HTTPS credentials
- **Branch conflict**: Pull latest changes first with `git pull <remote> <branch>`

### Formatting Failed

If `make fmt` fails:
- Verify `clang-format` is installed: `clang-format --version`
- Check that `Makefile` exists in the project root
- Try running `make fmt` manually to see error details

### Missing Git Identity

The skill requires git user configuration:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Or configure locally for this repository:

```bash
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

## Technical Details

### Git Command Sequence

1. `make fmt` - Format code
2. `git branch --show-current` - Get current branch
3. `git status` - Check change status
4. `git add -A` or `git add <files>` - Stage changes
5. `git diff --staged` - Review staged changes
6. `git checkout -b <branch>` - Create new branch (if on main/master)
7. `git commit -s -m "message"` - Commit with DCO sign-off
8. `git remote -v` - List all remotes
9. `git push -u <remote> <branch>` - Push to selected remote

### Branch Naming Convention

Auto-generated branch names follow this pattern:
```
<type>-<brief-description>
```

Examples:
- `feat-hgraph-optimize`
- `fix-memory-leak`
- `docs-api-update`
- `refactor-index-interface`

### Remote Detection Logic

1. Run `git remote -v` to get all remotes
2. Parse remote names and URLs
3. Handle cases:
   - 0 remotes: Prompt user to add a remote
   - 1 remote: Confirm with user, then use it
   - 2+ remotes: Present options via `AskUserQuestion`

## Related VSAG Documentation

See `CLAUDE.md` in the project root for:
- Build system details (`make` commands)
- DCO requirements for VSAG project
- Testing procedures
- Code style guidelines
