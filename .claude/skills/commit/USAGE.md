# Commit Skill Usage Guide

## Overview

The `commit` skill helps you commit VSAG project code changes to a git remote repository following best practices: code formatting, branch management, and signed commits with DCO (Developer Certificate of Origin).

**Note:** This skill operates in **non-interactive mode** - all decisions are made automatically based on conventions.

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

## Automatic Workflow (Non-Interactive)

### 1. Code Formatting Check

Claude automatically runs `make fmt` to ensure code follows the project's formatting standards. Any formatting changes are staged automatically.

### 2. Branch Check and Creation

- If you have changes on the `main` or `master` branch, Claude automatically creates a new branch
- Branch names are auto-generated based on the commit type (e.g., `feat-hgraph-optimize`, `fix-memory-leak`)
- No confirmation needed - branch is created immediately

### 3. Commit Message Generation

Claude analyzes your changes and generates a commit message following the VSAG project format:

```
type(scope): description

Signed-off-by: Your Name <your.email@example.com>
```

For documentation-only changes, `[skip ci]` is prepended:

```
[skip ci] docs(claude): update commit skill documentation

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

Available scopes (based on changed files):
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

### 4. Remote Repository Selection

Claude automatically selects the remote to push to:
- **Prefers GitHub remotes** (URLs containing github.com)
- If multiple GitHub remotes exist, prefers `antgroup/vsag`
- If no GitHub remote, uses the first configured remote
- No confirmation needed - push happens automatically

### 5. Push

The branch is pushed to the selected remote immediately without confirmation.

## Usage Scenarios

### Scenario 1: Committing a New Feature

```
User: /commit
Claude: Automatically created branch: feat-optimize-search
Claude: [make fmt output]
Claude: Commit message: feat(hgraph): optimize vector search performance
Claude: Pushing to github-vsag/feat-optimize-search
Claude: Successfully pushed. Create PR at: https://github.com/antgroup/vsag/pull/new/feat-optimize-search
```

### Scenario 2: Committing Documentation

```
User: /commit
Claude: Automatically created branch: docs-update-guide
Claude: [make fmt output]
Claude: Commit message: [skip ci] docs(claude): update commit skill guide
Claude: Pushing to github-vsag/docs-update-guide
Claude: Successfully pushed. Create PR at: https://github.com/antgroup/vsag/pull/new/docs-update-guide
```

### Scenario 3: Committing on a Feature Branch

```
User: /commit
Claude: Already on branch feat-optimize, committing directly.
Claude: [make fmt output]
Claude: Commit message: refactor(hgraph): extract common search logic
Claude: Pushing to github-vsag/feat-optimize
Claude: Successfully pushed.
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
feat(hgraph): description here

Signed-off-by: Your Actual Name <your.actual@email.com>

Co-Authored-By: Claude <noreply@anthropic.com>
```

### [skip ci] Prefix

Documentation-only commits (only `.md` files or files in `docs/` directory) automatically get `[skip ci]` prepended to skip CI builds.

### Code Formatting

`make fmt` runs automatically before committing. Ensure:
- `clang-format` is installed
- The `Makefile` is present and properly configured

### Main Branch Protection

If changes are detected on `main` or `master`, a new branch is automatically created.

### Remote Repository Selection

The skill automatically handles various remote configurations:

```bash
# Single GitHub remote - used automatically
git remote -v
origin  https://github.com/antgroup/vsag.git (fetch)
origin  https://github.com/antgroup/vsag.git (push)

# Multiple remotes - GitHub is preferred
git remote -v
origin   https://code.alipay.com/octopus/vsag.git (fetch)
github   https://github.com/antgroup/vsag.git (fetch)  <- This one is chosen

# Different GitHub organization - antgroup/vsag is preferred
git remote -v
fork     https://github.com/yourname/vsag.git (fetch)
upstream https://github.com/antgroup/vsag.git (fetch)  <- This one is chosen
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
2. `git rev-parse --abbrev-ref HEAD` - Get current branch (compatible with older git versions)
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
3. Priority:
   - GitHub remote with `antgroup/vsag` (highest)
   - Any GitHub remote (medium)
   - First available remote (lowest)
4. Push automatically

### Commit Message Format

```
[skip ci] <type>(<scope>): <description>
```

The `[skip ci]` prefix is added for documentation-only changes.

## Related VSAG Documentation

See `CLAUDE.md` in the project root for:
- Build system details (`make` commands)
- DCO requirements for VSAG project
- Testing procedures
- Code style guidelines
