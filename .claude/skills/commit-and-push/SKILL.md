---
name: commit-and-push
description: Commit all changes by topic and push to remote
---

Commit all staged and unstaged changes, grouped by topic into separate commits, then push to the remote.

## Steps

1. Run `git status` to see all modified, staged, and untracked files.
2. Run `git diff` and `git diff --cached` to understand the changes.
3. Run `git log --oneline -5` to see the existing commit message style.
4. If any Rust files (`.rs`, `Cargo.toml`, `Cargo.lock`) are among the changes, run the following checks from the Rust project directory (e.g., `tools/gensfen/`) and fix any issues before committing:
   - `cargo fmt --check` — code formatting
   - `cargo clippy -- -D warnings` — lint warnings
   - `cargo test` — unit tests
   - `cargo audit` — known vulnerabilities in dependencies
   If any check fails, fix the issue and re-run until all pass.
5. Group related files into logical topics (e.g., Rust tool changes, Python changes, docs, config).
6. For each topic group, stage the relevant files and create a commit with:
   - A short summary line (imperative mood, under 72 chars)
   - An optional body with details if the change is non-trivial
   - A `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>` trailer
   - Use a HEREDOC for the commit message to ensure correct formatting
7. After all commits are created, run `git log --oneline origin/main..HEAD` (or the appropriate remote tracking branch) to list the commits that will be pushed.
8. Show the user the list of commits and ask for confirmation before pushing.
9. If confirmed, run `git push` to push all commits to the remote.

## Rules

- Never use `git add -A` or `git add .` — always add specific files by name.
- Do not commit files that likely contain secrets (.env, credentials, tokens).
- If there are no changes to commit, inform the user and stop.
- If there is no remote configured, inform the user and stop after committing.
