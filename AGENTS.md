# AGENTS.md

Operational rules for AI agents working in this repository. Read
`GIT_POLICY.md` for the full rationale; this file is the short enforceable list.

## Git

- **Never commit unless explicitly asked.** Stage nothing proactively beyond what
  the current task requires. If asked to commit, commit — do not ask again.
- **Never** `git push`, `git commit --amend` on pushed commits, `git rebase` on
  pushed history, `git reset --hard`, `git checkout -- .`, or `git clean -fd`.
  These destroy work. If one seems necessary, stop and ask.
- **Never force-push to `main`.**
- **Do not** modify `git config`, install hooks, or change the default branch
  without being asked. One standing exception: the repo-local `user.name` /
  `user.email` may be set when the user supplies the values or asks for it.
  Never write identity to `--global`, and never change `core.*` settings.
- Before committing, always run `git status` and `git diff --staged`, and read
  them.
- Stage files individually or by explicit path. Avoid `git add -A` /
  `git add .` — they sweep in build artifacts.
- Commit messages follow Conventional Commits and must explain *why*, not just
  *what*. Cite the affected `FINDINGS.md` section.

## Verification

- Run `python -m ruff check src/` for any file you touch. New warnings are not
  acceptable; three pre-existing ones in `download.py` / `predict.py` are known.
- There is no test suite. If you change a script that produces a number in
  `FINDINGS.md`, re-run it and confirm the printed output matches the document.
  **Never update a number in `FINDINGS.md` without re-running the script that
  produces it.**
- If a claim in `FINDINGS.md` does not survive re-verification, correct the
  document. Do not leave a stale figure in place, and do not quietly drop a
  section — annotate superseded text so the correction is visible.

## Data

- `data/raw/*.json` is the pinned source snapshot. Do not re-download or
  "refresh" it without being asked; upstream revisions would invalidate every
  reported figure.
- `data/raw/*.csv` are corrupted WAF artifacts, not data. Do not read them, and
  do not restore them to version control.
- Never delete or overwrite `data/processed/*.parquet` or `output/` as part of
  unrelated work. They are gitignored build outputs; regenerating them is fine,
  destroying them silently is not.

## Scope

- Do not create documentation files beyond what the task requires.
- Prefer editing existing files over creating new ones.
- Do not add comments to code unless asked.
- Report findings that contradict the user's premise rather than proceeding
  silently.
