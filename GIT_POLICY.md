# Git Policy

How version control is used in this repository. Written 2026-09-27, when the
project was initialized.

## 1. What is tracked

| Path | Tracked | Why |
|---|---|---|
| `src/*.py` | yes | The analysis itself. |
| `FINDINGS.md` | yes | The research output. Every number here must be traceable to a script. |
| `data/raw/*.json` | yes | ~40MB of gov.il source data, pinned deliberately. |
| `data/raw/manifest.json` | yes | Source resource IDs for each dataset. |
| `data/processed/*.csv` | yes | Small audit tables (`parties.csv`, `locality_demographics.csv`, `cv_summary.csv`, `letter_totals_validation.csv`) — the checkable evidence behind FINDINGS.md. |
| `data/processed/*.parquet` | no | ~18MB, fully derived. Regenerate via the pipeline. |
| `output/` | no | ~1.3MB of derived forecasts. |
| `data/raw/*.csv` | no | Blocked-download WAF artifacts, not data. See §6. |
| `.venv/`, `__pycache__/`, `.ruff_cache/` | no | Environment and caches. |

Raw JSON is committed rather than ignored on purpose: the Central Bureau of
Statistics revises published figures, so a re-download can silently change
every number in FINDINGS.md. Pinning the snapshot makes that visible as a diff
instead of an invisible drift. The cost is a ~40MB repository, accepted for a
research project where reproducibility outranks clone speed.

## 2. Branches

- `main` is the default branch and holds the state described by `FINDINGS.md`.
- Branch for any non-trivial change: `feat/<short-slug>`, `fix/<short-slug>`.
  Examples: `feat/leave-one-election-out-cv`, `fix/other-family-residual`.
- Short-lived. Merge or delete within a few days; there is no long-running
  integration branch, because there is no CI gate to protect.
- Exploratory spikes that do not change any reported number may go straight to
  `main` in a single `wip:` commit that is amended or squashed before the next
  real change.

## 3. Commits

[Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <imperative summary, <=72 chars>

<body: what changed and, more importantly, why>
```

Types in use here: `feat`, `fix`, `refactor`, `perf`, `docs`, `data`, `chore`,
`wip`.

Author identity is set in this repository's local config only:

```
user.name  = Laufer Zur
user.email = 129779217+zurlaufer@users.noreply.github.com
```

It is deliberately not written to `--global`, so it does not affect any other
project on this machine. The address is a GitHub `noreply` form, which keeps
commit authorship attributable without exposing a real inbox — appropriate if
this repository is later pushed to GitHub.

Rules:

- **One logical change per commit.** Code that changes a number in
  `FINDINGS.md` belongs in the same commit as the edit to that table.
- **The body must state the finding, not the diff.** A reviewer should learn
  what was learned. Cite the affected `FINDINGS.md` section (`§8`, `§5a`).
- **Never amend or rebase a commit that has been pushed.** Add a fixup commit.
  Force-pushing to `main` is not permitted.
- **Do not commit generated files you did not intend to pin.** If a re-run
  changes a tracked CSV, that diff belongs in a `data:` commit explaining why
  the inputs changed — not smuggled into a `feat:` commit.

### Verified in the initial history

- `chore: initialize repository, pin raw data snapshot, add git policy` —
  the pre-existing analysis imported as a single baseline commit. The original
  working order was never recorded, so it was not reconstructed; only the split
  between this baseline and the CV work that followed it is meaningful.
- `feat: add leave-one-election-out CV and differential-swing validation` —
  adds `src/cv.py`, `src/swing.py`, `cv_summary.csv`, and `FINDINGS.md` §8–§9.

## 4. Before committing

1. `python -m ruff check src/` — must be clean for files you touched. Three
   pre-existing warnings in `download.py` and `predict.py` are known and may be
   left alone; do not introduce new ones.
2. Re-run any script whose outputs you changed, and confirm the printed numbers
   match the table you edited in `FINDINGS.md`.
3. `git status` — confirm nothing under `.venv/`, `__pycache__/`, `output/`, or
   `data/processed/*.parquet` is staged.
4. `git diff --staged` — read it. Large binary additions should be intentional.

There is no test suite; the pipeline scripts are the check. If you add one, wire
it into this section.

## 5. FINDINGS.md is a contract

`FINDINGS.md` cites specific figures. When code changes, the document must be
updated **in the same commit**, and any figure that is no longer reproducible
must be deleted or marked, not quietly left behind. Section 8 exists precisely
because section 6's claim rested on one held-out election; that correction is
the model to follow.

When a finding is superseded, annotate the old text rather than deleting it
(see the caveat under §6 in `FINDINGS.md`). The history of what was believed and
when is part of the value of the document.

## 6. Known data problems

- **`data/raw/*.csv` are not data.** All six are ~42KB WAF anti-bot challenge
  pages returned by data.gov.il, each with a different random seed. No script
  reads them. They are gitignored and safe to delete locally.
- **`data/raw/manifest.json` has no checksums.** It records resource IDs and
  election dates only. Since the raw JSON is now pinned in git, drift from
  upstream is detectable by comparing against the committed files, but
  `download.py` will not warn on a re-fetch. Adding SHA256 verification to
  `download.py` is the outstanding fix; it is not yet implemented.
- **`download.py` skips existing files silently** (`if dest.exists()`). To
  force a clean re-fetch, delete the target file first.

## 7. Secrets

None currently, and none expected: all inputs are public government data. If
that ever changes, credentials go in the environment, never in a tracked file,
and `.gitignore` gains the pattern before the first commit that could leak them.
