#!/usr/bin/env python3
"""
STAGE 2 v3 — SZZ BUG LABELLER (STRICT PATTERNS + TEST-FILE AWARENESS)
AMARE DERGU MENGSTIE | 2026

═══════════════════════════════════════════════════════════════════════
  WHAT CHANGED FROM v2
═══════════════════════════════════════════════════════════════════════

  v2 → v3 (keyword precision):
    Used 15 broad keywords → file-level defect rate up to 78% on pandas.
    Now uses 5 high-precision keywords + expanded exclusion filter.
    Expected file-level rate: 8–30%.

  v3 now also fixes:

  FIX 1 — CSV output alongside parquet (same stem name):
    Every output parquet now also writes a .csv for inspection.

  FIX 2 — Test-file-aware health report:
    The file-level defect rate was misleadingly high because test files
    (tests/, test_*.py) are touched by almost every bug fix — SZZ then
    labels the previous commit on those test files as bug-inducing even
    though no production bug was introduced.
    The health report now splits rates for production files vs test files
    separately, so the diagnostic is honest and actionable.

  FIX 3 — is_test_file column added to output:
    Adds boolean column so Stage 4 ML can easily filter or stratify.

  FIX 4 — Health gate threshold corrected:
    Old gate used file-level rate across ALL files including test files.
    New gate uses PRODUCTION file-level rate only (< 5% or > 35%).
    This gives a realistic health signal for numpy (59% total → ~20% prod).

  UNCHANGED:
    Two-pass SZZ architecture, keyword patterns, exclusion filter,
    output schema (+ 2 new columns: is_test_file, is_production_file).

READS FROM : data/02_snapshots/<repo>_2020_2025_bimonthly_CLEAN.parquet
             OR a TOP-N subset parquet from create_test_subset.py
WRITES TO  : data/03_labeled/<stem>_LABELED.parquet
             data/03_labeled/<stem>_LABELED.csv
"""

import re
import pandas as pd
from pathlib import Path
from collections import defaultdict
from pydriller import Repository
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
#
#  To run on the TOP-50 test subset:
#    Set input to: data/03_labeled/django_2020_2025_FINAL_LABELED_TOP50.parquet
#    (produced by create_test_subset.py — note: still needs the FULL git repo
#     for SZZ traversal, because SZZ reads commit history not just the parquet)
#
#  To run on the full dataset:
#    Set input to: data/02_snapshots/django_2020_2025_bimonthly_CLEAN.parquet
#
# ══════════════════════════════════════════════════════════════════════════════

REPOS = [
    # ── TOP-50 test subset (commented out — full repos active) ────────────────
    # {
    #     "input":    "data/03_labeled/django_2020_2025_FINAL_LABELED_TOP50.parquet",
    #     "git_repo": "data/01_raw_data/django",
    #     "repo_name":"django",
    #     "output":   "data/03_labeled/django_2020_2025_FINAL_LABELED_TOP50_LABELED.parquet",
    # },
    # ── FULL REPOS ────────────────────────────────────────────────────────────
    # {
    #     "input":    "data/02_snapshots/django_2020_2025_bimonthly_CLEAN.parquet",
    #     "git_repo": "data/01_raw_data/django",
    #     "repo_name":"django",
    #     "output":   "data/03_labeled/django_2020_2025_FINAL_LABELED.parquet",
    # },
    # {
    #     "input":    "data/02_snapshots/fastapi_2020_2025_bimonthly_CLEAN.parquet",
    #     "git_repo": "data/01_raw_data/fastapi",
    #     "repo_name":"fastapi",
    #     "output":   "data/03_labeled/fastapi_2020_2025_FINAL_LABELED.parquet",
    # },
    # {
    #     "input":    "data/02_snapshots/numpy_2020_2025_bimonthly_CLEAN.parquet",
    #     "git_repo": "data/01_raw_data/numpy",
    #     "repo_name":"numpy",
    #     "output":   "data/03_labeled/numpy_2020_2025_FINAL_LABELED.parquet",
    # },
    {
        "input":    "data/02_snapshots/scikit_learn_2020_2025_bimonthly_CLEAN.parquet",
        "git_repo": "data/01_raw_data/scikit-learn",
        "repo_name":"scikit-learn",
        "output":   "data/03_labeled/scikit_learn_2020_2025_FINAL_LABELED.parquet",
    },
    {
        "input":    "data/02_snapshots/flask_2020_2025_bimonthly_CLEAN.parquet",
        "git_repo": "data/01_raw_data/flask",
        "repo_name":"flask",
        "output":   "data/03_labeled/flask_2020_2025_FINAL_LABELED.parquet",
    },
    {
        "input":    "data/02_snapshots/matplotlib_2020_2025_bimonthly_CLEAN.parquet",
        "git_repo": "data/01_raw_data/matplotlib",
        "repo_name":"matplotlib",
        "output":   "data/03_labeled/matplotlib_2020_2025_FINAL_LABELED.parquet",
    },
    # {
    #     "input":    "data/02_snapshots/sqlalchemy_2020_2025_bimonthly_CLEAN.parquet",
    #     "git_repo": "data/01_raw_data/sqlalchemy",
    #     "repo_name":"sqlalchemy",
    #     "output":   "data/03_labeled/sqlalchemy_2020_2025_FINAL_LABELED.parquet",
    # },
]

# ══════════════════════════════════════════════════════════════════════════════
#  TEST-FILE DETECTION
#  A file is a test file if its path contains a test directory or its name
#  matches the test naming convention used across all four repos.
# ══════════════════════════════════════════════════════════════════════════════

_TEST_PATH_RE = re.compile(
    r"(^|[/\\])(tests?)[/\\]"          # any /test/ or /tests/ directory
    r"|[/\\]test_[^/\\]+\.py$"         # /test_something.py
    r"|[/\\][^/\\]+_test\.py$"         # /something_test.py
    r"|(^|[/\\])conftest\.py$",        # conftest.py (pytest fixtures)
    re.IGNORECASE,
)


def is_test_file(file_path: str) -> bool:
    """Return True if the file path looks like a test file."""
    return bool(_TEST_PATH_RE.search(file_path.replace("\\", "/")))


# ══════════════════════════════════════════════════════════════════════════════
#  STRICT BUG-FIX CLASSIFIER
#  5 high-precision keywords — see WHY THIS VERSION EXISTS above.
# ══════════════════════════════════════════════════════════════════════════════

BUG_PATTERNS = re.compile(
    r"\bfix(es|ed|ing)?\b"    # fix, fixes, fixed, fixing
    r"|\bbug\b"               # bug (word boundary — not "debug")
    r"|\bregression\b"        # regression
    r"|\bcrash(es|ed)?\b"     # crash, crashes, crashed
    r"|\bdefect\b",           # defect
    re.IGNORECASE,
)

EXCLUDE_PATTERNS = re.compile(
    # Test-related (adding tests is not fixing a bug)
    r"\b(add|new|write|update|improve|create)\s+(test|tests|spec|specs|fixture|fixtures)\b"
    r"|\btest\s+only\b"
    r"|\bunit\s+test\b"
    # Refactoring
    r"|\brefactor\b"
    r"|\brestructure\b"
    r"|\breorganize\b"
    r"|\breorg\b"
    # Documentation and style
    r"|\btypo\b"
    r"|\bspelling\b"
    r"|\bgrammar\b"
    r"|\bformat(ting)?\b"
    r"|\bdoc(s|string|strings|umentation)?\b"
    r"|\bcomment\b"
    r"|\breadme\b"
    r"|\bchangelog\b"
    r"|\bwhitespace\b"
    r"|\bindent(ation)?\b"
    r"|\blint(ing)?\b"
    r"|\bpep\s*8\b"
    r"|\bcleanup\b"
    r"|\bclean\s+up\b"
    # Version / release management
    r"|\bversion\s+bump\b"
    r"|\brelease\s+\d"
    r"|\bprepare\s+release\b"
    r"|\bv\d+\.\d+\b"
    # Merge / CI
    r"|\bmerge\s+(branch|pull|pr|conflict)\b"
    r"|\bci\b"
    r"|\bpipeline\b",
    re.IGNORECASE,
)


def is_bug_fix(msg: str) -> bool:
    """Return True only for high-confidence bug-fix commits."""
    if not msg:
        return False
    if EXCLUDE_PATTERNS.search(msg):
        return False
    return bool(BUG_PATTERNS.search(msg))


# ══════════════════════════════════════════════════════════════════════════════
#  SZZ CORE (two-pass, unchanged from v2)
# ══════════════════════════════════════════════════════════════════════════════

def run_szz(git_repo: str, repo_name: str) -> set:
    """
    Two-pass SZZ implementation.

    Pass 1: traverse FULL git history (not restricted to 2020-2025).
            Builds per-file commit order and per-commit file set.
            Full history needed because a bug-inducing commit may predate
            the 2020 analysis window.

    Pass 2: for each bug-fix commit, find the immediately preceding
            commit per file → designate as bug-inducing.

    Returns set of (file_path, commit_hash) pairs labelled defective.
    """
    print(f"  Pass 1: building full file commit history (entire git history)…")

    file_history = defaultdict(list)   # file_path → [hash, hash, ...]
    commit_meta  = {}                  # hash → commit message
    commit_files = defaultdict(set)    # hash → set of .py paths touched

    for commit in tqdm(
        Repository(git_repo).traverse_commits(),
        desc=f"  [{repo_name}] Pass 1",
        unit="commit",
    ):
        commit_meta[commit.hash] = commit.msg or ""
        for mf in commit.modified_files:
            path = mf.new_path or mf.old_path
            if not path or not path.endswith(".py"):
                continue
            file_history[path].append(commit.hash)
            commit_files[commit.hash].add(path)

    total_commits = len(commit_meta)
    print(f"  Tracked {len(file_history):,} Python files "
          f"across {total_commits:,} commits.")

    # ── Keyword classifier audit ───────────────────────────────────────────
    positive_pre  = sum(1 for m in commit_meta.values() if BUG_PATTERNS.search(m or ""))
    positive_post = sum(1 for m in commit_meta.values() if is_bug_fix(m))
    print(f"\n  Keyword classifier audit:")
    print(f"    Total commits              : {total_commits:,}")
    print(f"    Positive pre-exclusion     : {positive_pre:,}  ({positive_pre/total_commits:.2%})")
    print(f"    Bug-fix post-exclusion     : {positive_post:,}  ({positive_post/total_commits:.2%})")
    print(f"    Exclusion filter removed   : {positive_pre - positive_post:,} false positives")

    # ── Pass 2 ────────────────────────────────────────────────────────────
    print(f"\n  Pass 2: tracing bug-inducing commits (SZZ)…")

    bug_fix_count    = 0
    bug_inducing_set = set()

    for chash, msg in commit_meta.items():
        if not is_bug_fix(msg):
            continue
        bug_fix_count += 1
        for fpath in commit_files.get(chash, set()):
            history = file_history.get(fpath, [])
            try:
                idx = history.index(chash)
            except ValueError:
                continue
            if idx > 0:
                bug_inducing_set.add((fpath, history[idx - 1]))

    print(f"  Bug-fix commits found             : {bug_fix_count:,}")
    print(f"  Bug-inducing (file, hash) pairs   : {len(bug_inducing_set):,}")

    return bug_inducing_set


# ══════════════════════════════════════════════════════════════════════════════
#  APPLY LABELS + TEST-FILE-AWARE HEALTH REPORT
# ══════════════════════════════════════════════════════════════════════════════

def apply_labels(df: pd.DataFrame,
                 bug_inducing_set: set,
                 repo_name: str) -> pd.DataFrame:
    df = df.copy()

    # ── Core label ─────────────────────────────────────────────────────────
    df["is_defective"] = df.apply(
        lambda r: int((r["file_path"], r["commit_hash"]) in bug_inducing_set),
        axis=1,
    )

    # ── Test-file classification  ──────────────────────────────
    df["is_test_file"]        = df["file_path"].apply(is_test_file).astype(int)
    df["is_production_file"]  = (1 - df["is_test_file"])

    prod_df = df[df["is_test_file"] == 0]
    test_df = df[df["is_test_file"] == 1]

    # ── Overall rates ──────────────────────────────────────────────────────
    row_rate_all  = df["is_defective"].mean()
    row_rate_prod = prod_df["is_defective"].mean() if len(prod_df) else 0.0
    row_rate_test = test_df["is_defective"].mean() if len(test_df) else 0.0

    # File-level ever-defective rates
    def file_rate(frame):
        if len(frame) == 0:
            return 0.0, 0, 0
        grp = frame.groupby("file_path")["is_defective"].max()
        return grp.mean(), int(grp.sum()), len(grp)

    frate_all,  n_def_all,  n_files_all  = file_rate(df)
    frate_prod, n_def_prod, n_files_prod = file_rate(prod_df)
    frate_test, n_def_test, n_files_test = file_rate(test_df)

    print(f"\n  ── Label health report ─────────────────────────────────────────")
    print(f"  {'Metric':<40} {'ALL':>10} {'PROD':>10} {'TEST':>10}")
    print(f"  {'─'*72}")
    print(f"  {'Snapshot rows':<40} {len(df):>10,} {len(prod_df):>10,} {len(test_df):>10,}")
    print(f"  {'Unique files':<40} {n_files_all:>10,} {n_files_prod:>10,} {n_files_test:>10,}")
    print(f"  {'Row defect rate':<40} {row_rate_all:>10.2%} {row_rate_prod:>10.2%} {row_rate_test:>10.2%}")
    print(f"  {'File-level ever-defective rate':<40} {frate_all:>10.2%} {frate_prod:>10.2%} {frate_test:>10.2%}")
    print(f"  {'─'*72}")

    # ── Health gates — production files only  ──────────────────────
    print(f"\n  Health gates (production files only):")
    if n_files_prod == 0:
        print(f"  ⚠ No production files in this subset — check file paths.")
    elif frate_prod < 0.05:
        print(f"  ⚠ Prod file rate {frate_prod:.2%} < 5% — too few bug fixes detected.")
        print(f"    Consider relaxing BUG_PATTERNS slightly.")
    elif frate_prod > 0.35:
        print(f"  ⚠ Prod file rate {frate_prod:.2%} > 35% — still over-labelling.")
        print(f"    Inspect flagged commit messages with the audit above.")
    else:
        print(f"  ✓ Prod file-level rate {frate_prod:.2%} is healthy (target: 5–30%)")

    if row_rate_prod < 0.02:
        print(f"  ⚠ Prod row rate {row_rate_prod:.2%} < 2% — very sparse.")
        print(f"    Use class_weight='balanced' in all classifiers.")
    elif row_rate_prod > 0.40:
        print(f"  ⚠ Prod row rate {row_rate_prod:.2%} > 40% — still elevated.")
    else:
        print(f"  ✓ Prod row-level rate {row_rate_prod:.2%} is healthy (typical: 5–30%)")

    print(f"\n  Note: test-file rates are expected to be high (SZZ traces")
    print(f"  back from bug-fix commits to test file changes — this is a")
    print(f"  known SZZ limitation). Exclude test files in Stage 4 ML.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(cfg: dict):
    input_path  = Path(cfg["input"])
    output_path = Path(cfg["output"])
    git_repo    = cfg["git_repo"]
    repo_name   = cfg["repo_name"]

    # Derive CSV path from parquet path (FIX 1)
    output_csv  = output_path.with_suffix(".csv")

    print(f"\n{'='*70}")
    print(f"  SZZ LABELLER v3 : {repo_name}")
    print(f"  Input            : {input_path}")
    print(f"  Output parquet   : {output_path}")
    print(f"  Output CSV       : {output_csv}")
    print(f"  Keywords         : fix/fixes/fixed/fixing | bug | "
          f"regression | crash | defect")
    print(f"  Git repo         : {git_repo}")
    print(f"{'='*70}")

    # Clean previous outputs
    for p in (output_path, output_csv):
        if p.exists():
            print(f"  Exists — deleting: {p.name}")
            p.unlink()

    if not input_path.exists():
        print(f"  [ERROR] Input not found: {input_path}")
        print(f"  Run create_test_subset.py or the miner (Stage 1) first.")
        return

    df = pd.read_parquet(input_path)
    print(f"  Loaded: {len(df):,} rows | {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # SZZ labelling
    bug_inducing = run_szz(git_repo, repo_name)
    df = apply_labels(df, bug_inducing, repo_name)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\n  ── Output ──────────────────────────────────────────────────────")
    print(f"  Parquet : {output_path}  ({output_path.stat().st_size/1024:.1f} KB)")
    print(f"  CSV     : {output_csv}  ({output_csv.stat().st_size/1024:.1f} KB)")
    print(f"  Shape   : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns : {list(df.columns)}")


def main():
    for cfg in REPOS:
        run(cfg)


if __name__ == "__main__":
    main()