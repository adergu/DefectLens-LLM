#!/usr/bin/env python3
"""
Bimonthly Repository Miner  v2.0
─────────────────────────────────────────────────────────────────────────────
WHAT CHANGED FROM v1:
  - Full git history traversal from the very beginning (no START_DATE skip)
    so that files modified before 2020 are correctly present in living_files
    at the first snapshot date (Jan 2020).
  - Snapshot rows emitted ONLY when a file's commit_hash has changed since
    the last time that file was emitted. No redundant duplicate rows.
    First appearance of a file always emits a row.
  - Snapshot window: 2020-01-31 to 2025-11-30 (36 dates, 5 years).
  - Process metrics (num_revisions, churn, etc.) accumulated from FULL
    git history so age_days and revision counts reflect true file lifetime.
  - Output: one row per (file_path, commit_hash) change event captured at
    the nearest bimonthly snapshot boundary.

OUTPUT SCHEMA (20 columns — identical to v1):
  repository_name, file_path, commit_hash, commit_date, snapshot_date,
  loc, cyclomatic_complexity, num_functions, avg_cc_per_function, max_cc,
  num_revisions, num_authors, age_days, churn, code_ownership,
  minor_contributors, num_bug_fixes, num_refactors,
  recent_churn_90d, recent_revisions_90d

DOWNSTREAM: labeler v3 and LLM scorer v4 need ZERO changes.
"""

import re
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Tuple, Optional
from collections import defaultdict

import pandas as pd
from pydriller import Repository
from tqdm import tqdm
import lizard
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

# ── Configuration ─────────────────────────────────────────────────────────────

# ── 7 repositories — pandas replaced by scikit-learn, flask, matplotlib, sqlalchemy
# Already mined (skip if output exists): django, fastapi, numpy
# New repos to mine: scikit-learn, flask, matplotlib, sqlalchemy
REPOS = [
    # {
    #     "repo_path": Path("data/01_raw_data/django"),
    #     "repo_name": "django",
    #     "output":    Path("data/02_snapshots/django_2020_2025_bimonthly_CLEAN.parquet"),
    # },
    # {
    #     "repo_path": Path("data/01_raw_data/fastapi"),
    #     "repo_name": "fastapi",
    #     "output":    Path("data/02_snapshots/fastapi_2020_2025_bimonthly_CLEAN.parquet"),
    # },
    # {
    #     "repo_path": Path("data/01_raw_data/numpy"),
    #     "repo_name": "numpy",
    #     "output":    Path("data/02_snapshots/numpy_2020_2025_bimonthly_CLEAN.parquet"),
    # },
    {
        "repo_path": Path("data/01_raw_data/scikit-learn"),
        "repo_name": "scikit-learn",
        "output":    Path("data/02_snapshots/scikit_learn_2020_2025_bimonthly_CLEAN.parquet"),
    },
    {
        "repo_path": Path("data/01_raw_data/flask"),
        "repo_name": "flask",
        "output":    Path("data/02_snapshots/flask_2020_2025_bimonthly_CLEAN.parquet"),
    },
    {
        "repo_path": Path("data/01_raw_data/matplotlib"),
        "repo_name": "matplotlib",
        "output":    Path("data/02_snapshots/matplotlib_2020_2025_bimonthly_CLEAN.parquet"),
    },
    # {
    #     "repo_path": Path("data/01_raw_data/sqlalchemy"),
    #     "repo_name": "sqlalchemy",
    #     "output":    Path("data/02_snapshots/sqlalchemy_2020_2025_bimonthly_CLEAN.parquet"),
    # },
]

OUT_DIR = Path("data/02_snapshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Snapshot grid: last day of every odd month, 2020–2025
SNAPSHOT_DATES = [
    datetime(y, m, d, tzinfo=timezone.utc)
    for y in range(2020, 2026)
    for m, d in [(1,31),(3,31),(5,31),(7,31),(9,30),(11,30)]
]

# Study window start — only emit rows from this date onward
STUDY_START = datetime(2020, 1, 1, tzinfo=timezone.utc)

# Process metrics config
RECENCY_DAYS     = 90    # window for recent_churn_90d / recent_revisions_90d
MINOR_THRESHOLD  = 0.05  # authors below this fraction = minor contributor
MAX_FILE_CHARS   = 200_000

_BUG_RE = re.compile(
    r"\b(fix|bug|error|patch|issue|defect|fault|crash|fail|broken|workaround|hotfix)\b",
    re.IGNORECASE,
)
_REF_RE = re.compile(
    r"\b(refactor|cleanup|clean.?up|restructure|rewrite|reorgani[sz]e|simplif)\b",
    re.IGNORECASE,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("miner_v2")


# ── Static metrics ─────────────────────────────────────────────────────────────

def compute_metrics(source: str) -> dict:
    try:
        ana   = lizard.analyze_file.analyze_source_code("tmp.py", source)
        funcs = ana.function_list
        return {
            "loc":                   ana.nloc or 0,
            "cyclomatic_complexity": ana.CCN  or 0,
            "num_functions":         len(funcs),
            "avg_cc_per_function":   ana.average_cyclomatic_complexity or 0.0,
            "max_cc":                max((f.cyclomatic_complexity for f in funcs), default=0),
        }
    except Exception:
        return {
            "loc": 0, "cyclomatic_complexity": 0, "num_functions": 0,
            "avg_cc_per_function": 0.0, "max_cc": 0,
        }


# ── Snapshot emitter ───────────────────────────────────────────────────────────

def emit_snapshot(
    repo_name:     str,
    snap_date:     datetime,
    living_files:  Dict[str, Tuple[str, str, datetime]],
    last_emitted:  Dict[str, str],          # file_path → last emitted commit_hash
    file_first_seen:     dict,
    file_authors:        dict,
    file_revisions:      dict,
    file_churn:          dict,
    file_author_commits: dict,
    file_bug_fixes:      dict,
    file_refactors:      dict,
    file_recent_events:  dict,
    rows:                list,
):
    """
    For each file in living_files, emit a row only if:
      1. The file has never been emitted before (first appearance), OR
      2. The file's commit_hash has changed since the last emission.
    """
    recency_cutoff = snap_date - timedelta(days=RECENCY_DAYS)

    for path, (src, commit_hash, commit_date) in living_files.items():

        # ── Emit only if changed since last snapshot ──────────────────────────
        if last_emitted.get(path) == commit_hash:
            continue   # same version as last snapshot — skip
        # ─────────────────────────────────────────────────────────────────────

        n_rev = file_revisions[path]

        # Ownership
        author_commits = file_author_commits[path]
        top_commits    = max(author_commits.values(), default=0)
        ownership      = round(top_commits / n_rev, 4) if n_rev else 0.0
        minor          = sum(
            1 for cnt in author_commits.values()
            if n_rev and cnt / n_rev < MINOR_THRESHOLD
        )

        # Recency window
        recent               = [(rd, rc) for rd, rc in file_recent_events[path]
                                 if rd >= recency_cutoff]
        recent_churn_90d     = sum(rc for _, rc in recent)
        recent_revisions_90d = len(recent)

        # Age
        first_seen = file_first_seen.get(path)
        age_days   = max(0, (snap_date - first_seen).days) if first_seen else 0

        rows.append({
            "repository_name":    repo_name,
            "file_path":          path,
            "commit_hash":        commit_hash,
            "commit_date":        commit_date.date(),
            "snapshot_date":      snap_date.date(),
            **compute_metrics(src),
            "num_revisions":      n_rev,
            "num_authors":        len(file_authors[path]),
            "age_days":           age_days,
            "churn":              file_churn[path],
            "code_ownership":     ownership,
            "minor_contributors": minor,
            "num_bug_fixes":      file_bug_fixes[path],
            "num_refactors":      file_refactors[path],
            "recent_churn_90d":   recent_churn_90d,
            "recent_revisions_90d": recent_revisions_90d,
        })

        # Mark this commit as emitted for this file
        last_emitted[path] = commit_hash


# ── Main mining loop ───────────────────────────────────────────────────────────

def mine_repo(repo_path: Path, repo_name: str, output: Path):

    if output.exists():
        log.info(f"  Output exists — deleting and re-mining: {output}")
        output.unlink()

    log.info(f"\n{'='*70}")
    log.info(f"  MINING  {repo_name.upper()}")
    log.info(f"  Repo    : {repo_path}")
    log.info(f"  Output  : {output}")
    log.info(f"  Traversal: FULL git history (living_files populated from start)")
    log.info(f"  Snapshots: {SNAPSHOT_DATES[0].date()} → {SNAPSHOT_DATES[-1].date()} "
             f"({len(SNAPSHOT_DATES)} dates)")
    log.info(f"  Emit rule: only when commit_hash changes since last snapshot")
    log.info(f"{'='*70}")

    # State
    living_files: Dict[str, Tuple[str, str, datetime]] = {}
    last_emitted: Dict[str, str] = {}          # file_path → last emitted commit_hash
    rows = []

    snapshot_iter = iter(SNAPSHOT_DATES)
    next_snapshot = next(snapshot_iter, None)

    # Process metric accumulators — full history
    file_first_seen     = {}
    file_authors        = defaultdict(set)
    file_revisions      = defaultdict(int)
    file_churn          = defaultdict(int)
    file_author_commits = defaultdict(lambda: defaultdict(int))
    file_bug_fixes      = defaultdict(int)
    file_refactors      = defaultdict(int)
    file_recent_events  = defaultdict(list)    # [(commit_date, churn), ...]

    # ── Traverse FULL git history ─────────────────────────────────────────────
    for commit in tqdm(
        Repository(str(repo_path)).traverse_commits(),
        desc=f"  [{repo_name}] commits", unit="commit"
    ):
        cdate  = commit.committer_date
        author = commit.author.email or commit.author.name or "unknown"
        msg    = commit.msg or ""
        is_fix = bool(_BUG_RE.search(msg))
        is_ref = bool(_REF_RE.search(msg))

        # ── Emit snapshots that fall before this commit ───────────────────────
        # Only emit rows for snapshot dates within the study window
        while next_snapshot and cdate >= next_snapshot:
            if next_snapshot >= STUDY_START:
                emit_snapshot(
                    repo_name, next_snapshot, living_files, last_emitted,
                    file_first_seen, file_authors, file_revisions,
                    file_churn, file_author_commits, file_bug_fixes,
                    file_refactors, file_recent_events, rows,
                )
                log.info(
                    f"  Snapshot {next_snapshot.date()} → "
                    f"{len(living_files):,} living files | "
                    f"{len(rows):,} rows emitted so far"
                )
            try:
                next_snapshot = next(snapshot_iter)
            except StopIteration:
                next_snapshot = None
        # ─────────────────────────────────────────────────────────────────────

        # ── Update living_files and process accumulators ──────────────────────
        for m in commit.modified_files:
            path = m.new_path or m.old_path
            if not path or not path.endswith(".py"):
                continue

            # Deletion or rename — remove from living files
            if m.change_type.name in ("DELETE", "RENAME"):
                living_files.pop(path, None)
                if m.old_path:
                    living_files.pop(m.old_path, None)
                continue

            # Process metric accumulators (full history, all commits)
            if path not in file_first_seen:
                file_first_seen[path] = cdate
            file_authors[path].add(author)
            file_revisions[path]              += 1
            file_author_commits[path][author] += 1
            churn = (m.added_lines or 0) + (m.deleted_lines or 0)
            file_churn[path]                  += churn
            file_recent_events[path].append((cdate, churn))
            if is_fix:
                file_bug_fixes[path]  += 1
            if is_ref:
                file_refactors[path]  += 1

            # Update living_files only if source code is available and not too large
            if m.source_code and len(m.source_code) <= MAX_FILE_CHARS:
                living_files[path] = (m.source_code, commit.hash, cdate)
        # ─────────────────────────────────────────────────────────────────────

    # ── Emit any remaining snapshot dates after last commit ───────────────────
    while next_snapshot:
        if next_snapshot >= STUDY_START:
            emit_snapshot(
                repo_name, next_snapshot, living_files, last_emitted,
                file_first_seen, file_authors, file_revisions,
                file_churn, file_author_commits, file_bug_fixes,
                file_refactors, file_recent_events, rows,
            )
            log.info(
                f"  Snapshot {next_snapshot.date()} → "
                f"{len(living_files):,} living files | "
                f"{len(rows):,} rows emitted so far"
            )
        try:
            next_snapshot = next(snapshot_iter)
        except StopIteration:
            break

    # ── Save ──────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)

    # Validation
    assert df.isnull().sum().sum() == 0, "NULL values found — check miner logic"
    assert len(df) == df[['file_path','commit_hash','snapshot_date']].drop_duplicates().__len__(), \
        "Duplicate (file, commit, snapshot) rows found"

    output_csv = output.with_suffix(".csv")
    df.to_parquet(output,     compression="snappy", index=False)
    df.to_csv(output_csv,     index=False)

    parquet_kb = output.stat().st_size     / 1024
    csv_kb     = output_csv.stat().st_size / 1024

    log.info(f"\n  {'─'*60}")
    log.info(f"  DONE: {repo_name.upper()}")
    log.info(f"  Parquet : {output}  ({parquet_kb:,.1f} KB)")
    log.info(f"  CSV     : {output_csv}  ({csv_kb:,.1f} KB)")
    log.info(f"  Rows    : {len(df):,}")
    log.info(f"  Unique files  : {df['file_path'].nunique():,}")
    log.info(f"  Snapshot dates: {df['snapshot_date'].nunique()}")
    log.info(f"  Columns       : {list(df.columns)}")
    log.info(f"  Defect rate   : N/A (added by labeler v3)")
    log.info(f"  {'─'*60}")

    return df


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    log.info("BIMONTHLY MINER v2.0")
    log.info(f"Repos to mine : {len(REPOS)}")
    log.info(f"Snapshot dates: {len(SNAPSHOT_DATES)} "
             f"({SNAPSHOT_DATES[0].date()} → {SNAPSHOT_DATES[-1].date()})")
    log.info(f"Emit rule     : only when commit_hash changes (no redundant rows)")
    log.info(f"Process metrics: accumulated from FULL git history")

    for cfg in REPOS:
        mine_repo(cfg["repo_path"], cfg["repo_name"], cfg["output"])

    log.info("\nALL REPOS MINED SUCCESSFULLY")
    log.info("Next step: python src/2_real_bug_labeler_v3.py")


if __name__ == "__main__":
    main()