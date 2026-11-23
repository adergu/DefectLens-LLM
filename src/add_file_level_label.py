#!/usr/bin/env python3
"""
ADD is_defective_file COLUMN TO ALL 4 PARQUET FILES
AMARE DERGU MENGSTIE | 2026

Reads each LLM-scored parquet, computes is_defective_file,
adds it as a permanent column, and overwrites the same file.

LOGIC
─────
  is_defective_file = 1  if:
    · the file was EVER labelled defective by SZZ  (n > 0)
    · less than 50% of its snapshots are defective  (frac < 0.50)
    · it has at least 3 snapshots                   (t >= 3)
  else 0

  The 50% threshold excludes chronically over-labelled files.
  The 3-snapshot minimum ensures enough temporal context.

VALIDATION PRINTED
──────────────────
  Per repo:
    · Row counts before / after
    · is_defective    : count, rate
    · is_defective_file: count, rate
    · Defect fraction distribution (histogram)
    · Cross-tabulation: SZZ label vs file-level label
    · Spot-check: 5 sample files showing both labels
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

FILES = [
    Path("data/04_llm_temp/django_2020_2025_FINAL_LABELED_12D_LLM_GROQ.parquet"),
    Path("data/04_llm_temp/fastapi_2020_2025_FINAL_LABELED_12D_LLM_GROQ.parquet"),
    Path("data/04_llm_temp/numpy_2020_2025_FINAL_LABELED_12D_LLM_GROQ.parquet"),
    Path("data/04_llm_temp/pandas_2020_2025_FINAL_LABELED_12D_LLM_GROQ.parquet"),
]

# Thresholds
FRAC_THRESHOLD   = 0.50   # max defect fraction to keep as defective
MIN_SNAPSHOTS    = 3      # minimum snapshots per file
TARGET           = "is_defective"
NEW_COL          = "is_defective_file"

W = 66

# ══════════════════════════════════════════════════════════════════════════════
#  COMPUTE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def compute_file_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds is_defective_file column.
    Safe to call multiple times — drops existing column first.
    """
    df = df.copy()

    # Drop if already exists (re-run safety)
    if NEW_COL in df.columns:
        df = df.drop(columns=[NEW_COL])

    # Per-file aggregation
    stats = (df.groupby("file_path")[TARGET]
               .agg(n="sum", t="count")
               .reset_index())
    stats["frac"] = stats["n"] / stats["t"]

    # Merge back
    df = df.merge(stats, on="file_path", how="left")

    # Apply label
    df[NEW_COL] = np.where(
        (df["n"] > 0) &
        (df["frac"] < FRAC_THRESHOLD) &
        (df["t"] >= MIN_SNAPSHOTS),
        1, 0
    ).astype(int)

    # Drop helper columns — keep parquet clean
    df = df.drop(columns=["n", "t", "frac"])

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

def spark(val, mx=1.0, w=20):
    filled = int(min(val / mx, 1.0) * w)
    return "█" * filled + "░" * (w - filled)

def print_validation(df: pd.DataFrame, repo: str):
    n_rows  = len(df)
    n_files = df["file_path"].nunique()

    # Row-level rates
    szz_n    = int(df[TARGET].sum())
    szz_rate = szz_n / n_rows

    fl_n     = int(df[NEW_COL].sum())
    fl_rate  = fl_n / n_rows

    # File-level rates
    file_szz_rate = df.groupby("file_path")[TARGET].max().mean()
    file_fl_rate  = df.groupby("file_path")[NEW_COL].max().mean()

    print(f"\n  ── {repo.upper()} ─────────────────────────────────────────────")
    print(f"  Rows          : {n_rows:,}")
    print(f"  Unique files  : {n_files:,}")
    print()
    print(f"  {'Metric':<35} {'Row rate':>9}  {'File rate':>10}")
    print(f"  {'─'*35}  {'─'*9}  {'─'*10}")
    print(f"  {'is_defective  (SZZ row-level)':<35} {szz_rate:>9.2%}  {file_szz_rate:>10.2%}")
    print(f"  {'is_defective_file (new column)':<35} {fl_rate:>9.2%}  {file_fl_rate:>10.2%}")

    # Defect fraction histogram (ever-defective files only)
    stats = df.groupby("file_path")[TARGET].agg(n="sum", t="count")
    stats["frac"] = stats["n"] / stats["t"]
    ever_def = stats[stats["n"] > 0]

    print(f"\n  Defect fraction distribution ({len(ever_def):,} ever-defective files):")
    bins   = [0, .10, .20, .30, .50, .70, 1.01]
    labels = ["0–10%","10–20%","20–30%","30–50%","50–70%","70–100%"]
    counts, _ = np.histogram(ever_def["frac"], bins=bins)
    mx = max(counts) or 1
    for lbl, cnt in zip(labels, counts):
        kept = "← kept" if lbl in ["0–10%","10–20%","20–30%","30–50%"] else "← excluded (frac≥50%)"
        print(f"    {lbl:<10}  {cnt:>4}  {spark(cnt, mx, 18)}  {kept}")

    # Cross-tabulation
    ct = pd.crosstab(df[TARGET], df[NEW_COL],
                     rownames=["is_defective"],
                     colnames=["is_defective_file"])
    print(f"\n  Cross-tabulation (rows = is_defective, cols = is_defective_file):")
    print(ct.to_string(index=True))
    print(f"\n  Interpretation:")
    print(f"  · SZZ=0, file=0 : clean files, all snapshots clean")
    print(f"  · SZZ=0, file=1 : clean snapshot of a defect-prone file  ← main training positive")
    print(f"  · SZZ=1, file=1 : the actual bug-inducing snapshot, also file-positive")
    print(f"  · SZZ=1, file=0 : bug snapshot but file excluded (frac≥50% or <3 snapshots)")

    # Spot-check: 5 files with is_defective_file=1 showing both labels
    print(f"\n  Spot-check — 5 sample defective files:")
    print(f"  {'file_path':<45}  {'SZZ(1s)':>7}  {'snaps':>6}  {'frac':>6}  {'file_lbl':>8}")
    print(f"  {'─'*45}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*8}")

    sample_files = (df[df[NEW_COL] == 1]["file_path"]
                    .drop_duplicates()
                    .sample(min(5, df[df[NEW_COL]==1]["file_path"].nunique()),
                            random_state=42))
    for fp in sample_files:
        sub  = df[df["file_path"] == fp]
        n    = int(sub[TARGET].sum())
        t    = len(sub)
        frac = n / t
        fl   = int(sub[NEW_COL].iloc[0])
        short = str(fp)[-45:] if len(str(fp)) > 45 else str(fp)
        print(f"  {short:<45}  {n:>7}  {t:>6}  {frac:>6.2%}  {fl:>8}")

    # Health verdict
    print()
    if 0.05 <= file_fl_rate <= 0.70:
        print(f"  ✔ File-level rate {file_fl_rate:.2%} — acceptable for prediction task")
    else:
        print(f"  ⚠ File-level rate {file_fl_rate:.2%} — check thresholds")

    if NEW_COL in df.columns:
        print(f"  ✔ {NEW_COL} column present and populated")
    else:
        print(f"  ✗ {NEW_COL} column MISSING — something went wrong")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*W}")
    print(f"  ADD is_defective_file — ALL 4 REPOS".center(W))
    print(f"  Threshold: frac < {FRAC_THRESHOLD:.0%}  |  min snapshots: {MIN_SNAPSHOTS}")
    print(f"{'='*W}")

    all_ok = True

    for path in FILES:
        repo = path.stem.split("_")[0]

        if not path.exists():
            print(f"\n  ⚠ FILE NOT FOUND — skipping: {path}")
            all_ok = False
            continue

        # Load
        df_orig = pd.read_parquet(path)
        cols_before = list(df_orig.columns)

        # Compute
        df_new = compute_file_label(df_orig)

        # Verify column added
        assert NEW_COL in df_new.columns, f"Column not added for {repo}"
        assert len(df_new) == len(df_orig), "Row count changed!"
        assert df_new.columns.duplicated().sum() == 0, "Duplicate columns!"

        # Save — overwrite same file, same path
        df_new.to_parquet(path, index=False, compression="snappy")

        # Print validation
        print_validation(df_new, repo)

        # Confirm new column is last
        print(f"\n  Columns before : {len(cols_before)}")
        print(f"  Columns after  : {len(df_new.columns)}")
        print(f"  New column     : '{NEW_COL}'  at position {list(df_new.columns).index(NEW_COL)}")
        print(f"  ✔ Saved → {path}")

    print(f"\n{'='*W}")
    if all_ok:
        print(f"  ✔ ALL DONE — is_defective_file added to all 4 files".center(W))
    else:
        print(f"  ⚠ SOME FILES SKIPPED — check warnings above".center(W))
    print(f"\n  Next: update ML scripts to use is_defective_file as target")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()