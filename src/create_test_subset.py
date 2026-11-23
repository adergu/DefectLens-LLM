#!/usr/bin/env python3
"""
CREATE SMALL TEST SUBSET
Slices the top N rows from a miner parquet for fast experimentation.
Writes both .parquet and .csv with the same stem name.

AMARE DERGU MENGSTIE | 2026
"""

import pandas as pd
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG — edit these to change repo or subset size
# ══════════════════════════════════════════════════════════════════════════════

INPUT_PARQUET  = Path("data/02_snapshots/django_2020_2025_bimonthly_CLEAN.parquet")
OUTPUT_DIR     = Path("data/03_labeled")
TOP_N          = 50

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PARQUET}")

    # Derive output stem: e.g. django_2020_2025_bimonthly_CLEAN → ...TOP50
    stem           = INPUT_PARQUET.stem                           # django_2020_2025_bimonthly_CLEAN
    repo_name      = stem.split("_")[0]                          # django
    out_stem       = f"{repo_name}_2020_2025_FINAL_LABELED_TOP{TOP_N}"
    output_parquet = OUTPUT_DIR / f"{out_stem}.parquet"
    output_csv     = OUTPUT_DIR / f"{out_stem}.csv"

    print(f"  Input    : {INPUT_PARQUET}  ({INPUT_PARQUET.stat().st_size/1024/1024:.1f} MB)")

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  Full rows: {len(df):,}  |  columns: {len(df.columns)}")

    df_small = df.head(TOP_N).copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df_small.to_parquet(output_parquet, index=False, compression="snappy")
    df_small.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"\n  Saved parquet : {output_parquet}  ({output_parquet.stat().st_size/1024:.1f} KB)")
    print(f"  Saved CSV     : {output_csv}  ({output_csv.stat().st_size/1024:.1f} KB)")
    print(f"  Rows          : {len(df_small):,}")
    print(f"  Columns       : {len(df_small.columns)}  {list(df_small.columns)}")
    print(f"\n  Snapshot dates in subset:")
    for d, cnt in df_small['snapshot_date'].value_counts().sort_index().items():
        print(f"    {d}  →  {cnt} files")


if __name__ == "__main__":
    main()
