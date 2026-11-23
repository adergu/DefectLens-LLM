#!/usr/bin/env python3
"""
Manual Files Miner.
Static metrics only — no git history, no process metrics.
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import lizard
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

# CONFIG
MANUAL_FILES_DIR = Path("data/01_raw_data/manual_files")
OUT_DIR          = Path("data/02_snapshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_OUTPUT_PARQUET = OUT_DIR / "manual_files_metrics.parquet"
FINAL_OUTPUT_CSV     = OUT_DIR / "manual_files_metrics.csv"

SNAPSHOT_DATE = datetime.now().date()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("manual_miner")


def compute_metrics(source: str) -> dict:
    try:
        ana   = lizard.analyze_file.analyze_source_code("tmp.py", source)
        funcs = ana.function_list
        return {
            "loc":                  ana.nloc or 0,
            "cyclomatic_complexity": ana.CCN or 0,
            "num_functions":        len(funcs),
            "avg_cc_per_function":  round(ana.average_cyclomatic_complexity or 0.0, 2),
            "max_cc":               max((f.cyclomatic_complexity for f in funcs), default=0),
        }
    except:
        return {
            "loc": 0, "cyclomatic_complexity": 0,
            "num_functions": 0, "avg_cc_per_function": 0.0, "max_cc": 0,
        }


def main():
    for f in (FINAL_OUTPUT_PARQUET, FINAL_OUTPUT_CSV):
        if f.exists():
            log.info(f"Output already exists — deleting and re-mining: {f}")
            f.unlink()

    if not MANUAL_FILES_DIR.exists():
        log.error(f"Directory not found: {MANUAL_FILES_DIR}")
        return

    py_files = sorted(MANUAL_FILES_DIR.glob("*.py"))
    if not py_files:
        log.error("No .py files found in manual_files directory.")
        return

    log.info(f"MINING {len(py_files)} manual files — snapshot date: {SNAPSHOT_DATE}")

    rows = []

    for file_path in py_files:
        try:
            source = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log.warning(f"Could not read {file_path.name}: {e}")
            continue

        rows.append({
            "repository_name": "manual_files",
            "file_path":       f"manual_files/{file_path.name}",
            "commit_hash":     "manual_snapshot",
            "commit_date":     SNAPSHOT_DATE,
            "snapshot_date":   SNAPSHOT_DATE,
            **compute_metrics(source),
        })

    # SAVE
    df = pd.DataFrame(rows)
    df.to_parquet(FINAL_OUTPUT_PARQUET, compression="snappy", index=False)
    df.to_csv(FINAL_OUTPUT_CSV, index=False, encoding="utf-8")
    log.info(f"\nDONE")
    log.info(f"   Parquet : {FINAL_OUTPUT_PARQUET}")
    log.info(f"   CSV     : {FINAL_OUTPUT_CSV}")
    log.info(f"   Files   : {len(df)}")

    print("\n" + "=" * 60)
    print("SUMMARY OF GENERATED METRICS")
    print("=" * 60)
    print(df[["file_path", "loc", "cyclomatic_complexity", "num_functions", "max_cc"]].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()