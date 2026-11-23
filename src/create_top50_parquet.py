# CREATE SMALL TEST PARQUET (TOP 50 ROWS)
# For fast experimentation with LLM scoring

import pandas as pd
from pathlib import Path

# ============================ PATHS ============================

INPUT_PARQUET = Path("data/03_labeled/fastapi_2020_2025_FINAL_LABELED.parquet")
OUTPUT_PARQUET = Path("data/03_labeled/fastapi_2020_2025_FINAL_LABELED_TOP50.parquet")

TOP_N = 50

# ============================ MAIN ============================

def main():
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PARQUET}")

    print(f"Reading: {INPUT_PARQUET}")
    df = pd.read_parquet(INPUT_PARQUET)

    print(f"Original rows: {len(df):,}")

    df_small = df.head(TOP_N).copy()

    print(f"Saving top {TOP_N} rows to:")
    print(f"{OUTPUT_PARQUET}")

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_small.to_parquet(
        OUTPUT_PARQUET,
        index=False,
        compression="snappy"
    )

    print("✓ Done.")
    print(f"New file rows: {len(df_small):,}")

if __name__ == "__main__":
    main()
