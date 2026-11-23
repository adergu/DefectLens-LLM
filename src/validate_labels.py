#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   LABEL VALIDATION — DO FEATURES DIFFER BETWEEN DEFECTIVE & CLEAN FILES?   ║
║   Author : Amare Dergu Mengstie  |  Academic Year 2025-2026                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE
───────
  Validates that is_defective_file is a meaningful label by checking
  whether files labelled 1 are statistically different from files
  labelled 0 in their static metrics, process metrics, and LLM scores.

  If defective files consistently show:
    · higher complexity, churn, revisions  → label is trustworthy
    · lower LLM quality scores             → LLM scores add value

  This script answers TWO questions simultaneously:
    1. Are the labels trustworthy?     (methodological validation)
    2. Which features discriminate?    (feature selection for RQ3 ML)

METHOD
──────
  · Mann-Whitney U test    — non-parametric, robust to skew/outliers
  · Rank-biserial r        — effect size (0=none, 1=perfect separation)
  · Direction              — is the feature higher in defective files?
  · Significance           — p < 0.05 after Bonferroni correction

  All tests use FILE-LEVEL aggregated values (mean per file across
  all its snapshots) to avoid pseudo-replication from repeated
  snapshots of the same file.

INPUT
─────
  data/04_llm_temp/<repo>_2020_2025_FINAL_LABELED_12D_LLM_GROQ.parquet
  (must have is_defective_file column — run add_file_level_label.py first)

OUTPUT
──────
  Console report per repo + cross-repo summary
  data/06_results/<repo>_label_validation.csv
  data/06_results/all_repos_label_validation.csv
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

try:
    from scipy import stats
except ImportError:
    print("\n  [ERROR] scipy not installed.  Run: pip install scipy\n")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

REPOS = [
    {
        "name": "django",
        "path": "data/04_llm_temp/django_2020_2025_FINAL_LABELED_12D_LLM_GROQ.parquet",
        "bp":   "llm_django_bp",
    },
    {
        "name": "fastapi",
        "path": "data/04_llm_temp/fastapi_2020_2025_FINAL_LABELED_12D_LLM_GROQ.parquet",
        "bp":   "llm_fastapi_bp",
    },
    {
        "name": "numpy",
        "path": "data/04_llm_temp/numpy_2020_2025_FINAL_LABELED_12D_LLM_GROQ.parquet",
        "bp":   "llm_numpy_bp",
    },
    {
        "name": "pandas",
        "path": "data/04_llm_temp/pandas_2020_2025_FINAL_LABELED_12D_LLM_GROQ.parquet",
        "bp":   "llm_pandas_bp",
    },
]

OUTPUT_DIR = Path("data/06_results")

STATIC_FEATURES = [
    "loc", "cyclomatic_complexity", "num_functions",
    "avg_cc_per_function", "max_cc",
]
PROCESS_FEATURES = [
    "num_revisions", "num_authors", "age_days", "churn",
    "code_ownership", "minor_contributors", "num_bug_fixes",
    "num_refactors", "recent_churn_90d", "recent_revisions_90d",
]
LLM_BASE = [
    "llm_readability", "llm_naming", "llm_structure", "llm_duplication",
    "llm_documentation", "llm_complexity", "llm_error_handling",
    "llm_testability", "llm_anti_patterns", "llm_magic_values",
    "llm_style", "llm_quality_score_12d",
]

TARGET     = "is_defective_file"
SIG_LEVEL  = 0.05
W          = 74

# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def banner():
    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + " LABEL VALIDATION — METRICS vs is_defective_file".center(W-2) + "║")
    print("║" + f" {datetime.now().strftime('%Y-%m-%d %H:%M')}".center(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")

def section(num, title):
    print()
    print(f"  ┌{'─'*(W-4)}┐")
    print(f"  │  {num}  {title:<{W-9}}│")
    print(f"  └{'─'*(W-4)}┘")

def rule():  print(f"  {'─'*(W-4)}")
def ok(m):   print(f"  ✔  {m}")
def warn(m): print(f"  ⚠  {m}")
def info(m): print(f"  ·  {m}")

def mini_bar(val, max_val=1.0, width=14):
    filled = int(min(abs(val) / max_val, 1.0) * width)
    return "█" * filled + "░" * (width - filled)

def effect_label(r):
    a = abs(r)
    if a >= 0.50: return "LARGE  ●●●"
    if a >= 0.30: return "MEDIUM ●●○"
    if a >= 0.10: return "SMALL  ●○○"
    return              "TINY   ○○○"

# ══════════════════════════════════════════════════════════════════════════════
#  CORE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def rank_biserial_r(u_stat, n1, n2):
    """Effect size for Mann-Whitney U: rank-biserial correlation."""
    return 1 - (2 * u_stat) / (n1 * n2)

def analyse_features(file_df: pd.DataFrame,
                     features: list,
                     n_tests: int) -> pd.DataFrame:
    """
    For each feature, compare defective vs clean files using
    Mann-Whitney U test and compute rank-biserial effect size.

    Uses FILE-LEVEL mean values (one row per file) to avoid
    pseudo-replication from repeated snapshots.
    """
    defective = file_df[file_df[TARGET] == 1]
    clean     = file_df[file_df[TARGET] == 0]

    bonferroni = SIG_LEVEL / n_tests   # corrected significance threshold

    rows = []
    for feat in features:
        if feat not in file_df.columns:
            continue

        x_def   = defective[feat].dropna().values
        x_clean = clean[feat].dropna().values

        if len(x_def) < 5 or len(x_clean) < 5:
            continue

        try:
            u_stat, p_val = stats.mannwhitneyu(
                x_def, x_clean, alternative="two-sided"
            )
            effect_r = rank_biserial_r(u_stat, len(x_def), len(x_clean))
        except Exception:
            continue

        mean_def   = float(np.mean(x_def))
        mean_clean = float(np.mean(x_clean))
        direction  = "↑ higher in defective" if mean_def > mean_clean \
                     else "↓ lower in defective"

        rows.append({
            "feature":      feat,
            "mean_defective": round(mean_def, 3),
            "mean_clean":     round(mean_clean, 3),
            "mean_diff":      round(mean_def - mean_clean, 3),
            "effect_r":       round(float(effect_r), 4),
            "abs_effect":     round(abs(float(effect_r)), 4),
            "p_value":        round(float(p_val), 6),
            "significant":    bool(p_val < SIG_LEVEL),
            "sig_bonferroni": bool(p_val < bonferroni),
            "direction":      direction,
        })

    return (pd.DataFrame(rows)
              .sort_values("abs_effect", ascending=False)
              .reset_index(drop=True))

# ══════════════════════════════════════════════════════════════════════════════
#  PRINT FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_table(result_df: pd.DataFrame,
                        group_name: str,
                        n_bonferroni: int):
    print()
    info(f"Group: {group_name}  |  Bonferroni threshold: p < {SIG_LEVEL/n_bonferroni:.4f}")
    rule()

    cw = [32, 9, 9, 10, 12, 10, 26]
    hdr = ["Feature", "Def mean", "Cln mean", "Effect r", "Strength",
           "p-value", "Direction"]
    print("  " + "  ".join(f"{h:<{w}}" for h, w in zip(hdr, cw)))
    print("  " + "  ".join("─"*w for w in cw))

    for _, r in result_df.iterrows():
        sig  = "**" if r["sig_bonferroni"] else \
               "*"  if r["significant"]    else "  "
        vals = [
            r["feature"],
            f"{r['mean_defective']:.3f}",
            f"{r['mean_clean']:.3f}",
            f"{r['effect_r']:+.4f}{sig}",
            effect_label(r["effect_r"]),
            f"{r['p_value']:.4f}",
            r["direction"],
        ]
        print("  " + "  ".join(f"{v:<{w}}" for v, w in zip(vals, cw)))

    print()
    info("** Bonferroni-corrected significant  * p<0.05  "
         "Effect: |r|≥0.50=LARGE, ≥0.30=MEDIUM, ≥0.10=SMALL")


def print_ranked_bar(result_df: pd.DataFrame, title: str, top_n: int = 15):
    print()
    info(f"Top {top_n} features by effect size  —  {title}")
    rule()

    max_r = result_df["abs_effect"].max() or 0.01
    for _, r in result_df.head(top_n).iterrows():
        b    = mini_bar(r["abs_effect"], max_val=max_r)
        sign = "+" if r["effect_r"] >= 0 else "−"
        sig  = "**" if r["sig_bonferroni"] else \
               "* " if r["significant"]    else "  "

        feat = r["feature"]
        if feat in STATIC_FEATURES:
            tag = "[S]"
        elif feat in PROCESS_FEATURES:
            tag = "[P]"
        else:
            tag = "[L]"

        print(f"  {tag} {feat:<33}  {sign}{r['abs_effect']:.4f}{sig}  {b}")

    print()
    info("[S]=Static  [P]=Process  [L]=LLM    ** Bonferroni  * p<0.05")


def print_group_summary(static_res: pd.DataFrame,
                        process_res: pd.DataFrame,
                        llm_res: pd.DataFrame):
    """Compare average effect size across feature groups."""
    print()
    info("Feature group comparison (average |effect r|):")
    rule()

    groups = {
        "Static  (Lizard)  ": static_res,
        "Process (git)     ": process_res,
        "LLM scores        ": llm_res,
    }

    group_stats = []
    for gname, df in groups.items():
        if df.empty:
            continue
        mean_eff  = df["abs_effect"].mean()
        max_eff   = df["abs_effect"].max()
        n_sig     = int(df["significant"].sum())
        n_sig_bon = int(df["sig_bonferroni"].sum())
        n_total   = len(df)
        group_stats.append((gname, mean_eff, max_eff, n_sig, n_sig_bon, n_total))

    group_stats.sort(key=lambda x: -x[1])

    print(f"  {'Group':<22}  {'Mean|r|':>8}  {'Max|r|':>7}  "
          f"{'p<0.05':>7}  {'Bonf':>5}  {'Bar'}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*5}  {'─'*14}")

    medals = ["① STRONGEST", "② 2nd", "③ 3rd"]
    for i, (gname, mean_r, max_r, n_sig, n_bon, n_tot) in enumerate(group_stats):
        b     = mini_bar(mean_r, max_val=group_stats[0][1])
        medal = medals[i] if i < 3 else ""
        print(f"  {gname:<22}  {mean_r:>8.4f}  {max_r:>7.4f}  "
              f"{n_sig:>4}/{n_tot:<2}  {n_bon:>4}/{n_tot:<2}  {b}  {medal}")

    return group_stats


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE REPO RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_repo(cfg: dict, all_rows: list) -> pd.DataFrame:
    repo_name = cfg["name"]
    path      = Path(cfg["path"])
    bp_col    = cfg["bp"]

    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + f"  {repo_name.upper()}".ljust(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")

    # ── Load ──────────────────────────────────────────────────────────────────
    if not path.exists():
        warn(f"File not found: {path}")
        warn(f"Run add_file_level_label.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(path)
    if "repository_name" in df.columns:
        df = df[df["repository_name"] == repo_name].copy()

    if TARGET not in df.columns:
        warn(f"Column '{TARGET}' not found.")
        warn(f"Run add_file_level_label.py first.")
        return pd.DataFrame()

    n_rows  = len(df)
    n_files = df["file_path"].nunique()
    n_def   = int(df.groupby("file_path")[TARGET].max().sum())
    n_clean = n_files - n_def

    info(f"Rows: {n_rows:,}  |  Files: {n_files:,}  |  "
         f"Defective: {n_def:,}  ({n_def/n_files:.1%})  |  "
         f"Clean: {n_clean:,}  ({n_clean/n_files:.1%})")

    # ── Aggregate to file level ────────────────────────────────────────────
    # Use MEAN per file across all snapshots — avoids pseudo-replication
    section("01", "AGGREGATING TO FILE LEVEL  (mean per file across snapshots)")

    all_features = (STATIC_FEATURES + PROCESS_FEATURES +
                    LLM_BASE + [bp_col])
    avail_feats  = [f for f in all_features if f in df.columns]

    file_df = (df.groupby("file_path")[avail_feats + [TARGET]]
                 .agg({**{f: "mean" for f in avail_feats}, TARGET: "max"})
                 .reset_index())

    info(f"File-level dataframe: {len(file_df):,} files  ×  "
         f"{len(avail_feats)} features")
    info(f"Defective files : {int(file_df[TARGET].sum()):,}  "
         f"({file_df[TARGET].mean():.1%})")
    info(f"Clean files     : {int((file_df[TARGET]==0).sum()):,}")

    # ── Feature analysis ──────────────────────────────────────────────────
    avail_static  = [f for f in STATIC_FEATURES  if f in file_df.columns]
    avail_process = [f for f in PROCESS_FEATURES if f in file_df.columns]
    avail_llm     = [f for f in LLM_BASE + [bp_col] if f in file_df.columns]
    n_total_tests = len(avail_static) + len(avail_process) + len(avail_llm)

    section("02", "STATIC METRICS  —  Mann-Whitney U + effect size")
    static_res = analyse_features(file_df, avail_static, n_total_tests)
    print_feature_table(static_res, "Static metrics (Lizard)", n_total_tests)

    section("03", "PROCESS METRICS  —  Mann-Whitney U + effect size")
    process_res = analyse_features(file_df, avail_process, n_total_tests)
    print_feature_table(process_res, "Process metrics (git)", n_total_tests)

    section("04", "LLM QUALITY SCORES  —  Mann-Whitney U + effect size")
    llm_res = analyse_features(file_df, avail_llm, n_total_tests)
    print_feature_table(llm_res, "LLM quality scores", n_total_tests)

    # ── Combined ranked bar ────────────────────────────────────────────────
    section("05", "ALL FEATURES RANKED BY EFFECT SIZE")
    all_res = pd.concat([static_res, process_res, llm_res], ignore_index=True)
    all_res = all_res.sort_values("abs_effect", ascending=False).reset_index(drop=True)
    print_ranked_bar(all_res, repo_name, top_n=15)

    # ── Group comparison ───────────────────────────────────────────────────
    section("06", "FEATURE GROUP COMPARISON")
    group_stats = print_group_summary(static_res, process_res, llm_res)

    # ── Label verdict ──────────────────────────────────────────────────────
    section("07", "LABEL VALIDITY VERDICT")

    n_sig_any  = int(all_res["significant"].sum())
    n_sig_bon  = int(all_res["sig_bonferroni"].sum())
    max_effect = float(all_res["abs_effect"].max())
    llm_sig    = int(llm_res["significant"].sum()) if not llm_res.empty else 0
    llm_max    = float(llm_res["abs_effect"].max()) if not llm_res.empty else 0

    print()
    info(f"Significant features (p<0.05)            : {n_sig_any} / {n_total_tests}")
    info(f"Significant after Bonferroni correction  : {n_sig_bon} / {n_total_tests}")
    info(f"Maximum effect size                      : {max_effect:.4f}")
    info(f"LLM features significant (p<0.05)        : {llm_sig} / {len(avail_llm)}")
    info(f"LLM maximum effect size                  : {llm_max:.4f}")
    print()
    rule()

    if n_sig_bon >= 3 and max_effect >= 0.30:
        ok(f"LABELS ARE VALID — {n_sig_bon} features significant after Bonferroni, "
           f"max effect={max_effect:.3f}")
        ok("Defective files are statistically distinguishable from clean files.")
        ok("is_defective_file is a trustworthy target for ML.")
    elif n_sig_any >= 5 and max_effect >= 0.15:
        ok(f"LABELS ARE ACCEPTABLE — {n_sig_any} features significant (p<0.05), "
           f"max effect={max_effect:.3f}")
        info("Moderate discrimination. Labels are usable for ML.")
    else:
        warn(f"LABELS ARE WEAK — only {n_sig_any} significant features, "
             f"max effect={max_effect:.3f}")
        warn("Consider re-running labeller with stricter SZZ patterns.")

    print()
    if llm_sig >= 3 and llm_max >= 0.10:
        ok(f"LLM SCORES ADD VALUE — {llm_sig} LLM dimensions discriminate "
           f"(max |r|={llm_max:.3f})")
        ok("Adding LLM features to ML models is justified.")
    elif llm_sig >= 1:
        info(f"LLM scores show partial discrimination ({llm_sig} significant, "
             f"max |r|={llm_max:.3f})")
        info("LLM features may add marginal value in ML.")
    else:
        warn(f"LLM scores show no discrimination for {repo_name}.")
        info("LLM features unlikely to improve ML for this repo.")

    # ── Feature selection recommendation ──────────────────────────────────
    section("08", "FEATURE SELECTION RECOMMENDATION FOR RQ3 ML")

    strong = all_res[all_res["abs_effect"] >= 0.10]["feature"].tolist()
    weak   = all_res[all_res["abs_effect"] <  0.05]["feature"].tolist()

    llm_strong = [f for f in strong if f in avail_llm]
    llm_weak   = [f for f in weak   if f in avail_llm]

    print()
    info(f"Features with |r| ≥ 0.10  (KEEP for ML)  : {len(strong)}")
    for f in strong[:10]:
        tag = "[S]" if f in STATIC_FEATURES else \
              "[P]" if f in PROCESS_FEATURES else "[L]"
        r   = all_res.loc[all_res["feature"]==f, "abs_effect"].values[0]
        print(f"    {tag} {f:<35}  |r|={r:.4f}")
    if len(strong) > 10:
        info(f"    ... and {len(strong)-10} more")

    print()
    info(f"LLM features recommended for ML  : {llm_strong}")
    if llm_weak:
        info(f"LLM features to consider dropping : {llm_weak}")

    # ── Save ──────────────────────────────────────────────────────────────
    all_res["repo"] = repo_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{repo_name}_label_validation.csv"
    all_res.to_csv(out, index=False)
    ok(f"Saved → {out}")

    all_rows.extend(all_res.to_dict("records"))
    return all_res


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    banner()

    all_rows     = []
    repo_results = {}

    for cfg in REPOS:
        result = run_repo(cfg, all_rows)
        if not result.empty:
            repo_results[cfg["name"]] = result

    if not all_rows:
        warn("No results — check that parquet files exist and have is_defective_file.")
        return

    # ── Cross-repo summary ────────────────────────────────────────────────
    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + " CROSS-REPO SUMMARY".center(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")

    print()
    info("Top 5 most discriminative features per repo (by |effect r|):")
    rule()
    print(f"  {'Repo':<10}  {'Feature':<33}  {'|r|':>6}  {'Sig':>5}  Group")
    print(f"  {'─'*10}  {'─'*33}  {'─'*6}  {'─'*5}  {'─'*10}")

    for repo, res in repo_results.items():
        for _, r in res.head(5).iterrows():
            sig  = "**" if r["sig_bonferroni"] else "* " if r["significant"] else "  "
            feat = r["feature"]
            tag  = "[S]" if feat in STATIC_FEATURES else \
                   "[P]" if feat in PROCESS_FEATURES else "[L]"
            print(f"  {repo:<10}  {feat:<33}  {r['abs_effect']:>6.4f}  {sig:>5}  {tag}")
        print()

    # LLM cross-repo summary
    print()
    info("LLM score discrimination across repos:")
    rule()
    print(f"  {'Repo':<10}  {'LLM sig (p<0.05)':>17}  {'LLM max |r|':>12}  "
          f"{'LLM mean |r|':>13}  Verdict")
    print(f"  {'─'*10}  {'─'*17}  {'─'*12}  {'─'*13}  {'─'*15}")

    for repo, res in repo_results.items():
        llm_only = res[res["feature"].isin(LLM_BASE + [cfg["bp"]
                   for cfg in REPOS if cfg["name"] == repo])]
        if llm_only.empty:
            continue
        n_sig  = int(llm_only["significant"].sum())
        n_tot  = len(llm_only)
        max_r  = float(llm_only["abs_effect"].max())
        mean_r = float(llm_only["abs_effect"].mean())
        v = "✔ adds value" if n_sig >= 3 and max_r >= 0.10 else \
            "~ marginal"   if n_sig >= 1 else "✗ no signal"
        print(f"  {repo:<10}  {n_sig:>8}/{n_tot:<8}  {max_r:>12.4f}  "
              f"{mean_r:>13.4f}  {v}")

    # Save combined
    df_all = pd.DataFrame(all_rows)
    out_all = OUTPUT_DIR / "all_repos_label_validation.csv"
    df_all.to_csv(out_all, index=False)

    elapsed = time.time() - t0
    print()
    print("╔" + "═"*(W-2) + "╗")
    print("║" + f"  ✔  VALIDATION COMPLETE  ·  {elapsed:.1f}s".center(W-2) + "║")
    print("║" + f"  Saved → {out_all}".center(W-2) + "║")
    print("║" + "  Next: 7_ml_rq3_classification.py".center(W-2) + "║")
    print("╚" + "═"*(W-2) + "╝")
    print()


if __name__ == "__main__":
    main()