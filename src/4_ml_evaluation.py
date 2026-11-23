#!/usr/bin/env python3
"""
Stage 4 — Machine Learning Evaluation
══════════════════════════════════════════════════════════════════════
Thesis: Predicting Software Defects from Code Structure and Quality
        Metrics Enhanced with LLM-based Quality Assessment
Author: Amare Dergu Mengstie | FH Upper Austria 2026

Research Questions answered:
  RQ1 — Metric-defect correlations (static + process metrics)
  RQ2 — LLM score validity (variance, coverage, discriminative power)
  RQ3 — Classification performance: metrics-only vs metrics+LLM
  RQ4 — Cross-repository generalisation (leave-one-repo-out)
         → Run once all 4 repos are fully scored

INPUT:
  Single repo  : data/04_llm_temp/django_2020_2025_PARTIAL_SCORED.parquet
  All 4 repos  : set REPOS list below (uncomment when ready)

OUTPUT:
  results/rq1_correlations.csv
  results/rq2_llm_validity.csv
  results/rq3_classification.csv
  results/rq4_crossrepo.csv        (requires all 4 repos)
  results/stage4_summary.txt
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (roc_auc_score, f1_score,
                                    precision_score, recall_score,
                                    classification_report)
import lightgbm as lgb

# ── Configuration ──────────────────────────────────────────────────────────────

# Single repo (partial — use while scorer is still running)
SCORED_FILES = [
    "data/04_llm_temp/django_2020_2025_PARTIAL_SCORED.parquet",
]

# All 4 repos — uncomment when scoring is complete
# SCORED_FILES = [
#     "data/04_llm_temp/django_2020_2025_FINAL_LABELED_12D_LLM_GROQ_v4.parquet",
#     "data/04_llm_temp/fastapi_2020_2025_FINAL_LABELED_12D_LLM_GROQ_v4.parquet",
#     "data/04_llm_temp/numpy_2020_2025_FINAL_LABELED_12D_LLM_GROQ_v4.parquet",
#     "data/04_llm_temp/pandas_2020_2025_FINAL_LABELED_12D_LLM_GROQ_v4.parquet",
# ]

OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

STATIC_FEATURES = [
    "loc", "cyclomatic_complexity", "num_functions",
    "avg_cc_per_function", "max_cc",
]

PROCESS_FEATURES = [
    "num_revisions", "num_authors", "age_days", "churn",
    "code_ownership", "num_bug_fixes",
]

LLM_DIMS = [
    "llm_readability", "llm_naming", "llm_structure",
    "llm_duplication", "llm_documentation", "llm_complexity",
    "llm_error_handling", "llm_testability", "llm_maintainability",
    "llm_anti_patterns", "llm_security", "llm_style",
]

COMPOSITE    = ["llm_quality_score_12d"]
METRICS_ONLY = STATIC_FEATURES + PROCESS_FEATURES
METRICS_LLM  = STATIC_FEATURES + PROCESS_FEATURES + LLM_DIMS + COMPOSITE

TARGET = "is_ever_defective"

# ── Helper functions ───────────────────────────────────────────────────────────

def precision_at_k(y_true, y_prob, k=0.20):
    """Precision@K%: among top K% ranked by prob, fraction truly defective."""
    n_top   = max(1, int(len(y_true) * k))
    top_idx = np.argsort(y_prob)[::-1][:n_top]
    return y_true.iloc[top_idx].mean()


def evaluate_classifier(name, model, X_train, X_test, y_train, y_test,
                         scaler=None):
    """Train, predict, return metrics dict."""
    if scaler:
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    auc   = roc_auc_score(y_test, y_prob)
    f1    = f1_score(y_test, y_pred, zero_division=0)
    p20   = precision_at_k(y_test.reset_index(drop=True),
                           pd.Series(y_prob), k=0.20)

    return {"classifier": name, "auc_roc": round(auc,4),
            "f1": round(f1,4), "precision_at_20pct": round(p20,4)}


def run_classifiers(X_train, X_test, y_train, y_test, feature_set_name):
    """Run all 3 classifiers on a feature set."""
    results = []
    pos_rate = y_train.mean()
    cw = "balanced" if pos_rate < 0.1 else None

    classifiers = [
        ("LogisticRegression",
         LogisticRegression(max_iter=1000, class_weight=cw, random_state=42),
         StandardScaler()),
        ("RandomForest",
         RandomForestClassifier(n_estimators=200, class_weight=cw,
                                random_state=42, n_jobs=-1),
         None),
        ("LightGBM",
         lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                             class_weight=cw, random_state=42,
                             verbosity=-1, n_jobs=-1),
         None),
    ]

    for clf_name, clf, scaler in classifiers:
        res = evaluate_classifier(
            clf_name, clf,
            X_train.copy(), X_test.copy(),
            y_train, y_test,
            scaler=scaler,
        )
        res["feature_set"] = feature_set_name
        results.append(res)
        print(f"    {clf_name:<22} AUC={res['auc_roc']:.4f}  "
              f"F1={res['f1']:.4f}  P@20%={res['precision_at_20pct']:.4f}")
    return results


# ── Load and aggregate data ────────────────────────────────────────────────────

print("="*70)
print("  STAGE 4 — ML EVALUATION")
print("="*70)

dfs = []
for path in SCORED_FILES:
    if Path(path).exists():
        dfs.append(pd.read_parquet(path))
        print(f"  Loaded : {path}  ({len(dfs[-1]):,} rows)")
    else:
        print(f"  ⚠ Missing: {path} — skipping")

if not dfs:
    raise FileNotFoundError("No scored parquet files found.")

df_raw = pd.concat(dfs, ignore_index=True)
print(f"  Total snapshot rows: {len(df_raw):,}")

# Normalise paths
df_raw["file_path"] = df_raw["file_path"].str.replace("\\", "/", regex=False)

# ── Aggregate to file level ────────────────────────────────────────────────────

print("\n  Aggregating to file level...")

agg = {TARGET: ("is_defective", "max")}

# Static — mean across snapshots
for f in STATIC_FEATURES:
    if f in df_raw.columns:
        agg[f] = (f, "mean")

# Process — last snapshot (cumulative metrics)
for f in PROCESS_FEATURES:
    if f in df_raw.columns:
        agg[f] = (f, "last")

# LLM — mean across scored snapshots
for f in LLM_DIMS + COMPOSITE:
    if f in df_raw.columns:
        agg[f] = (f, "mean")

# Metadata
if "is_test_file" in df_raw.columns:
    agg["is_test_file"] = ("is_test_file", "max")
if "snapshot_date" in df_raw.columns:
    agg["last_snapshot_date"] = ("snapshot_date", "max")
if "repository_name" in df_raw.columns:
    agg["repository_name"] = ("repository_name", "first")

group_cols = ["file_path"]
df_file = df_raw.groupby(group_cols).agg(**agg).reset_index()

# Drop rows where target is missing
df_file = df_file.dropna(subset=[TARGET])
df_file[TARGET] = df_file[TARGET].astype(int)

print(f"  Unique files       : {len(df_file):,}")
print(f"  Defective files    : {df_file[TARGET].sum():,}  "
      f"({df_file[TARGET].mean()*100:.1f}%)")
print(f"  Clean files        : {(df_file[TARGET]==0).sum():,}  "
      f"({(1-df_file[TARGET].mean())*100:.1f}%)")
if "is_test_file" in df_file.columns:
    print(f"  Test files         : {df_file['is_test_file'].sum():,}  "
          f"({df_file['is_test_file'].mean()*100:.1f}%)")

# ── Train/Test Split ───────────────────────────────────────────────────────────
# SPLIT_MODE controls which strategy is used:
#   "stratified" — 80/20 stratified random split (use for partial/single-repo)
#   "temporal"   — train <2024, test >=2024     (use for full 4-repo dataset)
#
# Switch to "temporal" when all 4 repos are fully scored.

SPLIT_MODE = "stratified"   # ← change to "temporal" for full dataset

from sklearn.model_selection import train_test_split

if SPLIT_MODE == "temporal" and "last_snapshot_date" in df_file.columns:
    df_file["last_snapshot_date"] = pd.to_datetime(df_file["last_snapshot_date"])
    train_mask = df_file["last_snapshot_date"] <  "2024-01-01"
    test_mask  = df_file["last_snapshot_date"] >= "2024-01-01"
    df_train   = df_file[train_mask].copy()
    df_test    = df_file[test_mask].copy()

    # Safety check — if distributions diverge >20pp, warn and fall back
    diff = abs(df_train[TARGET].mean() - df_test[TARGET].mean())
    if diff > 0.20:
        print(f"  ⚠ Temporal split: train={df_train[TARGET].mean()*100:.1f}% "
              f"vs test={df_test[TARGET].mean()*100:.1f}% defective")
        print(f"    Distribution gap={diff*100:.1f}pp > 20pp — "
              f"falling back to stratified split")
        SPLIT_MODE = "stratified"

if SPLIT_MODE == "stratified":
    df_train, df_test = train_test_split(
        df_file, test_size=0.20, random_state=42, stratify=df_file[TARGET])
    print(f"\n  Stratified 80/20 split (balanced — for partial/single-repo data):")
else:
    print(f"\n  Temporal split (train <2024 / test >=2024):")

print(f"    Train : {len(df_train):,} files  "
      f"({df_train[TARGET].mean()*100:.1f}% defective)")
print(f"    Test  : {len(df_test):,} files  "
      f"({df_test[TARGET].mean()*100:.1f}% defective)")
print(f"    Split mode used : {SPLIT_MODE}")

# Available features (those present in the data)
avail_metrics = [f for f in METRICS_ONLY if f in df_file.columns]
avail_llm     = [f for f in METRICS_LLM   if f in df_file.columns]

print(f"\n  Features available:")
print(f"    metrics-only : {len(avail_metrics)}")
print(f"    metrics+LLM  : {len(avail_llm)}")

y_train = df_train[TARGET]
y_test  = df_test[TARGET]

# ════════════════════════════════════════════════════════════════════════════════
# RQ1 — Metric-defect correlations
# ════════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  RQ1 — METRIC-DEFECT CORRELATIONS")
print(f"{'='*70}\n")

rq1_rows = []
all_features = avail_llm  # includes static + process + LLM

for feat in all_features:
    vals = df_file[feat].dropna()
    tgt  = df_file.loc[vals.index, TARGET]

    # Pearson
    if vals.std() > 0:
        pear_r, pear_p = stats.pearsonr(vals, tgt)
    else:
        pear_r, pear_p = 0.0, 1.0

    # Spearman
    spear_r, spear_p = stats.spearmanr(vals, tgt)

    # Point-biserial (same as Pearson for binary target)
    pb_r, pb_p = stats.pointbiserialr(tgt, vals)

    # Univariate AUC
    try:
        uauc = roc_auc_score(tgt, vals)
        uauc = max(uauc, 1 - uauc)  # flip if inverted
    except Exception:
        uauc = 0.5

    group = ("static"  if feat in STATIC_FEATURES  else
             "process" if feat in PROCESS_FEATURES  else "llm")

    rq1_rows.append({
        "feature":       feat,
        "group":         group,
        "pearson_r":     round(pear_r, 4),
        "pearson_p":     round(pear_p, 4),
        "spearman_r":    round(spear_r, 4),
        "spearman_p":    round(spear_p, 4),
        "pointbiserial": round(pb_r, 4),
        "univariate_auc":round(uauc, 4),
    })

df_rq1 = pd.DataFrame(rq1_rows).sort_values("univariate_auc", ascending=False)
df_rq1.to_csv(OUT_DIR / "rq1_correlations.csv", index=False)

print(f"  {'Feature':<32} {'Group':<8} {'Pearson':>8} {'Spearman':>9} {'AUC':>7}")
print(f"  {'-'*68}")
for _, row in df_rq1.head(15).iterrows():
    print(f"  {row['feature']:<32} {row['group']:<8} "
          f"{row['pearson_r']:>8.4f} {row['spearman_r']:>9.4f} "
          f"{row['univariate_auc']:>7.4f}")

# Group summary
print(f"\n  Mean |Pearson r| by group:")
for grp in ["static","process","llm"]:
    grp_df = df_rq1[df_rq1["group"]==grp]
    if len(grp_df):
        print(f"    {grp:<10} mean|r|={grp_df['pearson_r'].abs().mean():.4f}  "
              f"mean AUC={grp_df['univariate_auc'].mean():.4f}")

print(f"\n  Saved → results/rq1_correlations.csv")

# ════════════════════════════════════════════════════════════════════════════════
# RQ2 — LLM score validity
# ════════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  RQ2 — LLM SCORE VALIDITY")
print(f"{'='*70}\n")

rq2_rows = []
avail_llm_dims = [f for f in LLM_DIMS if f in df_file.columns]

for dim in avail_llm_dims + COMPOSITE:
    if dim not in df_file.columns:
        continue
    vals = df_file[dim].dropna()
    tgt  = df_file.loc[vals.index, TARGET]

    coverage = vals.notna().mean()
    std      = vals.std()
    # Flat check: >80% in single integer bin
    bins = vals.round(0).value_counts(normalize=True)
    flat = float(bins.max()) > 0.80

    pb_r, pb_p = stats.pointbiserialr(tgt, vals)
    try:
        uauc = roc_auc_score(tgt, vals)
        uauc = max(uauc, 1 - uauc)
    except Exception:
        uauc = 0.5

    rq2_rows.append({
        "dimension":      dim,
        "coverage_pct":   round(coverage*100, 2),
        "mean":           round(vals.mean(), 3),
        "std":            round(std, 3),
        "min":            round(vals.min(), 2),
        "max":            round(vals.max(), 2),
        "flat_flag":      flat,
        "pointbiserial":  round(pb_r, 4),
        "pb_pvalue":      round(pb_p, 4),
        "univariate_auc": round(uauc, 4),
        "significant":    pb_p < 0.05,
    })

df_rq2 = pd.DataFrame(rq2_rows)
df_rq2.to_csv(OUT_DIR / "rq2_llm_validity.csv", index=False)

print(f"  {'Dimension':<32} {'Cover':>6} {'Mean':>6} {'Std':>5} "
      f"{'Flat':>5} {'PB_r':>7} {'AUC':>7} {'Sig':>5}")
print(f"  {'-'*75}")
for _, row in df_rq2.iterrows():
    print(f"  {row['dimension']:<32} {row['coverage_pct']:>6.1f} "
          f"{row['mean']:>6.2f} {row['std']:>5.2f} "
          f"{'YES' if row['flat_flag'] else 'no':>5} "
          f"{row['pointbiserial']:>7.4f} {row['univariate_auc']:>7.4f} "
          f"{'✓' if row['significant'] else '✗':>5}")

sig = df_rq2["significant"].sum()
print(f"\n  Significant dimensions (p<0.05): {sig}/{len(df_rq2)}")
print(f"  Mean AUC across dimensions      : {df_rq2['univariate_auc'].mean():.4f}")
print(f"  Flat dimensions flagged         : {df_rq2['flat_flag'].sum()}")
print(f"\n  Saved → results/rq2_llm_validity.csv")

# ════════════════════════════════════════════════════════════════════════════════
# RQ3 — Classification performance
# ════════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  RQ3 — CLASSIFICATION PERFORMANCE")
print(f"{'='*70}")

all_rq3 = []

for feat_name, feat_list in [("metrics-only", avail_metrics),
                               ("metrics+LLM",  avail_llm)]:
    print(f"\n  Feature set: {feat_name}  ({len(feat_list)} features)")
    X_tr = df_train[feat_list].fillna(0)
    X_te = df_test[feat_list].fillna(0)
    res  = run_classifiers(X_tr, X_te, y_train, y_test, feat_name)
    all_rq3.extend(res)

df_rq3 = pd.DataFrame(all_rq3)
df_rq3.to_csv(OUT_DIR / "rq3_classification.csv", index=False)

print(f"\n  === RQ3 SUMMARY ===")
print(f"\n  {'Classifier':<22} {'Features':<14} {'AUC':>7} {'F1':>7} {'P@20%':>7}")
print(f"  {'-'*60}")
for _, row in df_rq3.iterrows():
    print(f"  {row['classifier']:<22} {row['feature_set']:<14} "
          f"{row['auc_roc']:>7.4f} {row['f1']:>7.4f} "
          f"{row['precision_at_20pct']:>7.4f}")

# Delta: LLM improvement
print(f"\n  === LLM IMPROVEMENT (metrics+LLM vs metrics-only) ===\n")
for clf in df_rq3["classifier"].unique():
    base = df_rq3[(df_rq3["classifier"]==clf) &
                  (df_rq3["feature_set"]=="metrics-only")].iloc[0]
    aug  = df_rq3[(df_rq3["classifier"]==clf) &
                  (df_rq3["feature_set"]=="metrics+LLM")].iloc[0]
    d_auc = aug["auc_roc"] - base["auc_roc"]
    d_f1  = aug["f1"]      - base["f1"]
    d_p20 = aug["precision_at_20pct"] - base["precision_at_20pct"]
    sign  = lambda x: "+" if x >= 0 else ""
    print(f"  {clf:<22}  ΔAUC={sign(d_auc)}{d_auc:.4f}  "
          f"ΔF1={sign(d_f1)}{d_f1:.4f}  "
          f"ΔP@20%={sign(d_p20)}{d_p20:.4f}")

print(f"\n  Saved → results/rq3_classification.csv")

# ════════════════════════════════════════════════════════════════════════════════
# RQ4 — Cross-repo generalisation (only if multiple repos loaded)
# ════════════════════════════════════════════════════════════════════════════════

if "repository_name" in df_file.columns and df_file["repository_name"].nunique() > 1:
    print(f"\n{'='*70}")
    print(f"  RQ4 — CROSS-REPOSITORY GENERALISATION")
    print(f"{'='*70}")

    repos = df_file["repository_name"].unique()
    rq4_rows = []

    for held_out in repos:
        df_tr = df_file[df_file["repository_name"] != held_out]
        df_te = df_file[df_file["repository_name"] == held_out]
        y_tr  = df_tr[TARGET]
        y_te  = df_te[TARGET]

        if y_te.nunique() < 2 or len(df_te) < 10:
            print(f"  ⚠ Skipping {held_out} — insufficient test data")
            continue

        print(f"\n  Held-out: {held_out}  "
              f"(train={len(df_tr)}, test={len(df_te)})")

        for feat_name, feat_list in [("metrics-only", avail_metrics),
                                      ("metrics+LLM",  avail_llm)]:
            X_tr = df_tr[feat_list].fillna(0)
            X_te = df_te[feat_list].fillna(0)

            # LightGBM only for RQ4 (strongest generaliser)
            clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                      random_state=42, verbosity=-1, n_jobs=-1)
            res = evaluate_classifier(
                "LightGBM", clf, X_tr, X_te, y_tr, y_te)
            res["feature_set"]  = feat_name
            res["held_out_repo"] = held_out
            rq4_rows.append(res)
            print(f"    {feat_name:<14} AUC={res['auc_roc']:.4f}  "
                  f"F1={res['f1']:.4f}  P@20%={res['precision_at_20pct']:.4f}")

    if rq4_rows:
        df_rq4 = pd.DataFrame(rq4_rows)
        df_rq4.to_csv(OUT_DIR / "rq4_crossrepo.csv", index=False)
        print(f"\n  Saved → results/rq4_crossrepo.csv")
else:
    print(f"\n  RQ4 skipped — only one repository loaded.")
    print(f"  Run again with all 4 repos scored to get RQ4 results.")

# ════════════════════════════════════════════════════════════════════════════════
# Summary report
# ════════════════════════════════════════════════════════════════════════════════

summary = []
summary.append("STAGE 4 ML EVALUATION — SUMMARY REPORT")
summary.append("="*60)
summary.append(f"Dataset          : {len(df_file):,} unique files")
summary.append(f"Defective        : {df_file[TARGET].sum():,} ({df_file[TARGET].mean()*100:.1f}%)")
summary.append(f"Train/Test split : {len(df_train):,} / {len(df_test):,}")
summary.append(f"Feature sets     : metrics-only ({len(avail_metrics)}) | "
               f"metrics+LLM ({len(avail_llm)})")
summary.append("")
summary.append("RQ1 — Top 5 features by univariate AUC:")
for _, row in df_rq1.head(5).iterrows():
    summary.append(f"  {row['feature']:<32} AUC={row['univariate_auc']:.4f}")
summary.append("")
summary.append("RQ2 — LLM score validity:")
summary.append(f"  Coverage        : {df_rq2['coverage_pct'].mean():.1f}%")
summary.append(f"  Significant dims: {df_rq2['significant'].sum()}/{len(df_rq2)}")
summary.append(f"  Mean AUC        : {df_rq2['univariate_auc'].mean():.4f}")
summary.append("")
summary.append("RQ3 — Best classifier per feature set (by AUC-ROC):")
for fs in ["metrics-only","metrics+LLM"]:
    best = df_rq3[df_rq3["feature_set"]==fs].sort_values("auc_roc",ascending=False).iloc[0]
    summary.append(f"  {fs:<14} {best['classifier']:<22} "
                   f"AUC={best['auc_roc']:.4f}  "
                   f"F1={best['f1']:.4f}  "
                   f"P@20%={best['precision_at_20pct']:.4f}")

report = "\n".join(summary)
(OUT_DIR / "stage4_summary.txt").write_text(report)
print(f"\n{'='*70}")
print(report)
print(f"\n  All results saved to: results/")
print(f"{'='*70}")