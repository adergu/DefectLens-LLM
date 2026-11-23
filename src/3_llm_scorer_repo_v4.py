# ================================================================================
#  LLM-BASED CODE QUALITY SCORER — REPOSITORY EDITION  v4.0
#  AMARE DERGU MENGSTIE — Master's Thesis, FH Upper Austria, 2026
# ================================================================================
#
#  WHAT CHANGED FROM v1 (3_llm_scorer_groq_V1.py):
#  ─────────────────────────────────────────────────
#  ✓ DIMENSIONS: framework_bp → maintainability | magic_values → security
#    (same change as manual scorer v4 — see manual scorer for full rationale)
#
#  ✓ REMOVED: ast.parse() entirely — replaced by lizard + regex
#    (was failing silently on files with Python 2 syntax or encoding issues,
#     causing parse_ok=False and all validation constraints being bypassed)
#
#  ✓ REMOVED: ASTMetrics dataclass, extract_ast_metrics(), ast_summary_str()
#
#  ✓ ADDED: Three-layer validation (NO LLM SCORE IS EVER MODIFIED):
#    Layer 1 — Regex on full source (never fails on any Python file)
#    Layer 2 — Miner parquet static metrics (already computed — just read)
#    Layer 3 — Lizard (4 new features miner did not save)
#    Outputs: val_* consistency columns (audit only, not ML features)
#
#  ✓ ADDED: Framework detection (8 frameworks + generic fallback)
#    FastAPI, Django, NumPy, Pandas, Flask, pytest, sklearn, torch, generic
#    Each detected framework gets its own maintainability scoring rubric
#
#  ✓ CHANGED: Single model (openai/gpt-oss-20b) — removed round-robin
#    Decision made with supervisor. Simpler, more consistent scores.
#    Still supports 5 API keys for quota rotation.
#
#  ✓ CHANGED: MAX_CODE_LEN 6,000 → 14,000 characters
#    Covers 95%+ of real repository files without LLM context degradation
#
#  ✓ CHANGED: Output schema — clean 37-column design (same as manual v4):
#    5 identifiers + 15 miner columns + 12 LLM scores + composite +
#    11 vf_ validation features + 7 val_ consistency flags
#    (see OUTPUT SCHEMA section below)
#
#  ✓ KEPT: All v1 infrastructure unchanged:
#    PyDriller source code extraction + cache parquet (critical for repos)
#    5-key quota rotation, rate limiting, quota-sleep timeout fix,
#    checkpoint/resume, failure re-run pass, multi-repo loop,
#    Ctrl+C handler, daily log, raw response log, audit log
#
#  OUTPUT SCHEMA (37 columns):
#  ────────────────────────────
#  From miner parquet (20 cols):
#    repository_name, file_path, commit_hash, commit_date, snapshot_date
#    loc, cyclomatic_complexity, num_functions, avg_cc_per_function, max_cc
#    num_revisions, num_authors, age_days, churn, code_ownership,
#    minor_contributors, num_bug_fixes, num_refactors,
#    recent_churn_90d, recent_revisions_90d
#  From labeler (1 col):
#    is_defective
#  LLM scores (13 cols):
#    llm_readability, llm_naming, llm_structure, llm_duplication,
#    llm_documentation, llm_complexity, llm_error_handling, llm_testability,
#    llm_maintainability, llm_anti_patterns, llm_security, llm_style,
#    llm_quality_score_12d
#  Validation features (11 cols, vf_ prefix):
#    vf_num_docstrings, vf_num_type_hints, vf_num_try_except, vf_num_raise,
#    vf_num_bare_except, vf_num_magic_numbers, vf_num_global,
#    vf_has_hardcoded_secret, vf_has_md5_password,
#    vf_liz_max_params, vf_liz_long_funcs
#  Consistency flags (7 cols, val_ prefix):
#    val_error_handling_consistent, val_security_consistent,
#    val_anti_patterns_consistent, val_maintainability_consistent,
#    val_consistency_score (0–4), val_high_confidence (bool),
#    val_detected_framework
#
#  ENVIRONMENT VARIABLES:
#  ───────────────────────
#  $env:GROQ_API_KEY1="gsk_xxxx"   (PowerShell)
#  ... up to GROQ_API_KEY5
#  At least ONE key required.
#
#  INSTALL:
#  ─────────
#  pip install requests pandas pyarrow tqdm lizard pydriller
#
#  RUN ORDER:
#  ──────────
#  1. Stage 1 miner  → produces CLEAN.parquet
#  2. Stage 2 labeler → produces FINAL_LABELED.parquet
#  3. This script     → produces *_12D_LLM_GROQ_v4.parquet + .csv
#
#  TO TEST ON TOP-50 SUBSET:
#  ──────────────────────────
#  Set parquet to: data/03_labeled/django_2020_2025_FINAL_LABELED_TOP50_LABELED.parquet
#  git_repo stays as:  data/01_raw_data/django
#  (PyDriller still needs full git history for source code extraction)
# ================================================================================

import os
import sys
import time
import json
import hashlib
import pickle
import re
import signal
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional

import lizard
import pandas as pd
import requests
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
#    MULTI-REPOSITORY CONFIG  
#  To test on the TOP-50 subset, set parquet to the TOP50_LABELED file.
#  To run full repos, uncomment the full-dataset entries.
# ══════════════════════════════════════════════════════════════════════════════

REPOSITORIES = [
    # ── TOP-50 test subset (commented out — full repos active) ────────────────
    # {
    #     "parquet":  "data/03_labeled/django_2020_2025_FINAL_LABELED_TOP50_LABELED.parquet",
    #     "git_repo": "data/01_raw_data/django",
    # },
    # ── FULL REPOS ────────────────────────────────────────────────────────────
    {
        "parquet":  "data/03_labeled/django_2020_2025_FINAL_LABELED.parquet",
        "git_repo": "data/01_raw_data/django",
    },
    {
        "parquet":  "data/03_labeled/fastapi_2020_2025_FINAL_LABELED.parquet",
        "git_repo": "data/01_raw_data/fastapi",
    },
    {
        "parquet":  "data/03_labeled/numpy_2020_2025_FINAL_LABELED.parquet",
        "git_repo": "data/01_raw_data/numpy",
    },
    {
        "parquet":  "data/03_labeled/scikit_learn_2020_2025_FINAL_LABELED.parquet",
        "git_repo": "data/01_raw_data/scikit-learn",
    },
    {
        "parquet":  "data/03_labeled/flask_2020_2025_FINAL_LABELED.parquet",
        "git_repo": "data/01_raw_data/flask",
    },
    {
        "parquet":  "data/03_labeled/matplotlib_2020_2025_FINAL_LABELED.parquet",
        "git_repo": "data/01_raw_data/matplotlib",
    },
    # {
    #     "parquet":  "data/03_labeled/sqlalchemy_2020_2025_FINAL_LABELED.parquet",
    #     "git_repo": "data/01_raw_data/sqlalchemy",
    # },
]

# ══════════════════════════════════════════════════════════════════════════════
#  API KEYS
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEYS = []
# Load all GROQ_API_KEY1 ... GROQ_API_KEY99 automatically — no hardcoded limit
for _i in range(1, 100):
    _name = f"GROQ_API_KEY{_i}"
    _val  = os.environ.get(_name)
    if _val and _val.strip():
        GROQ_API_KEYS.append((_name, _val.strip()))
    elif _i > 5 and not _val:
        break   # stop scanning after first gap beyond KEY5

# Deduplicate — same key value from two env vars wastes one quota slot
_seen_vals = set()
_deduped   = []
for _name, _val in GROQ_API_KEYS:
    if _val not in _seen_vals:
        _seen_vals.add(_val)
        _deduped.append((_name, _val))
    else:
        print(f"  ⚠ Duplicate key value on {_name} — skipping (same quota as another key)")
GROQ_API_KEYS = _deduped

if not GROQ_API_KEYS:
    sys.exit(
        "\n[FATAL] No Groq API keys found.\n"
        "  PowerShell: $env:GROQ_API_KEY1='gsk_xxxx'\n"
    )

if not REPOSITORIES:
    sys.exit("\n[FATAL] REPOSITORIES list is empty.\n")

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL PATHS
# ══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR = Path("data/04_llm_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LLM_CACHE  = OUTPUT_DIR / "_llm_response_cache_repo_v4.json"
DAILY_LOG  = OUTPUT_DIR / "_daily_progress_log_repo_v4.jsonl"

# ══════════════════════════════════════════════════════════════════════════════
#  GROQ CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
#GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL = "openai/gpt-oss-20b"

MAX_CODE_LEN                 = 14_000  # raised from 6,000
SAVE_EVERY                   = 10
REQUEST_TIMEOUT              = 30
MAX_PARSE_RETRIES            = 3
MAX_INNER_ATTEMPTS           = 5
MAX_429_WAIT                 = 30
FILE_TIMEOUT_SECONDS         = 480
REQUESTS_PER_MINUTE          = 25
MIN_SECONDS_BETWEEN_REQUESTS = 60.0 / REQUESTS_PER_MINUTE
SLEEP_DETECTION_GAP          = 300
QUOTA_EXHAUSTED_THRESHOLD    = 300

DIMENSIONS = [
    "readability", "naming", "structure", "duplication", "documentation",
    "complexity", "error_handling", "testability", "maintainability",
    "anti_patterns", "security", "style",
]

# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION THRESHOLDS
# ══════════════════════════════════════════════════════════════════════════════

VAL = {
    # error_handling
    "err_cap"             : 3.0,
    "bare_except_min"     : 3,
    "bare_except_score"   : 6.0,
    # security
    "secret_sec_score"    : 5.0,
    "md5_sec_score"       : 6.0,
    "bare_sec_min"        : 5,
    "bare_sec_score"      : 5.0,
    # anti_patterns
    "global_threshold"    : 3,
    "global_score"        : 6.0,
    "secret_score"        : 5.0,
    # maintainability (uses miner CC — critical for repository files)
    "very_high_cc"        : 40,
    "high_max_cc"         : 20,
    "maint_score_high"    : 7.0,
    "maint_score_very"    : 5.0,
    # framework detection
    "framework_min_markers": 2,
}

# ══════════════════════════════════════════════════════════════════════════════
#  FRAMEWORK DETECTION
# ══════════════════════════════════════════════════════════════════════════════

FRAMEWORK_PATTERNS = {
    "fastapi": [
        r"\bAPIRouter\b", r"\bDepends\b", r"\bHTTPException\b",
        r"\bBaseModel\b", r"from fastapi", r"@router\.",
    ],
    "django": [
        r"\bmodels\.Model\b", r"\bQuerySet\b", r"\bHttpResponse\b",
        r"request\.(GET|POST)", r"from django", r"\bget_object_or_404\b",
        r"\bmodels\.CharField\b", r"\bmodels\.ForeignKey\b",
    ],
    "numpy": [
        r"\bnp\.\w+", r"\bndarray\b", r"import numpy",
        r"from numpy", r"\bnp\.array\b", r"\bnp\.zeros\b",
    ],
    "pandas": [
        r"\bpd\.\w+", r"\bDataFrame\b", r"\bSeries\b",
        r"import pandas", r"from pandas", r"\.groupby\(",
        r"\.loc\[", r"\.iloc\[",
    ],
    "flask": [
        r"\bFlask\b", r"@app\.route", r"from flask",
        r"\bblueprint\b", r"\brender_template\b", r"\burl_for\b",
    ],
    "pytest": [
        r"\bpytest\b", r"@pytest\.fixture", r"def test_\w+",
        r"from pytest", r"pytest\.raises", r"pytest\.mark",
    ],
    "sklearn": [
        r"from sklearn", r"import sklearn", r"\.fit\(", r"\.predict\(",
        r"\bBaseEstimator\b", r"\bTransformerMixin\b",
    ],
    "torch": [
        r"\btorch\b", r"\bnn\.Module\b", r"import torch",
        r"from torch", r"\.cuda\(\)", r"\.backward\(",
    ],
}

FRAMEWORK_RUBRICS = {
    "fastapi" : (
        "FastAPI: score correct use of APIRouter, Depends for DI, "
        "response_model on endpoints, HTTPException for errors, "
        "Pydantic BaseModel schemas, proper HTTP status codes. "
        "High = all used correctly."
    ),
    "django"  : (
        "Django: score correct ORM usage (QuerySet, select_related, "
        "prefetch_related), class-based views, get_object_or_404, "
        "login_required, proper form validation. High = idiomatic Django."
    ),
    "numpy"   : (
        "NumPy: score vectorised operations over Python loops, correct dtype, "
        "broadcasting, np.testing for assertions, no unnecessary copies. "
        "High = fully vectorised, idiomatic NumPy."
    ),
    "pandas"  : (
        "Pandas: score method chaining, vectorised ops over .apply(), "
        "explicit .copy(), correct NA handling, avoid iterrows(). "
        "High = idiomatic pandas."
    ),
    "flask"   : (
        "Flask: score blueprints, url_for(), render_template(), error handlers, "
        "application factory pattern. High = idiomatic Flask."
    ),
    "pytest"  : (
        "pytest: score fixture usage, parametrize, pytest.raises(), "
        "meaningful names, conftest.py. High = fully idiomatic pytest."
    ),
    "sklearn" : (
        "scikit-learn: score Pipeline usage, fit/transform/predict separation, "
        "custom estimators, no data leakage. High = production sklearn."
    ),
    "torch"   : (
        "PyTorch: score nn.Module subclassing, forward() design, device handling, "
        "DataLoader, zero_grad/backward/step. High = idiomatic PyTorch."
    ),
    "generic" : (
        "Generic Python: score how easy this code would be to CHANGE SAFELY "
        "in the future. Consider: are functions small and focused? Are "
        "dependencies explicit? Can you modify one part without breaking others? "
        "CRITICAL: god classes, global mutable state, 8+ parameter functions, "
        "or deeply nested logic MUST score LOW (0-3). "
        "SCALE: 8-10=safe to extend, 5-7=careful refactoring needed, "
        "2-4=high risk of regression, 0-1=unmaintainable."
    ),
}


def detect_framework(source: str) -> tuple:
    """Return (framework_name, marker_count). Falls back to 'generic'."""
    best_name  = "generic"
    best_count = 0
    for name, patterns in FRAMEWORK_PATTERNS.items():
        count = sum(1 for p in patterns if re.search(p, source))
        if count > best_count:
            best_count = count
            best_name  = name
    if best_count < VAL["framework_min_markers"]:
        best_name = "generic"
    return best_name, best_count

# ══════════════════════════════════════════════════════════════════════════════
#  KEY STATE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

_key_state_lock = threading.Lock()
_key_states = {name: {"exhausted": False, "exhausted_until": 0.0}
               for name, _ in GROQ_API_KEYS}

def _mark_key_exhausted(key_name: str):
    wait = seconds_until_groq_reset()
    with _key_state_lock:
        _key_states[key_name]["exhausted"]       = True
        _key_states[key_name]["exhausted_until"] = time.time() + wait
    print(f"\n  🚫 Key {key_name} daily quota exhausted.")

def _is_key_available(key_name: str) -> bool:
    with _key_state_lock:
        s = _key_states[key_name]
        if not s["exhausted"]:
            return True
        if time.time() >= s["exhausted_until"]:
            s["exhausted"] = False; s["exhausted_until"] = 0.0
            print(f"\n  ✅ Key {key_name} quota reset.")
            return True
        return False

def _get_available_keys() -> list:
    return [(n, k) for n, k in GROQ_API_KEYS if _is_key_available(n)]

def _all_keys_exhausted() -> bool:
    return len(_get_available_keys()) == 0

# ── Quota-sleep flag (prevents file timeouts during quota reset) ──────────────
_quota_sleeping      = False
_quota_sleeping_lock = threading.Lock()

def _set_quota_sleeping(value: bool):
    global _quota_sleeping
    with _quota_sleeping_lock:
        _quota_sleeping = value

def _is_quota_sleeping() -> bool:
    with _quota_sleeping_lock:
        return _quota_sleeping

# ══════════════════════════════════════════════════════════════════════════════
#  RATE LIMITER
# ══════════════════════════════════════════════════════════════════════════════

_rate_lock      = threading.Lock()
_last_call_time = 0.0

def rate_limited_sleep():
    global _last_call_time
    with _rate_lock:
        now     = time.time()
        elapsed = now - _last_call_time
        if elapsed > SLEEP_DETECTION_GAP and _last_call_time > 0:
            print(f"\n  💤 Wake-up detected ({elapsed/60:.1f} min) — resetting rate limiter.")
            _last_call_time = now - MIN_SECONDS_BETWEEN_REQUESTS
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.time() - _last_call_time)
        _last_call_time = time.time()
    if wait > 0:
        time.sleep(wait)

# ══════════════════════════════════════════════════════════════════════════════
#  KEY SELECTOR (round-robin across available keys, single model)
# ══════════════════════════════════════════════════════════════════════════════

_slot_index      = 0
_slot_index_lock = threading.Lock()

def next_key() -> tuple:
    global _slot_index
    available = _get_available_keys()
    if not available:
        return "", ""
    with _slot_index_lock:
        pair = available[_slot_index % len(available)]
        _slot_index += 1
    return pair

def advance_key():
    global _slot_index
    with _slot_index_lock:
        _slot_index += 1

# ══════════════════════════════════════════════════════════════════════════════
#  QUOTA MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

def seconds_until_groq_reset() -> int:
    now_utc   = datetime.now(timezone.utc)
    reset_utc = (now_utc + timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return int((reset_utc - now_utc).total_seconds()) + 60

PROBE_INTERVAL_SECONDS = 30 * 60   # probe every 30 minutes

def _probe_keys_alive() -> bool:
    """
    Send one lightweight test request per key.
    Returns True if at least one key responds with 200 or 429 (quota-limited
    but alive — meaning the key is back and will work on the next real request).
    Updates _key_states in-place: exhausted keys that respond are un-marked.
    """
    any_alive = False
    for key_name, api_key in GROQ_API_KEYS:
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "Reply with exactly: OK"},
                        {"role": "user",   "content": "ping"},
                    ],
                    "temperature": 0.0,
                    "max_tokens":  5,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                # Key is fully working — un-exhaust it
                with _key_state_lock:
                    _key_states[key_name]["exhausted"]       = False
                    _key_states[key_name]["exhausted_until"] = 0.0
                print(f"  ✅ {key_name} is alive (200) — quota reset!")
                any_alive = True
            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", 9999))
                if retry_after <= QUOTA_EXHAUSTED_THRESHOLD:
                    # Short 429 = rate limit, not daily quota — key is back
                    with _key_state_lock:
                        _key_states[key_name]["exhausted"]       = False
                        _key_states[key_name]["exhausted_until"] = 0.0
                    print(f"  ✅ {key_name} is alive (429 short) — quota reset!")
                    any_alive = True
                else:
                    print(f"  ⏳ {key_name} still exhausted (retry-after={retry_after}s)")
            else:
                print(f"  ⚠ {key_name} returned HTTP {resp.status_code}")
        except Exception as exc:
            print(f"  ⚠ {key_name} probe failed: {exc}")
    return any_alive


def handle_all_keys_exhausted():
    """
    Instead of sleeping until midnight, probe every 30 minutes.
    The moment any key comes back, stop waiting and resume scoring.
    This recovers as soon as Groq resets (which often happens before midnight).
    """
    wait_max = seconds_until_groq_reset()
    wake_at  = datetime.now() + timedelta(seconds=wait_max)
    print(f"\n  🚫 ALL {len(GROQ_API_KEYS)} KEYS EXHAUSTED.")
    print(f"  🔄 Probing every 30 min — max wait until "
          f"{wake_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  💡 Will resume as soon as any key responds. Safe to leave running.\n")
    _log_daily("all_keys_exhausted", {
        "wait_sec": wait_max, "resume_at": wake_at.isoformat()})

    _set_quota_sleeping(True)
    deadline    = time.time() + wait_max
    probe_count = 0

    while time.time() < deadline:
        # Sleep 30 minutes in small chunks so Ctrl+C still works
        sleep_end = time.time() + PROBE_INTERVAL_SECONDS
        while time.time() < sleep_end and time.time() < deadline:
            time.sleep(5)

        probe_count += 1
        now_str = datetime.now().strftime('%H:%M:%S')
        print(f"\n  🔍 Probe #{probe_count} at {now_str} — testing all keys…")

        if _probe_keys_alive():
            # At least one key is back — stop sleeping
            break
        else:
            remaining = int(deadline - time.time())
            print(f"  ⏳ All keys still exhausted. "
                  f"Next probe in 30 min (max {remaining//3600}h {(remaining%3600)//60}m left).")

    _set_quota_sleeping(False)

    # Final safety: un-exhaust any key whose timer has expired
    with _key_state_lock:
        for name, _ in GROQ_API_KEYS:
            if _key_states[name]["exhausted"] and \
               time.time() >= _key_states[name]["exhausted_until"]:
                _key_states[name]["exhausted"]       = False
                _key_states[name]["exhausted_until"] = 0.0

    print(f"\n  ✅ Resuming at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_daily("all_keys_reset", {"probe_count": probe_count})

def _log_daily(event: str, data: dict):
    entry = {"event": event, "timestamp": datetime.now().isoformat(), **data}
    try:
        with open(DAILY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION FEATURES  (lizard + regex — replaces ast.parse, never fails)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationFeatures:
    # validates llm_documentation
    num_docstrings:       int   = 0
    num_type_hints:       int   = 0
    # validates llm_error_handling
    num_try_except:       int   = 0
    num_raise:            int   = 0
    num_bare_except:      int   = 0
    # kept for anti_patterns context
    num_magic_numbers:    int   = 0
    # validates llm_anti_patterns
    num_global:           int   = 0
    has_hardcoded_secret: bool  = False
    # validates llm_security
    has_md5_password:     bool  = False
    # validates llm_testability (lizard — not in miner)
    liz_max_params:       int   = 0
    # validates llm_structure + llm_maintainability (lizard — not in miner)
    liz_long_funcs:       int   = 0
    # framework detection
    detected_framework:   str   = "generic"
    framework_markers:    int   = 0


def _strip_docs_and_comments(source: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '""', source)
    src = re.sub(r'#[^\n]*', '', src)
    return src


def extract_validation_features(source: str) -> ValidationFeatures:
    """Extract all validation features. Never raises SyntaxError."""
    f     = ValidationFeatures()
    clean = _strip_docs_and_comments(source)

    f.detected_framework, f.framework_markers = detect_framework(source)

    # documentation
    f.num_docstrings = len(re.findall(
        r'^\s*(?:"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')',
        source, re.MULTILINE
    ))
    f.num_type_hints = len(re.findall(
        r':\s*(?:str|int|float|bool|list|dict|List|Dict|Optional|'
        r'Union|Any|Tuple|Set|FrozenSet|Sequence|Callable|Iterator)\b',
        source
    ))

    # error handling
    f.num_try_except  = len(re.findall(r'\bexcept\b', source))
    f.num_raise       = len(re.findall(r'\braise\b', source))
    f.num_bare_except = len(re.findall(r'except\s*:', source))

    # magic numbers (clean source — no docstrings/comments)
    f.num_magic_numbers = len(set(re.findall(
        r'(?<![.\w])\b(?!(?:0|1|2|10|100)\b)\d+(?:\.\d+)?\b', clean
    )))

    # anti-patterns + security
    f.num_global = len(re.findall(r'\bglobal\b', source))
    f.has_hardcoded_secret = bool(re.search(
        r'(?i)(password|secret|token|api_key|apikey)\s*=\s*["\'][^"\']{4,}["\']',
        source
    ))
    f.has_md5_password = bool(re.search(r'hashlib\.(md5|sha1)\(', source))

    # lizard: features the miner did NOT save
    try:
        ana   = lizard.analyze_file.analyze_source_code("tmp.py", source)
        funcs = ana.function_list
        if funcs:
            f.liz_max_params = max(len(fn.parameters) for fn in funcs)
            f.liz_long_funcs = sum(1 for fn in funcs if fn.length > 30)
    except Exception:
        pass

    return f


def validation_features_summary(f: ValidationFeatures) -> str:
    """Compact string passed into LLM prompt as pre-computed context."""
    return (
        f"docstrings={f.num_docstrings} | type_hints={f.num_type_hints} | "
        f"try_except={f.num_try_except} | raise={f.num_raise} | "
        f"bare_except={f.num_bare_except} | "
        f"magic_numbers={f.num_magic_numbers}(excl.docstrings) | "
        f"global_stmts={f.num_global} | "
        f"hardcoded_secret={f.has_hardcoded_secret} | "
        f"md5_password={f.has_md5_password} | "
        f"max_params={f.liz_max_params} | long_funcs(>30lines)={f.liz_long_funcs} | "
        f"framework={f.detected_framework}({f.framework_markers} markers)"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  CONSISTENCY VALIDATION  (no LLM score is ever modified)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsistencyResult:
    """
    Four active validation checks.
    val_consistency_score: 0–4 (4 = all pass = high confidence)
    LLM scores are NEVER modified — these are audit flags only.
    """
    val_error_handling_consistent:  bool = True
    val_security_consistent:        bool = True
    val_anti_patterns_consistent:   bool = True
    val_maintainability_consistent: bool = True
    val_consistency_score:          int  = 4
    val_high_confidence:            bool = True
    val_detected_framework:         str  = "generic"


def compute_consistency(
    scores:      dict,
    feat:        ValidationFeatures,
    miner_cc:    float = 0.0,
    miner_max_cc:float = 0.0,
) -> ConsistencyResult:
    """
    Cross-check LLM scores against static evidence.
    Returns consistency flags — no score is modified.
    Inconsistency = LLM gave HIGH score where evidence says code cannot score that high.
    """
    c = ConsistencyResult(val_detected_framework=feat.detected_framework)
    has_error_handling = feat.num_try_except > 0 or feat.num_raise > 0

    # 1. error_handling
    c.val_error_handling_consistent = not (
        (not has_error_handling and scores["error_handling"] > VAL["err_cap"]) or
        (feat.num_bare_except >= VAL["bare_except_min"] and
         scores["error_handling"] >= VAL["bare_except_score"])
    )

    # 2. security
    c.val_security_consistent = not (
        (feat.has_hardcoded_secret and
         scores["security"] >= VAL["secret_sec_score"]) or
        (feat.has_md5_password and
         scores["security"] >= VAL["md5_sec_score"]) or
        (feat.num_bare_except >= VAL["bare_sec_min"] and
         scores["security"] >= VAL["bare_sec_score"])
    )

    # 3. anti_patterns
    c.val_anti_patterns_consistent = not (
        (feat.num_global >= VAL["global_threshold"] and
         scores["anti_patterns"] >= VAL["global_score"]) or
        (feat.has_hardcoded_secret and
         scores["anti_patterns"] >= VAL["secret_score"])
    )

    # 4. maintainability (uses miner CC — key for repository files)
    c.val_maintainability_consistent = not (
        (miner_cc > VAL["very_high_cc"] and
         scores["maintainability"] >= VAL["maint_score_very"]) or
        (miner_max_cc > VAL["high_max_cc"] and
         scores["maintainability"] >= VAL["maint_score_high"]) or
        (feat.liz_long_funcs >= 3 and
         scores["maintainability"] >= VAL["maint_score_high"])
    )

    c.val_consistency_score = sum([
        c.val_error_handling_consistent,
        c.val_security_consistent,
        c.val_anti_patterns_consistent,
        c.val_maintainability_consistent,
    ])
    c.val_high_confidence = (c.val_consistency_score == 4)
    return c

# ══════════════════════════════════════════════════════════════════════════════
#  LLM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a strict, research-grade code quality auditor. "
    "You score code HONESTLY using the full 0-10 range — most files score 4-7. "
    "You NEVER give 8+ unless the code is genuinely exceptional. "
    "You ALWAYS respond with ONLY valid JSON. No markdown, no backticks, no explanation."
)

PROMPT_V0 = """\
You are a strict, unbiased code quality auditor doing research-grade analysis.
Your job is to DISCRIMINATE between files — not to be polite.

SCORING SCALE (apply strictly):
  9.0–10.0 = EXCEPTIONAL: production-grade, best practices fully applied
  7.0–8.9  = GOOD: solid code but has clear, specific gaps
  5.0–6.9  = AVERAGE: typical open-source code, notable missing practices
  3.0–4.9  = POOR: missing key practices, hard to maintain
  0.0–2.9  = VERY POOR: serious quality problems
  → Most real-world files score 4–7. Only exceptional files score 8+.

PRE-COMPUTED STATIC METRICS (use to calibrate your scores):
{features_summary}

DETECTED FRAMEWORK: {framework}
FRAMEWORK SCORING RUBRIC FOR 'maintainability':
{framework_rubric}

DIMENSIONS — score each 0.0–10.0 using EXACT JSON keys:
  readability      — logic clarity, no deep nesting, easy to follow
  naming           — PEP 8, descriptive identifiers, no single-letter vars
  structure        — separation of concerns, modularity, file organisation
  duplication      — DRY principle; penalise repeated blocks and copy-paste
  documentation    — docstrings present, type hints used, comments explain WHY
  complexity       — LOW cyclomatic complexity = HIGH score; simple = good
  error_handling   — try/except OR raise used, specific exception types,
                     bare 'except:' clauses are a strong NEGATIVE signal
  testability      — pure/mockable functions, no hidden global state
  maintainability  — how easy is it to safely CHANGE this code in the future?
                     Score LOW for: tight coupling, hidden dependencies, functions
                     that do too many things. Score HIGH for: isolated changes,
                     clear interfaces. Use CC and long_funcs metrics to calibrate.
                     SCALE: 8-10=safe to extend, 5-7=moderate effort,
                     2-4=risky to change, 0-1=unmaintainable.
  anti_patterns    — ABSENCE of: god classes, global mutable state, hardcoded
                     secrets. HIGH score = code is CLEAN.
  security         — absence of security vulnerabilities. Score LOW for:
                     hardcoded secrets/tokens, MD5/SHA1 for passwords, eval()/exec(),
                     bare excepts hiding failures. Score HIGH for: none of these.
                     SCALE: 8-10=no issues, 5-7=minor concern, 2-4=two concerns,
                     0-1=critical vulnerabilities.
  style            — PEP 8 compliance, spacing, import ordering

CRITICAL SCORING RULES:
  anti_patterns: 10.0=ZERO problems (GOOD), 0.0=FULL of problems (BAD)
  security:      10.0=ZERO vulnerabilities (GOOD), 0.0=many vulnerabilities (BAD)
  maintainability: if CC>40 → score MUST be <5.0; if max_cc>20 → score MUST be <7.0
  If metrics show hardcoded secret or MD5 password → security MUST be <5.0

File: {path}
Code:
{code}

Respond with ONLY this JSON:
{{
  "readability": <float 0.0-10.0>,
  "naming": <float 0.0-10.0>,
  "structure": <float 0.0-10.0>,
  "duplication": <float 0.0-10.0>,
  "documentation": <float 0.0-10.0>,
  "complexity": <float 0.0-10.0>,
  "error_handling": <float 0.0-10.0>,
  "testability": <float 0.0-10.0>,
  "maintainability": <float 0.0-10.0>,
  "anti_patterns": <float 0.0-10.0>,
  "security": <float 0.0-10.0>,
  "style": <float 0.0-10.0>
}}
"""

PROMPT_V1 = """\
Strict code quality scoring. Most files score 4–7. Only exceptional code scores 8+.

CRITICAL RULES:
  anti_patterns=10 means ZERO anti-patterns (GOOD). 0=full of problems (BAD).
  security=10 means ZERO vulnerabilities (GOOD). 0=many vulnerabilities (BAD).
  maintainability: if CC>40 → score<5.0; if max_cc>20 → score<7.0.
  If hardcoded secret or MD5 password → security MUST be <5.0.

Metrics: {features_summary}
Framework: {framework} — {framework_rubric}

File: {path}
Code:
{code}

Return ONLY valid JSON:
{{"readability":0.0,"naming":0.0,"structure":0.0,"duplication":0.0,\
"documentation":0.0,"complexity":0.0,"error_handling":0.0,"testability":0.0,\
"maintainability":0.0,"anti_patterns":0.0,"security":0.0,"style":0.0}}
No markdown. No extra text."""

PROMPT_V2 = """\
Score this Python file. Average=5, good=7, exceptional=9+.
anti_patterns=10 means ZERO anti-patterns (good). 0=many (bad).
security=10 means ZERO vulnerabilities (good). 0=many (bad).
maintainability=score future change safety. High CC means LOW score.
Framework: {framework}
Metrics: {features_summary}
File: {path}
Code: {code}

Output ONLY JSON with keys: readability, naming, structure, duplication,
documentation, complexity, error_handling, testability, maintainability,
anti_patterns, security, style"""

PROMPT_VARIANTS = [PROMPT_V0, PROMPT_V1, PROMPT_V2]

# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScoringResult:
    idx:            int
    file_path:      str
    scores:         Optional[dict]
    confidence:     int
    key_used:       str                = ""
    failure_reason: Optional[str]      = None
    raw_responses:  list               = field(default_factory=list)
    error_types:    list               = field(default_factory=list)
    val_features:   Optional[dict]     = None
    consistency:    Optional[dict]     = None

# ══════════════════════════════════════════════════════════════════════════════
#  LLM RESPONSE CACHE
# ══════════════════════════════════════════════════════════════════════════════

_cache_lock    = threading.Lock()
LLM_CACHE_DATA = {}

def load_llm_cache():
    global LLM_CACHE_DATA
    if LLM_CACHE.exists():
        try:
            LLM_CACHE_DATA = json.loads(LLM_CACHE.read_text(encoding="utf-8"))
            print(f"  ↗ LLM cache loaded: {len(LLM_CACHE_DATA):,} cached responses.")
        except Exception:
            LLM_CACHE_DATA = {}
    else:
        LLM_CACHE_DATA = {}

def save_llm_cache():
    acquired = _cache_lock.acquire(blocking=False)
    if not acquired:
        return
    try:
        with open(LLM_CACHE, "w", encoding="utf-8") as f:
            json.dump(LLM_CACHE_DATA, f, indent=2, ensure_ascii=False)
    finally:
        _cache_lock.release()

def _cache_key(code: str, variant: int) -> str:
    payload = f"groq::repo_v4::{variant}::{GROQ_MODEL}::{code[:MAX_CODE_LEN]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# ══════════════════════════════════════════════════════════════════════════════
#  API CONNECTIVITY TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_api_connection() -> bool:
    print(f"  ↳ Testing {len(GROQ_API_KEYS)} key(s) with {GROQ_MODEL}…")
    any_ok = False
    for key_name, api_key in GROQ_API_KEYS:
        print(f"    {key_name}…", end=" ", flush=True)
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "Reply with exactly: OK"},
                        {"role": "user",   "content": "ping"},
                    ],
                    "temperature": 0.0, "max_tokens": 5,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"].strip()
                print(f"✓  ('{reply}')")
                any_ok = True
            elif resp.status_code == 429:
                print(f"⚠  429 (key works)")
                any_ok = True
            else:
                print(f"✗  HTTP {resp.status_code}")
        except Exception as exc:
            print(f"✗  {exc}")
    return any_ok

# ══════════════════════════════════════════════════════════════════════════════
#  GROQ API CALL
# ══════════════════════════════════════════════════════════════════════════════

def call_groq_raw(prompt: str, code: str, variant: int) -> tuple:
    ck = _cache_key(code, variant)
    with _cache_lock:
        if ck in LLM_CACHE_DATA:
            return LLM_CACHE_DATA[ck], "", GROQ_MODEL, "cache"

    for _attempt in range(MAX_INNER_ATTEMPTS):
        if _all_keys_exhausted():
            handle_all_keys_exhausted()

        key_name, api_key = next_key()
        if not api_key:
            time.sleep(5); continue

        try:
            rate_limited_sleep()
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens":  400,
                },
                timeout=REQUEST_TIMEOUT,
            )

            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"].strip()
                if not text:
                    advance_key(); continue
                with _cache_lock:
                    LLM_CACHE_DATA[ck] = text
                    if len(LLM_CACHE_DATA) % 50 == 0:
                        save_llm_cache()
                return text, "", GROQ_MODEL, key_name

            elif resp.status_code == 429:
                retry_after = 30
                try:
                    retry_after = int(resp.headers.get("retry-after", 30))
                except (ValueError, TypeError):
                    pass
                if retry_after > QUOTA_EXHAUSTED_THRESHOLD:
                    _mark_key_exhausted(key_name)
                else:
                    retry_after = min(retry_after, MAX_429_WAIT)
                    print(f"\n  ⚠ 429 on {key_name} — waiting {retry_after}s…")
                    advance_key(); time.sleep(retry_after)
                continue

            elif resp.status_code in (500, 502, 503, 504):
                time.sleep(10); continue
            else:
                time.sleep(5); continue

        except requests.Timeout:
            print(f"\n  ⏱ Timeout on {key_name} — retrying…")
            advance_key(); time.sleep(5); continue
        except Exception:
            time.sleep(5); continue

    return "", "all_slots_failed", "", ""

# ══════════════════════════════════════════════════════════════════════════════
#  RESPONSE PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_12d(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[^\n]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$",        "", cleaned).strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group(0))
        scores = {}
        for d in DIMENSIONS:
            v = data.get(d)
            if v is None:
                return None
            try:
                scores[d] = max(0.0, min(10.0, round(float(v), 2)))
            except (TypeError, ValueError):
                return None
        return scores
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  LOGGERS
# ══════════════════════════════════════════════════════════════════════════════

_log_lock   = threading.Lock()
_audit_lock = threading.Lock()

def _log_raw(path: str, variant: int, raw: str,
             parsed_ok: bool, raw_log: Path, error_type: str = ""):
    entry = {
        "file": path, "variant": variant, "model": GROQ_MODEL,
        "parsed_ok": parsed_ok, "error_type": error_type,
        "response": raw[:500], "timestamp": datetime.now().isoformat(),
    }
    with _log_lock:
        with open(raw_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

def _log_audit(file_path: str, reason: str, error_types: list,
               raws: list, audit_report: Path):
    entry = {
        "file_path": file_path, "failure_reason": reason,
        "error_types": sorted(set(e for e in error_types if e)),
        "raw_sample": [r[:300] for r in raws],
        "timestamp": datetime.now().isoformat(),
    }
    with _audit_lock:
        audit_report.parent.mkdir(parents=True, exist_ok=True)
        with open(audit_report, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE FILE SCORER
# ══════════════════════════════════════════════════════════════════════════════

def score_single_file(
    idx:          int,
    path:         str,
    code:         str,
    miner_cc:     float,
    miner_max_cc: float,
    raw_log:      Path,
    audit_report: Path,
) -> ScoringResult:
    truncated    = str(code)[:MAX_CODE_LEN]
    feat         = extract_validation_features(truncated)
    feat_summary = validation_features_summary(feat)
    framework    = feat.detected_framework
    rubric       = FRAMEWORK_RUBRICS[framework]

    raw_history  = []
    error_types  = []
    key_used     = ""

    for attempt in range(MAX_PARSE_RETRIES):
        variant = min(attempt, len(PROMPT_VARIANTS) - 1)
        prompt  = PROMPT_VARIANTS[variant].format(
            path=path, code=truncated,
            features_summary=feat_summary,
            framework=framework, framework_rubric=rubric,
        )

        raw, err_type, _, key_used = call_groq_raw(prompt, truncated, variant)
        raw_history.append(raw)
        error_types.append(err_type)
        _log_raw(path, variant, raw, parsed_ok=False,
                 raw_log=raw_log, error_type=err_type)

        scores = parse_12d(raw)
        if scores:
            _log_raw(path, variant, raw, parsed_ok=True, raw_log=raw_log)
            consistency = compute_consistency(
                scores, feat,
                miner_cc=miner_cc,
                miner_max_cc=miner_max_cc,
            )
            return ScoringResult(
                idx=idx, file_path=path, scores=scores,
                confidence=MAX_PARSE_RETRIES - attempt,
                key_used=key_used,
                val_features=asdict(feat),
                consistency=asdict(consistency),
            )

        if err_type == "all_slots_failed":
            # All API keys exhausted mid-file — do NOT save as scored.
            # Return None so this file stays in the retry queue on resume.
            print(f"\n  ⚠ All slots failed: {str(path)[:70]} — will retry on resume")
            return None
        time.sleep(2.0)

    reason = (f"All {MAX_PARSE_RETRIES} attempts failed. "
              f"Errors: {sorted(set(error_types))}")
    _log_audit(path, reason, error_types, raw_history, audit_report)
    return ScoringResult(
        idx=idx, file_path=path, scores=None, confidence=0,
        key_used=key_used, failure_reason=reason,
        raw_responses=raw_history, error_types=error_types,
        val_features=asdict(feat),
    )


def score_single_file_with_timeout(
    idx:          int,
    path:         str,
    code:         str,
    miner_cc:     float,
    miner_max_cc: float,
    raw_log:      Path,
    audit_report: Path,
) -> ScoringResult:
    result_box: list = [None]
    exc_box:    list = [None]

    def _worker():
        try:
            result_box[0] = score_single_file(
                idx, path, code, miner_cc, miner_max_cc, raw_log, audit_report)
        except Exception:
            exc_box[0] = traceback.format_exc()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # Pause-aware deadline — does not count quota-sleep seconds
    active_seconds = 0
    poll_interval  = 1.0
    while t.is_alive():
        time.sleep(poll_interval)
        if not _is_quota_sleeping():
            active_seconds += poll_interval
        if active_seconds >= FILE_TIMEOUT_SECONDS:
            break

    if t.is_alive():
        reason = (f"FORCE TIMEOUT after {FILE_TIMEOUT_SECONDS}s active time — "
                  f"will retry on resume.")
        print(f"\n  ⏱ FORCE TIMEOUT: {str(path)[:70]} — will retry on resume")
        _log_audit(path, reason, ["force_timeout"], [], audit_report)
        # Return None so results[idx] stays None → file retried on resume
        return None

    if exc_box[0]:
        reason = f"Unhandled exception: {exc_box[0]}"
        _log_audit(path, reason, ["exception"], [], audit_report)
        return ScoringResult(
            idx=idx, file_path=path, scores=None, confidence=0,
            failure_reason=reason, error_types=["exception"],
        )

    return result_box[0]

# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE CODE EXTRACTION  (PyDriller + cache to parquet)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_source_code(df: pd.DataFrame, git_repo: str,
                        source_cache: Path) -> pd.DataFrame:
    """
    Attach source_code column by reading from git history via PyDriller.
    Caches results to a parquet so re-runs are instant.
    This is the correct approach for repository files — unlike manual files,
    we cannot just read from disk because we need the exact version at each
    commit hash.
    """
    if source_cache.exists():
        cached = pd.read_parquet(source_cache)
        if "source_code" in cached.columns:
            cached = cached.drop_duplicates(
                subset=["commit_hash", "file_path"], keep="last")
            df_merged = df.merge(cached, on=["commit_hash", "file_path"], how="left")
            df_merged = df_merged.drop_duplicates(
                subset=["commit_hash", "file_path"], keep="first")
            n_matched = df_merged["source_code"].notna().sum()
            coverage  = n_matched / max(len(df_merged), 1)

            if coverage >= 0.80:
                # Cache covers >=80% of rows — valid to use
                df_out = df_merged.dropna(
                    subset=["source_code"]).reset_index(drop=True)
                print(f"  ↗ Source code cache found — skipping PyDriller traversal.")
                print(f"  ↗ {len(df_out):,} / {len(df_merged):,} rows matched "
                      f"({coverage*100:.1f}% coverage).")
                return df_out
            else:
                # Cache was built from a different row set (e.g. pre-filter was
                # added after the cache was created). Delete and rebuild.
                print(f"  ⚠ Cache coverage only {coverage*100:.1f}% "
                      f"({n_matched:,}/{len(df_merged):,} rows matched).")
                print(f"  ↳ Stale cache — deleting and rebuilding from "
                      f"current {len(df):,} filtered rows.")
                source_cache.unlink()
                # fall through to PyDriller rebuild below

    print(f"  ↳ Traversing git repo: {git_repo}…")
    from pydriller import Repository

    code_map: dict = {}
    for commit in tqdm(
        Repository(git_repo).traverse_commits(),
        desc="  Reading commits", unit="commit"
    ):
        for m in commit.modified_files:
            fpath = m.new_path or m.old_path
            if fpath and fpath.endswith(".py") and m.source_code:
                code_map[(commit.hash, fpath)] = m.source_code

    df["source_code"] = df.apply(
        lambda r: code_map.get((r["commit_hash"], r["file_path"])), axis=1)
    df[["commit_hash", "file_path", "source_code"]].to_parquet(
        source_cache, index=False)
    print(f"  ↳ Source cache saved → {source_cache}")
    return df.dropna(subset=["source_code"]).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD OUTPUT  (clean 37-column schema)
# ══════════════════════════════════════════════════════════════════════════════

# Columns to keep from the labeled parquet (all miner + label columns)
PARQUET_KEEP = [
    "repository_name", "file_path", "commit_hash", "commit_date", "snapshot_date",
    "loc", "cyclomatic_complexity", "num_functions", "avg_cc_per_function", "max_cc",
    "num_revisions", "num_authors", "age_days", "churn", "code_ownership",
    "minor_contributors", "num_bug_fixes", "num_refactors",
    "recent_churn_90d", "recent_revisions_90d",
    "is_defective",
]

# vf_ validation features to save (11 columns)
VF_KEEP = [
    "num_docstrings", "num_type_hints",
    "num_try_except", "num_raise", "num_bare_except",
    "num_magic_numbers",
    "num_global", "has_hardcoded_secret", "has_md5_password",
    "liz_max_params", "liz_long_funcs",
]

# val_ consistency columns to save (7 columns)
VAL_KEEP = [
    "val_error_handling_consistent",
    "val_security_consistent",
    "val_anti_patterns_consistent",
    "val_maintainability_consistent",
    "val_consistency_score",
    "val_high_confidence",
    "val_detected_framework",
]


def build_output(df: pd.DataFrame, results: list) -> pd.DataFrame:
    # Start from parquet columns only
    keep_cols = [c for c in PARQUET_KEEP if c in df.columns]
    out = df[keep_cols].copy()

    # LLM scores (12 dimensions)
    for d in DIMENSIONS:
        out[f"llm_{d}"] = [
            (r.scores[d] if r and r.scores else None) for r in results]

    # Composite score
    out["llm_quality_score_12d"] = (
        out[[f"llm_{d}" for d in DIMENSIONS]]
        .mean(axis=1, skipna=True).round(3))

    # vf_ validation features
    for k in VF_KEEP:
        out[f"vf_{k}"] = [
            (r.val_features.get(k) if r and r.val_features else None)
            for r in results]

    # val_ consistency flags
    for col in VAL_KEEP:
        out[col] = [
            (r.consistency.get(col) if r and r.consistency else None)
            for r in results]

    # Drop source code if it crept in
    if "source_code" in out.columns:
        out = out.drop(columns=["source_code"])

    return out

# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(df: pd.DataFrame, results: list, total: int,
                    checkpoint: Path, progress_file: Path):
    with open(checkpoint, "wb") as f:
        pickle.dump({"results": results}, f)
    completed = sum(1 for r in results if r is not None)
    succeeded = sum(1 for r in results if r is not None and r.scores is not None)
    progress_file.write_text(json.dumps({
        "completed": completed, "succeeded": succeeded,
        "failed": completed - succeeded, "total": total,
        "percent": round(completed / total * 100, 2),
        "timestamp": datetime.now().isoformat(),
    }, indent=2))
    save_llm_cache()

# ══════════════════════════════════════════════════════════════════════════════
#  SCORE VALIDITY + CONSISTENCY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def validate_scores(out: pd.DataFrame):
    print("\n" + "═" * 80)
    print("  SCORE VALIDITY + CONSISTENCY REPORT")
    print("═" * 80)

    scored   = out[out["llm_quality_score_12d"].notna()].copy()
    n_scored = len(scored)
    n_total  = len(out)

    print(f"\n  Files total  : {n_total:,}")
    print(f"  Files scored : {n_scored:,}  ({n_scored/n_total*100:.1f}%)")
    print(f"  Files failed : {n_total - n_scored:,}")

    if "llm_key_used" in out.columns:
        kc = out[out["llm_quality_score_12d"].notna()]["llm_key_used"].value_counts()
        if len(kc):
            print("\n  Key usage:")
            for k, v in kc.items():
                print(f"    {k:<20} {v:>6,} files")

    if n_scored == 0:
        print("\n  ✗ No scored rows.")
        print("═" * 80); return

    llm_cols = [f"llm_{d}" for d in DIMENSIONS]
    issues, passes = [], []

    if scored[llm_cols].isna().sum().sum() == 0:
        passes.append("No NaN values in scored rows")
    else:
        issues.append("NaN values found in LLM scores")

    range_ok = True
    for col in llm_cols:
        vals = scored[col].dropna()
        if (vals < 0.0).any() or (vals > 10.0).any():
            issues.append(f"{col}: out-of-range values"); range_ok = False
    if range_ok:
        passes.append("All scores in valid range [0.0, 10.0]")

    comp = scored["llm_quality_score_12d"].dropna()
    if comp.mean() > 9.2:   issues.append(f"Composite mean={comp.mean():.2f} — HIGH")
    elif comp.mean() < 2.0: issues.append(f"Composite mean={comp.mean():.2f} — LOW")
    else:                   passes.append(f"Composite mean={comp.mean():.2f} — healthy range")

    if comp.std() < 0.3:    issues.append(f"Composite std={comp.std():.3f} — low variance")
    else:                   passes.append(f"Composite std={comp.std():.3f} — good variance")

    print(f"\n  {'Dimension':<22} {'mean':>6} {'std':>6} {'min':>5} {'max':>5}  verdict")
    print("  " + "─" * 62)
    for d in DIMENSIONS:
        col  = scored[f"llm_{d}"].dropna()
        mean = col.mean(); std = col.std()
        v    = ("⚠ HIGH" if mean > 9.2 else
                "⚠ LOW"  if mean < 1.5 else "✓ ok")
        print(f"  {d:<22} {mean:>6.2f} {std:>6.2f} "
              f"{col.min():>5.1f} {col.max():>5.1f}  {v}")

    # Consistency report
    if "val_consistency_score" in scored.columns:
        active_checks = [
            ("val_error_handling_consistent",  "err"),
            ("val_security_consistent",        "sec"),
            ("val_anti_patterns_consistent",   "anti"),
            ("val_maintainability_consistent", "maint"),
        ]
        print(f"\n  CONSISTENCY VALIDATION (no scores modified — audit only):")
        print(f"  {'File':<45} {'cons':>5}  {'HiConf':>7}  {'framework':>10}  issues")
        print("  " + "─" * 88)
        for _, r in scored.sort_values(
                "llm_quality_score_12d", ascending=False).head(20).iterrows():
            fname = str(r["file_path"])[-43:]
            sc    = r.get("val_consistency_score", "?")
            hc    = "✓" if r.get("val_high_confidence") else "✗"
            fw    = str(r.get("val_detected_framework", "?"))[:10]
            bad   = [label for col, label in active_checks
                     if not r.get(col, True)]
            incon = ", ".join(bad) if bad else "—"
            print(f"  {fname:<45} {sc:>5}/4  {hc:>7}  {fw:>10}  {incon}")

        hi_conf = int(scored["val_high_confidence"].sum())
        mean_cs = scored["val_consistency_score"].mean()
        print(f"\n  High-confidence files  : {hi_conf:,}/{n_scored:,} "
              f"({hi_conf/n_scored*100:.1f}%)")
        print(f"  Mean consistency score : {mean_cs:.2f}/4")

    # Defect correlation (only if is_defective column present)
    if "is_defective" in scored.columns:
        defective = scored[scored["is_defective"] == 1]
        clean     = scored[scored["is_defective"] == 0]
        if len(defective) > 0 and len(clean) > 0:
            def_mean  = defective["llm_quality_score_12d"].mean()
            cln_mean  = clean["llm_quality_score_12d"].mean()
            print(f"\n  Defect signal check:")
            print(f"    Defective files mean composite : {def_mean:.3f}")
            print(f"    Clean files mean composite     : {cln_mean:.3f}")
            direction = "✓ CORRECT (defective < clean)" if def_mean < cln_mean else "⚠ CHECK"
            print(f"    Direction                      : {direction}")

    print(f"\n  CHECKS PASSED ({len(passes)}):")
    for p in passes: print(f"    ✓  {p}")
    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for i in issues: print(f"    ⚠  {i}")
    else:
        print(f"\n  Overall verdict: ✓ SCORES LOOK VALID AND HEALTHY")
    print("═" * 80)

# ══════════════════════════════════════════════════════════════════════════════
#  FAILURE RE-RUN  (logic unchanged from v1, signature updated)
# ══════════════════════════════════════════════════════════════════════════════

def rerun_failures(df: pd.DataFrame, results: list, total: int,
                   checkpoint: Path, progress_file: Path,
                   raw_log: Path, audit_report: Path,
                   repo_num: int, total_repos: int) -> list:
    timeout_indices = [
        i for i, r in enumerate(results)
        if r is not None and r.scores is None
        and r.error_types and "force_timeout" in r.error_types
    ]
    retry_indices = [
        i for i, r in enumerate(results)
        if (r is None or r.scores is None)
        and not (r is not None and r.error_types
                 and "force_timeout" in r.error_types)
    ]

    if timeout_indices:
        print(f"\n  ⏩ Skipping {len(timeout_indices)} timed-out file(s).")
    if not retry_indices:
        print("\n  ♻  No retryable failures.")
        return results

    print(f"\n  ♻  FAILURE RE-RUN: {len(retry_indices)} retryable files…")
    pbar = tqdm(total=len(retry_indices),
                desc=f"  [{repo_num}/{total_repos}] Re-scoring", unit="file",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    save_counter = 0
    for idx in retry_indices:
        row  = df.iloc[idx]
        path = str(row["file_path"])
        code = str(row["source_code"])
        mcc  = float(row.get("cyclomatic_complexity", 0))
        mxcc = float(row.get("max_cc", 0))
        try:
            results[idx] = score_single_file_with_timeout(
                idx, path, code, mcc, mxcc, raw_log, audit_report)
        except Exception:
            reason = f"Re-run exception: {traceback.format_exc()}"
            _log_audit(path, reason, [], [], audit_report)
            results[idx] = ScoringResult(
                idx=idx, file_path=path, scores=None,
                confidence=0, failure_reason=reason)
        pbar.update(1)
        save_counter += 1
        if save_counter % SAVE_EVERY == 0:
            save_checkpoint(df, results, total, checkpoint, progress_file)
    pbar.close()

    recovered    = sum(1 for i in retry_indices
                       if results[i] and results[i].scores)
    still_failed = len(retry_indices) - recovered + len(timeout_indices)
    print(f"\n  ♻  Re-run: {recovered} recovered | {still_failed} still failed")
    return results

# ══════════════════════════════════════════════════════════════════════════════
#  RUN ONE REPOSITORY
# ══════════════════════════════════════════════════════════════════════════════

def run_repo(repo_cfg: dict, repo_num: int, total_repos: int,
             interrupt_flag: list):
    parquet_path = Path(repo_cfg["parquet"])
    git_repo     = repo_cfg["git_repo"]

    if not parquet_path.exists():
        print(f"\n  [SKIP] Parquet not found: {parquet_path}")
        return

    stem           = parquet_path.stem
    output_parquet = OUTPUT_DIR / f"{stem}_12D_LLM_GROQ_v4.parquet"
    output_csv     = OUTPUT_DIR / f"{stem}_12D_LLM_GROQ_v4.csv"
    audit_report   = OUTPUT_DIR / f"{stem}_AUDIT_failures_v4.jsonl"
    checkpoint     = OUTPUT_DIR / f"{stem}_CHECKPOINT_v4.pkl"
    progress_file  = OUTPUT_DIR / f"{stem}_progress_v4.json"
    source_cache   = OUTPUT_DIR / f"{stem}_source_code_cache.parquet"
    raw_log        = OUTPUT_DIR / f"{stem}_raw_responses_v4.jsonl"

    print(f"\n{'═'*80}")
    print(f"  REPOSITORY [{repo_num}/{total_repos}] : {stem}")
    print(f"  Input parquet   : {parquet_path}")
    print(f"  Output parquet  : {output_parquet}")
    print(f"  Git repo        : {git_repo}")
    print(f"  Source cache    : {source_cache}")
    print(f"{'═'*80}")

    # Load parquet
    df = pd.read_parquet(parquet_path)
    print(f"  ↳ {len(df):,} rows | {len(df.columns)} columns")

    # ── Pre-filter: remove only zero-function files ──────────────────────────
    # Zero-function files (empty __init__.py, locale constants, config files)
    # have no executable logic for the LLM to assess. Scoring them gives
    # meaningless results and inflates the composite mean.
    # Test files are KEPT — they are legitimate Python code with real logic
    # and are included in ML training as a feature group.
    n_before = len(df)
    df = df[df["num_functions"] > 0].reset_index(drop=True)
    n_after  = len(df)
    print(f"  ↳ Pre-filter applied: {n_before:,} → {n_after:,} rows")
    print(f"     Removed {n_before-n_after:,} zero-function rows (empty __init__.py, locale, config)")
    print(f"     Test files KEPT — scored alongside production files")

    # Attach source code (PyDriller + cache)
    df = prepare_source_code(df, git_repo, source_cache)
    total = len(df)
    if total == 0:
        print(f"  [SKIP] No source code loaded.")
        return

    # Validate required miner columns
    required = ["loc", "cyclomatic_complexity", "num_functions", "max_cc"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        print(f"  [FATAL] Miner columns missing: {missing}")
        return

    # Resume from checkpoint
    results: list[Optional[ScoringResult]] = [None] * total
    if checkpoint.exists():
        try:
            state  = pickle.load(open(checkpoint, "rb"))
            loaded = state.get("results", [])
            if len(loaded) == total:
                results = loaded
                done = sum(1 for r in results if r is not None)
                print(f"  ↳ Resumed: {done:,}/{total:,} already scored.")
            else:
                print(f"  ↳ Checkpoint size mismatch ({len(loaded)} vs {total}) — fresh start.")
                print(f"     Previously scored files are in the LLM cache and will be")
                print(f"     recovered instantly without API calls.")
                # Delete stale checkpoint to avoid confusion
                checkpoint.unlink(missing_ok=True)
        except Exception as exc:
            print(f"  ⚠ Checkpoint load failed ({exc}) — fresh start.")

    work = [
        (i,
         str(df.iloc[i]["file_path"]),
         str(df.iloc[i]["source_code"]),
         float(df.iloc[i].get("cyclomatic_complexity", 0)),
         float(df.iloc[i].get("max_cc", 0)))
        for i in range(total) if results[i] is None
    ]
    remaining = len(work)

    if remaining == 0:
        print("  ✓ All files already scored — writing outputs.")
    else:
        est_hours = remaining / REQUESTS_PER_MINUTE / 60
        print(f"  ↳ Files to score   : {remaining:,}")
        print(f"  ↳ Active keys      : {len(GROQ_API_KEYS)}")
        print(f"  ↳ Daily capacity   : ~{len(GROQ_API_KEYS)*14400:,} req/day")
        print(f"  ↳ Est. time        : ~{est_hours:.1f} hours at {REQUESTS_PER_MINUTE} req/min")

    save_counter = 0
    pbar = tqdm(
        total=remaining,
        desc=f"  [{repo_num}/{total_repos}] Scoring",
        unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for idx, path, code, mcc, mxcc in work:
        if interrupt_flag[0]:
            save_checkpoint(df, results, total, checkpoint, progress_file)
            pbar.close(); return

        try:
            _result = score_single_file_with_timeout(
                idx, path, code, mcc, mxcc, raw_log, audit_report)
            if _result is not None:
                # Scored successfully (or failed after all retries but not
                # due to quota exhaustion) — save to checkpoint
                results[idx] = _result
            # If _result is None: all API slots were exhausted mid-file.
            # results[idx] stays None → file will be retried on resume.
        except Exception:
            fp     = str(path)
            reason = f"Unhandled exception: {traceback.format_exc()}"
            _log_audit(fp, reason, [], [], audit_report)
            results[idx] = ScoringResult(
                idx=idx, file_path=fp, scores=None,
                confidence=0, failure_reason=reason)

        pbar.update(1)
        save_counter += 1
        if save_counter % SAVE_EVERY == 0:
            save_checkpoint(df, results, total, checkpoint, progress_file)

    pbar.close()
    time.sleep(2)

    # Failure re-run
    n_failed    = sum(1 for r in results if r is None or r.scores is None)
    n_timed_out = sum(
        1 for r in results
        if r is not None and r.scores is None
        and r.error_types and "force_timeout" in r.error_types
    )
    if n_failed - n_timed_out > 0 and not interrupt_flag[0]:
        results = rerun_failures(
            df, results, total, checkpoint, progress_file,
            raw_log, audit_report, repo_num, total_repos)
    elif n_timed_out > 0:
        print(f"\n  ⏩ {n_timed_out} timed-out file(s) — skipping re-run.")

    save_checkpoint(df, results, total, checkpoint, progress_file)

    # Write final outputs
    print(f"\n  Writing final outputs…")
    write_done: list[Optional[pd.DataFrame]] = [None]

    def _write():
        try:
            out_local = build_output(df, results)
            out_local.to_parquet(output_parquet, index=False, compression="snappy")
            out_local.to_csv(output_csv, index=False, encoding="utf-8-sig")
            write_done[0] = out_local
        except Exception as e:
            print(f"  ⚠ Output write error: {e}")

    wt = threading.Thread(target=_write, daemon=True)
    wt.start(); wt.join(timeout=120)

    if write_done[0] is None:
        print("  ⚠ Output write failed — results in checkpoint.")
        return

    out: pd.DataFrame = write_done[0]  # type: ignore[assignment]
    succeeded = int(out["llm_quality_score_12d"].notna().sum())
    overall   = out["llm_quality_score_12d"].dropna()

    print(f"  ↳ Parquet : {out.shape[0]:,} rows × {out.shape[1]} cols → {output_parquet.name}")
    print(f"  ↳ CSV     : {out.shape[0]:,} rows × {out.shape[1]} cols → {output_csv.name}")
    print(f"  ↳ Scored  : {succeeded:,}/{total:,}  |  "
          f"mean={overall.mean():.3f}  std={overall.std():.3f}")

    validate_scores(out)
    save_llm_cache()

    _log_daily("repo_completed", {
        "repo": stem, "total": total,
        "succeeded": succeeded, "failed": total - succeeded,
        "mean": round(float(overall.mean()), 3) if len(overall) else None,
    })

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    n_keys = len(GROQ_API_KEYS)
    print("═" * 80)
    print("  LLM CODE QUALITY SCORER — REPOSITORY EDITION  v4.0")
    print(f"  Model                : {GROQ_MODEL}  (single model)")
    print(f"  Keys loaded          : {n_keys} / {n_keys}")
    for name, key in GROQ_API_KEYS:
        print(f"    {name:<20} {key[:8]}…{key[-4:]}")
    print(f"  Max code length      : {MAX_CODE_LEN:,} chars")
    print(f"  Rate limit           : {REQUESTS_PER_MINUTE} req/min")
    print(f"  Validation checks    : 4 active (err, security, anti, maintainability)")
    print(f"  LLM scores modified  : NEVER")
    print(f"  Dimensions           : {', '.join(DIMENSIONS)}")
    print(f"  Repositories queued  : {len(REPOSITORIES)}")
    for i, r in enumerate(REPOSITORIES, 1):
        print(f"    {i}. {Path(r['parquet']).stem}")
    print(f"  Started              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 80)

    load_llm_cache()

    print("\n[STARTUP] Testing all API keys…")
    if not test_api_connection():
        sys.exit("\n[FATAL] No working Groq API keys found.\n")

    interrupt_flag = [False]

    def _signal_handler(sig, frame):
        print("\n\n  ⚡ Ctrl+C — saving checkpoint after current file…")
        interrupt_flag[0] = True

    signal.signal(signal.SIGINT, _signal_handler)

    _log_daily("session_started", {
        "keys":  [n for n, _ in GROQ_API_KEYS],
        "repos": [Path(r["parquet"]).stem for r in REPOSITORIES],
        "model": GROQ_MODEL,
    })

    total_repos = len(REPOSITORIES)
    for repo_num, repo_cfg in enumerate(REPOSITORIES, 1):
        if interrupt_flag[0]:
            print("\n  ⚡ Interrupted — stopping.")
            break
        run_repo(repo_cfg, repo_num, total_repos, interrupt_flag)

    save_llm_cache()

    print(f"\n{'═' * 80}")
    print(f"  ALL REPOSITORIES PROCESSED")
    print(f"  Finished  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cache     : {len(LLM_CACHE_DATA):,} total cached responses")
    print(f"  Log       : {DAILY_LOG}")
    print(f"{'═' * 80}\n")

    _log_daily("session_finished", {"cache_size": len(LLM_CACHE_DATA)})


if __name__ == "__main__":
    main()