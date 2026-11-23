# ================================================================================
#  LLM-BASED CODE QUALITY SCORER — MANUAL FILES EDITION  v4.0
#  AMARE DERGU MENGSTIE — Master's Thesis, FH Upper Austria, 2026
# ================================================================================
#
#  WHAT CHANGED FROM v3:
#  ─────────────────────
#  ✓ REPLACED: framework_bp → maintainability
#              magic_values → security
#
#    framework_bp was bimodal (11/20 files scored exactly 10.0, only 6 unique
#    values). The LLM could not distinguish gradations of "generic Python idioms".
#    maintainability is genuinely continuous — the LLM reads future change cost
#    from naming, coupling, function length, and hidden dependencies.
#    Directly cited in thesis literature (Rahman & Devanbu, Bird et al.).
#
#    magic_values was bimodal (11 zeros, 5 tens, 6 unique values).
#    It overlapped conceptually with anti_patterns.
#    security is genuinely continuous — hardcoded secrets, MD5 passwords,
#    eval/exec usage, bare excepts hiding failures — each is a distinct risk.
#    Directly relevant to defect prediction: security bugs are defects.
#    Static validator signals already available (hardcoded_secret, md5, bare_except).
#
#  ✓ UPDATED: ValidationFeatures — replaced num_magic_numbers with:
#             has_md5_password (bool)    — validates llm_security
#             (hardcoded_secret already present — also validates llm_security)
#
#  ✓ UPDATED: ConsistencyResult — replaced val_magic_values_consistent with:
#             val_security_consistent  — fires when hardcoded secret or MD5
#             password present but LLM scored security >= 5.0
#             val_maintainability_consistent replaces val_framework_consistent
#             (removed in v3) — fires when CC > 40 or max_cc > 20 but LLM
#             scored maintainability >= 7.0
#
#  ✓ UPDATED: Prompts — all three variants updated with new dimension names,
#             definitions, and calibration anchors for maintainability/security
#
#  ✓ UPDATED: Output schema — llm_framework_bp → llm_maintainability,
#             llm_magic_values → llm_security, vf_num_magic_numbers →
#             vf_has_md5_password, val_magic_values_consistent →
#             val_security_consistent, added val_maintainability_consistent
#
#  UNCHANGED FROM v3:
#  ──────────────────
#  All other 10 dimensions, API key rotation, rate limiting, quota management,
#  checkpointing, retry logic, cache, logging, framework detection, lizard+regex
#  extraction, all other validation checks — identical to v3.
#
#  OUTPUT SCHEMA (35 columns — same count as v3, clean replacements):
#  ────────────────────────────────────────────────────────────────────
#  file_path                          — primary key
#  loc, cyclomatic_complexity,        — 5 static metrics (ML features)
#  num_functions, avg_cc_per_function, max_cc
#  llm_readability … llm_style        — 12 LLM dimension scores (ML features)
#    includes: llm_maintainability (replaces llm_framework_bp)
#              llm_security         (replaces llm_magic_values)
#  llm_quality_score_12d              — composite score
#  vf_num_docstrings                  — validates llm_documentation
#  vf_num_type_hints                  — validates llm_documentation
#  vf_num_try_except                  — validates llm_error_handling
#  vf_num_raise                       — validates llm_error_handling
#  vf_num_bare_except                 — validates llm_error_handling + llm_security
#  vf_num_magic_numbers               — kept: still useful for anti_patterns context
#  vf_num_global                      — validates llm_anti_patterns
#  vf_has_hardcoded_secret            — validates llm_anti_patterns + llm_security
#  vf_has_md5_password                — validates llm_security (NEW)
#  vf_liz_max_params                  — validates llm_testability
#  vf_liz_long_funcs                  — validates llm_maintainability + llm_structure
#  val_error_handling_consistent      — fires on avg_5_duplication
#  val_security_consistent            — replaces val_magic_values_consistent
#  val_anti_patterns_consistent       — fires on nightmare_16
#  val_maintainability_consistent     — NEW: fires when CC very high but maint. scored high
#  val_consistency_score              — 0–4 (4 = all active checks pass)
#  val_high_confidence                — True if val_consistency_score == 4
#  val_detected_framework             — fastapi/django/.../generic
#
#  HOW TO RUN:
#  ────────────
#  Step 1:  python manual_files_miner.py
#  Step 2:  $env:GROQ_API_KEY1="gsk_xxxx"
#           python manual_files_llm_scorer_v4.py
#
#  INPUT:   data/02_snapshots/manual_files_metrics.parquet
#  OUTPUT:  data/04_llm_temp/manual_files_metrics_12D_LLM_GROQ_v4.parquet
#           data/04_llm_temp/manual_files_metrics_12D_LLM_GROQ_v4.csv
#
#  INSTALL: pip install requests pandas pyarrow tqdm lizard
# ================================================================================
#  ─────────────────────
#  ✓ FIX 1 — Output columns reduced from 64 → 35 (clean, no redundancy):
#             Dropped: zero-variance cols, exact duplicates, high-corr redundancies,
#             always-False framework flags, operational-only metadata
#             Kept: identifiers, static metrics, 12 LLM scores, composite,
#                   8 vf_ validation features, 3 val_ consistency flags,
#                   val_consistency_score, val_high_confidence, val_detected_framework
#
#  ✓ FIX 2 — val_high_confidence threshold raised 10 → 11:
#             nightmare_16 (10/12) now correctly marked low-confidence
#             because it has 3 wrong LLM scores (anti_patterns, magic_values,
#             duplication) — the first two are caught, raising threshold fixes it
#
#  ✓ FIX 3 — framework_score_cap for GENERIC files removed:
#             Generic rubric explicitly asks LLM to score general Python idioms
#             so a generic file scoring 8.0 on framework_bp is CORRECT, not
#             inconsistent. The cap was causing 9 false-positive flags.
#             val_framework_consistent now only fires when a specific framework
#             IS detected but the LLM contradicts it.
#
#  ✓ FIX 4 — Removed redundant vf_ columns:
#             vf_liz_avg_func_length (corr=0.937 with avg_cc_per_function)
#             vf_liz_max_func_length (corr=0.864 with max_cc)
#             Both already covered by miner columns.
#
#  ✓ FIX 5 — Removed always-True val_ consistency columns:
#             val_documentation, val_complexity, val_structure, val_testability,
#             val_readability, val_naming, val_style, val_duplication
#             These never fired on any of the 16 files — zero information value.
#             The 3 that DO fire kept: val_error_handling, val_magic_values,
#             val_anti_patterns
#
#  ✓ FIX 6 — val_consistency_score now counts only the 3 active checks (0–3)
#             not 12 checks where 9 are always True. Score of 3 = fully consistent.
#
#  UNCHANGED FROM v2:
#  ──────────────────
#  All API key rotation, rate limiting, quota management, checkpointing,
#  retry logic, cache, logging, prompt variants, framework detection,
#  lizard+regex extraction — identical to v2.
#
#  OUTPUT SCHEMA (35 columns):
#  ────────────────────────────
#  file_path                      — primary key
#  loc, cyclomatic_complexity,    — 5 static metrics (ML features)
#  num_functions, avg_cc_per_function, max_cc
#  llm_readability … llm_style    — 12 LLM dimension scores (ML features)
#  llm_quality_score_12d          — composite score
#  vf_num_docstrings              — validates llm_documentation
#  vf_num_type_hints              — validates llm_documentation
#  vf_num_try_except              — validates llm_error_handling
#  vf_num_raise                   — validates llm_error_handling
#  vf_num_bare_except             — validates llm_error_handling
#  vf_num_magic_numbers           — validates llm_magic_values
#  vf_num_global                  — validates llm_anti_patterns
#  vf_has_hardcoded_secret        — validates llm_anti_patterns
#  vf_liz_max_params              — validates llm_testability
#  vf_liz_long_funcs              — validates llm_structure
#  val_error_handling_consistent  — fired on avg_5_duplication
#  val_magic_values_consistent    — fired on nightmare_16
#  val_anti_patterns_consistent   — fired on nightmare_16
#  val_consistency_score          — 0–3 (3 = all active checks pass)
#  val_high_confidence            — True if val_consistency_score == 3
#  val_detected_framework         — fastapi/django/numpy/pandas/flask/
#                                   pytest/sklearn/torch/generic
#
#  HOW TO RUN:
#  ────────────
#  Step 1:  python manual_files_miner.py
#  Step 2:  $env:GROQ_API_KEY1="gsk_xxxx"
#           python manual_files_llm_scorer_v3.py
#
#  INPUT:   data/02_snapshots/manual_files_metrics.parquet
#  OUTPUT:  data/04_llm_temp/manual_files_metrics_12D_LLM_GROQ_v3.parquet
#           data/04_llm_temp/manual_files_metrics_12D_LLM_GROQ_v3.csv
#
#  INSTALL: pip install requests pandas pyarrow tqdm lizard
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
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════

MANUAL_FILES_DIR = Path("data/01_raw_data/manual_files")
INPUT_PARQUET    = Path("data/02_snapshots/manual_files_metrics.parquet")

OUTPUT_DIR     = Path("data/04_llm_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STEM           = "manual_files_metrics"
OUTPUT_PARQUET = OUTPUT_DIR / f"{STEM}_12D_LLM_GROQ_v4.parquet"
OUTPUT_CSV     = OUTPUT_DIR / f"{STEM}_12D_LLM_GROQ_v4.csv"
AUDIT_REPORT   = OUTPUT_DIR / f"{STEM}_AUDIT_failures_v4.jsonl"
CHECKPOINT     = OUTPUT_DIR / f"{STEM}_CHECKPOINT_v4.pkl"
PROGRESS_FILE  = OUTPUT_DIR / f"{STEM}_progress_v4.json"
RAW_LOG        = OUTPUT_DIR / f"{STEM}_raw_responses_v4.jsonl"
LLM_CACHE      = OUTPUT_DIR / "_llm_response_cache_v4.json"
DAILY_LOG      = OUTPUT_DIR / "_daily_progress_log_v4.jsonl"

# ══════════════════════════════════════════════════════════════════════════════
#  API KEYS
# ══════════════════════════════════════════════════════════════════════════════

GROQ_API_KEYS = []
for _name in ["GROQ_API_KEY1","GROQ_API_KEY2","GROQ_API_KEY3",
              "GROQ_API_KEY4","GROQ_API_KEY5"]:
    _val = os.environ.get(_name)
    if _val and _val.strip():
        GROQ_API_KEYS.append((_name, _val.strip()))

if not GROQ_API_KEYS:
    sys.exit(
        "\n[FATAL] No Groq API keys found.\n"
        "  PowerShell: $env:GROQ_API_KEY1='gsk_xxxx'\n"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  GROQ CONFIG
# ══════════════════════════════════════════════════════════════════════════════

GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
#GROQ_MODEL = "llama-3.1-8b-instant"
GROQ_MODEL = "openai/gpt-oss-20b"

MAX_CODE_LEN                 = 14_000
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
    "err_cap"             : 3.0,   # no try/except AND no raise → score must be ≤ this
    "bare_except_min"     : 3,     # ≥ N bare excepts → err should not be ≥ 6.0
    "bare_except_score"   : 6.0,
    # security — replaces magic_values thresholds
    "secret_sec_score"    : 5.0,   # hardcoded secret → security must be < this
    "md5_sec_score"       : 6.0,   # MD5/SHA1 for passwords → security must be < this
    "bare_sec_min"        : 5,     # ≥ N bare excepts = hiding failures → security < 5.0
    "bare_sec_score"      : 5.0,
    # anti_patterns
    "global_threshold"    : 3,     # ≥ N global statements → anti_patterns suspicious
    "global_score"        : 6.0,
    "secret_score"        : 5.0,   # hardcoded secret → anti_patterns must be < this
    # maintainability — replaces framework_bp thresholds
    "very_high_cc"        : 40,    # total CC > 40 → maintainability must be < 5.0
    "high_max_cc"         : 20,    # max_cc > 20 → maintainability must be < 7.0
    "maint_score_high"    : 7.0,
    "maint_score_very"    : 5.0,
    # framework detection (unchanged)
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
        "FastAPI: score correct use of APIRouter, Depends for dependency injection, "
        "response_model on endpoints, HTTPException for errors, Pydantic BaseModel "
        "schemas, proper HTTP status codes. High = all used correctly."
    ),
    "django"  : (
        "Django: score correct ORM usage (QuerySet, select_related, prefetch_related), "
        "class-based views preferred, get_object_or_404, login_required, "
        "proper form validation. High = idiomatic Django patterns."
    ),
    "numpy"   : (
        "NumPy: score vectorised operations over Python loops, correct dtype, "
        "broadcasting, np.testing for assertions, no unnecessary copies. "
        "High = fully vectorised, idiomatic NumPy."
    ),
    "pandas"  : (
        "Pandas: score method chaining, vectorised ops over .apply(), explicit .copy(), "
        "correct NA handling, avoid iterrows(). High = idiomatic pandas."
    ),
    "flask"   : (
        "Flask: score blueprints, url_for(), render_template(), proper error handlers, "
        "application factory pattern. High = idiomatic Flask structure."
    ),
    "pytest"  : (
        "pytest: score fixture usage, parametrize, pytest.raises(), meaningful names, "
        "conftest.py, no unittest mixing. High = fully idiomatic pytest."
    ),
    "sklearn" : (
        "scikit-learn: score Pipeline usage, fit/transform/predict separation, "
        "custom estimators via BaseEstimator, no data leakage. High = production sklearn."
    ),
    "torch"   : (
        "PyTorch: score nn.Module subclassing, forward() design, device handling, "
        "DataLoader, zero_grad/backward/step. High = idiomatic PyTorch."
    ),
    "generic" : (
        "Generic Python (no specific framework detected): score how easy this code "
        "would be to CHANGE SAFELY in the future. Consider: are functions small and "
        "focused (single responsibility)? Are dependencies explicit rather than hidden? "
        "Can you modify one part without breaking unrelated parts? Is the public "
        "interface clear? "
        "CRITICAL: A file with god classes, global mutable state, 8+ parameter "
        "functions, or deeply nested logic MUST score LOW (0-3) — these make future "
        "changes risky regardless of how the Python syntax looks. "
        "SCALE: 8-10=safe to extend, 5-7=careful refactoring needed, "
        "2-4=high risk of regression, 0-1=unmaintainable."
    ),
}


def detect_framework(source: str) -> tuple:
    """
    Detect the primary framework. Returns (name, marker_count).
    Falls back to 'generic' if no framework reaches VAL['framework_min_markers'].
    """
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
            s["exhausted"] = False
            s["exhausted_until"] = 0.0
            print(f"\n  ✅ Key {key_name} quota reset.")
            return True
        return False

def _get_available_keys() -> list:
    return [(n, k) for n, k in GROQ_API_KEYS if _is_key_available(n)]

def _all_keys_exhausted() -> bool:
    return len(_get_available_keys()) == 0

# ── Quota-sleep flag ──────────────────────────────────────────────────────────
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
#  KEY SELECTOR
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

def handle_all_keys_exhausted():
    wait    = seconds_until_groq_reset()
    wake_at = datetime.now() + timedelta(seconds=wait)
    print(f"\n  🚫 ALL {len(GROQ_API_KEYS)} KEYS EXHAUSTED.")
    print(f"  ⏰ Sleeping {wait//3600}h {(wait%3600)//60}m — resuming at "
          f"{wake_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  💡 Safe to leave running — will auto-resume.\n")
    _log_daily("all_keys_exhausted", {
        "wait_sec": wait, "resume_at": wake_at.isoformat()})
    _set_quota_sleeping(True)
    time.sleep(wait)
    _set_quota_sleeping(False)
    with _key_state_lock:
        for name, _ in GROQ_API_KEYS:
            _key_states[name]["exhausted"]       = False
            _key_states[name]["exhausted_until"] = 0.0
    print(f"\n  ✅ Keys reset — resuming at "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log_daily("all_keys_reset", {})

def _log_daily(event: str, data: dict):
    entry = {"event": event, "timestamp": datetime.now().isoformat(), **data}
    try:
        with open(DAILY_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION FEATURES  (regex + lizard — never fails on any Python file)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ValidationFeatures:
    """
    Features extracted from source code using regex and lizard.
    Used ONLY for validation — never as ML features themselves.
    Each field validates one or more LLM dimensions.
    """
    # validates llm_documentation
    num_docstrings:       int   = 0
    num_type_hints:       int   = 0
    # validates llm_error_handling
    num_try_except:       int   = 0
    num_raise:            int   = 0
    num_bare_except:      int   = 0
    # kept for anti_patterns context (was magic_values validator in v3)
    num_magic_numbers:    int   = 0
    # validates llm_anti_patterns
    num_global:           int   = 0
    has_hardcoded_secret: bool  = False
    # validates llm_security (NEW in v4)
    has_md5_password:     bool  = False
    # validates llm_testability
    liz_max_params:       int   = 0
    # validates llm_structure + llm_maintainability
    liz_long_funcs:       int   = 0
    # framework detection
    detected_framework:   str   = "generic"
    framework_markers:    int   = 0


def _strip_docs_and_comments(source: str) -> str:
    """Strip triple-quoted strings and inline comments before counting magic numbers."""
    src = re.sub(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'', '""', source)
    src = re.sub(r'#[^\n]*', '', src)
    return src


def extract_validation_features(source: str) -> ValidationFeatures:
    """
    Extract all validation features. Never raises SyntaxError.
    Regex covers presence/absence checks; lizard covers structural metrics
    that the miner did not save.
    """
    f     = ValidationFeatures()
    clean = _strip_docs_and_comments(source)

    # Framework detection
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

    # error handling (try/except AND raise both count as error handling)
    f.num_try_except  = len(re.findall(r'\bexcept\b', source))
    f.num_raise       = len(re.findall(r'\braise\b', source))
    f.num_bare_except = len(re.findall(r'except\s*:', source))

    # magic numbers (clean source — excludes counts inside docstrings/comments)
    f.num_magic_numbers = len(set(re.findall(
        r'(?<![.\w])\b(?!(?:0|1|2|10|100)\b)\d+(?:\.\d+)?\b',
        clean
    )))

    # anti-patterns
    f.num_global = len(re.findall(r'\bglobal\b', source))
    f.has_hardcoded_secret = bool(re.search(
        r'(?i)(password|secret|token|api_key|apikey)\s*=\s*["\'][^"\']{4,}["\']',
        source
    ))

    # lizard: max_params and long_funcs — not saved by miner, no redundancy
    try:
        ana   = lizard.analyze_file.analyze_source_code("tmp.py", source)
        funcs = ana.function_list
        if funcs:
            f.liz_max_params  = max(len(fn.parameters) for fn in funcs)
            f.liz_long_funcs  = sum(1 for fn in funcs if fn.length > 30)
    except Exception:
        pass  # lizard fails gracefully — features stay at default 0

    return f


def validation_features_summary(f: ValidationFeatures) -> str:
    """Summary string passed into the LLM prompt as context."""
    return (
        f"docstrings={f.num_docstrings} | type_hints={f.num_type_hints} | "
        f"try_except={f.num_try_except} | raise={f.num_raise} | "
        f"bare_except={f.num_bare_except} | "
        f"magic_numbers={f.num_magic_numbers}(excl.docstrings/comments) | "
        f"global_stmts={f.num_global} | "
        f"hardcoded_secret={f.has_hardcoded_secret} | "
        f"md5_password={f.has_md5_password} | "
        f"max_params={f.liz_max_params} | long_funcs(>30lines)={f.liz_long_funcs} | "
        f"framework={f.detected_framework}({f.framework_markers} markers)"
    )

# ══════════════════════════════════════════════════════════════════════════════
#  CONSISTENCY VALIDATION  (no LLM score is ever modified)
#
#  Only 3 checks are active because only 3 dimensions showed actual variation
#  across the 16 manual files. The other checks (documentation, complexity,
#  structure, testability, readability) never fired — all consistent already.
#  Naming, style, duplication cannot be validated by static means.
#
#  val_consistency_score: 0–3  (3 = fully consistent, trustworthy)
#  val_high_confidence:   True if score == 3  (threshold raised from 10/12)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConsistencyResult:
    """
    Four active validation checks + summary score.
    val_consistency_score: 0–4  (4 = all checks pass = high confidence)

    Active checks:
      error_handling    — no err handling present but LLM scored it high
      security          — hardcoded secret or MD5 password but security scored high
      anti_patterns     — many globals or hardcoded secret but anti_patterns scored high
      maintainability   — very high CC but maintainability scored high
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
    Returns consistency flags only — NO score is ever modified.

    Four active checks (v4 — replaces magic_values check with
    security + maintainability checks):

    1. error_handling  — no error handling present but LLM scored high
    2. security        — verifiable security flaw present but LLM scored high
    3. anti_patterns   — many globals or hardcoded secret but LLM scored high
    4. maintainability — very high CC / many long functions but LLM scored high

    Inconsistency = LLM gave HIGH score where static evidence says it CANNOT.
    """
    c = ConsistencyResult(val_detected_framework=feat.detected_framework)

    has_error_handling = feat.num_try_except > 0 or feat.num_raise > 0

    # ── 1. error_handling ─────────────────────────────────────────────────────
    # raise counts as valid — explicitly raising is correct error handling
    # bare except ≥ 3 = silently swallowing errors = should not score ≥ 6.0
    c.val_error_handling_consistent = not (
        (not has_error_handling and
         scores["error_handling"] > VAL["err_cap"]) or
        (feat.num_bare_except >= VAL["bare_except_min"] and
         scores["error_handling"] >= VAL["bare_except_score"])
    )

    # ── 2. security ───────────────────────────────────────────────────────────
    # hardcoded secret (password/token/key) → security must be < 5.0
    # MD5/SHA1 used for password hashing → real vulnerability → security < 6.0
    # Many bare excepts hide security failures → security should not be high
    c.val_security_consistent = not (
        (feat.has_hardcoded_secret and
         scores["security"] >= VAL["secret_sec_score"]) or
        (feat.has_md5_password and
         scores["security"] >= VAL["md5_sec_score"]) or
        (feat.num_bare_except >= VAL["bare_sec_min"] and
         scores["security"] >= VAL["bare_sec_score"])
    )

    # ── 3. anti_patterns ──────────────────────────────────────────────────────
    # global statements ≥ 3 OR hardcoded secret → anti_patterns should be low
    c.val_anti_patterns_consistent = not (
        (feat.num_global >= VAL["global_threshold"] and
         scores["anti_patterns"] >= VAL["global_score"]) or
        (feat.has_hardcoded_secret and
         scores["anti_patterns"] >= VAL["secret_score"])
    )

    # ── 4. maintainability ────────────────────────────────────────────────────
    # Very high total CC > 40 → code is objectively hard to change → maint < 5.0
    # High max_cc > 20 (one very complex function) → maint should not be ≥ 7.0
    # Many long functions (>30 lines) → maint should not be very high
    c.val_maintainability_consistent = not (
        (miner_cc > VAL["very_high_cc"] and
         scores["maintainability"] >= VAL["maint_score_very"]) or
        (miner_max_cc > VAL["high_max_cc"] and
         scores["maintainability"] >= VAL["maint_score_high"]) or
        (feat.liz_long_funcs >= 3 and
         scores["maintainability"] >= VAL["maint_score_high"])
    )

    # ── summary ───────────────────────────────────────────────────────────────
    c.val_consistency_score = sum([
        c.val_error_handling_consistent,
        c.val_security_consistent,
        c.val_anti_patterns_consistent,
        c.val_maintainability_consistent,
    ])
    # Threshold = 4 (ALL four checks pass)
    # Any active inconsistency → low-confidence
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
  → Most real-world files score 4–7. Only exceptional code scores 8+.

PRE-COMPUTED STATIC METRICS (use these to calibrate your scores):
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
                     that do too many things, no clear contracts between components.
                     Score HIGH for: isolated changes, clear interfaces, low coupling.
                     SCALE: 8-10=easy to extend safely, 5-7=moderate effort needed,
                     2-4=risky to change without breaking things, 0-1=unmaintainable.
                     Use the pre-computed CC and long_funcs metrics above to calibrate.
  anti_patterns    — ABSENCE of: god classes, global mutable state, hardcoded
                     secrets, circular logic. HIGH score = code is CLEAN.
  security         — absence of security vulnerabilities. Score LOW for: hardcoded
                     passwords/secrets/tokens, MD5/SHA1 for password hashing,
                     eval()/exec() on input, subprocess shell=True, bare excepts
                     hiding security failures. Score HIGH for: no such issues.
                     SCALE: 8-10=no issues, 5-7=one minor concern, 2-4=two concerns,
                     0-1=multiple critical vulnerabilities.
  style            — PEP 8 compliance, spacing, import ordering

CRITICAL SCORING RULES:
  anti_patterns: 10.0=ZERO problems (GOOD), 0.0=FULL of problems (BAD)
  security:      10.0=ZERO vulnerabilities (GOOD), 0.0=many vulnerabilities (BAD)
  maintainability: if pre-computed CC > 40, score MUST be < 5.0
                   if max_cc > 20, score MUST be < 7.0
  If metrics show hardcoded secret or MD5 password hashing, security MUST be < 5.0.

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
Framework: {framework}
Framework rubric for maintainability: {framework_rubric}

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
anti_patterns=10 means ZERO anti-patterns (good). 0=full of problems (bad).
security=10 means ZERO vulnerabilities (good). 0=many vulnerabilities (bad).
maintainability=score how easy to change safely. High CC means LOW score.
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
    payload = f"groq::v4::{variant}::{GROQ_MODEL}::{code[:MAX_CODE_LEN]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

# ══════════════════════════════════════════════════════════════════════════════
#  API CONNECTIVITY TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_api_connection() -> bool:
    print(f"  ↳ Testing {len(GROQ_API_KEYS)} key(s) with {GROQ_MODEL}")
    any_ok = False
    for key_name, api_key in GROQ_API_KEYS:
        print(f"    {key_name}…", end=" ", flush=True)
        try:
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": "Reply with exactly: OK"},
                        {"role": "user",   "content": "ping"},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 5,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                reply = resp.json()["choices"][0]["message"]["content"].strip()
                print(f"✓  ('{reply}')")
                any_ok = True
            elif resp.status_code == 429:
                print(f"⚠  429 (key works, rate limited)")
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
            time.sleep(5)
            continue

        try:
            rate_limited_sleep()
            resp = requests.post(
                GROQ_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type":  "application/json",
                },
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
                    advance_key()
                    continue
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
                    advance_key()
                    time.sleep(retry_after)
                continue

            elif resp.status_code in (500, 502, 503, 504):
                time.sleep(10)
                continue
            else:
                time.sleep(5)
                continue

        except requests.Timeout:
            print(f"\n  ⏱ Timeout on {key_name} — retrying…")
            advance_key()
            time.sleep(5)
            continue
        except Exception:
            time.sleep(5)
            continue

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
             parsed_ok: bool, error_type: str = ""):
    entry = {
        "file": path, "variant": variant, "model": GROQ_MODEL,
        "parsed_ok": parsed_ok, "error_type": error_type,
        "response": raw[:500], "timestamp": datetime.now().isoformat(),
    }
    with _log_lock:
        with open(RAW_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

def _log_audit(file_path: str, reason: str,
               error_types: list, raws: list):
    entry = {
        "file_path":      file_path,
        "failure_reason": reason,
        "error_types":    sorted(set(e for e in error_types if e)),
        "raw_sample":     [r[:300] for r in raws],
        "timestamp":      datetime.now().isoformat(),
    }
    with _audit_lock:
        AUDIT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_REPORT, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE FILE SCORER
# ══════════════════════════════════════════════════════════════════════════════

def score_single_file(
    idx:          int,
    path:         str,
    code:         str,
    miner_cc:     float = 0.0,
    miner_max_cc: float = 0.0,
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
            path=path,
            code=truncated,
            features_summary=feat_summary,
            framework=framework,
            framework_rubric=rubric,
        )

        raw, err_type, _, key_used = call_groq_raw(prompt, truncated, variant)
        raw_history.append(raw)
        error_types.append(err_type)
        _log_raw(path, variant, raw, parsed_ok=False, error_type=err_type)

        scores = parse_12d(raw)
        if scores:
            _log_raw(path, variant, raw, parsed_ok=True)
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
            print(f"\n  ⚠ All slots failed: {str(path)[:70]}")
            break

        time.sleep(2.0)

    reason = (f"All {MAX_PARSE_RETRIES} attempts failed. "
              f"Errors: {sorted(set(error_types))}")
    _log_audit(path, reason, error_types, raw_history)

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
    miner_cc:     float = 0.0,
    miner_max_cc: float = 0.0,
) -> ScoringResult:
    result_box: list = [None]
    exc_box:    list = [None]

    def _worker():
        try:
            result_box[0] = score_single_file(
                idx, path, code,
                miner_cc=miner_cc,
                miner_max_cc=miner_max_cc,
            )
        except Exception:
            exc_box[0] = traceback.format_exc()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    # Pause-aware deadline — extends during quota-reset sleep
    deadline = time.time() + FILE_TIMEOUT_SECONDS
    while t.is_alive():
        if _is_quota_sleeping():
            deadline += 1.0
        if time.time() > deadline:
            break
        t.join(timeout=1.0)

    if t.is_alive():
        reason = (f"FORCE TIMEOUT after {FILE_TIMEOUT_SECONDS}s — "
                  f"will retry in re-run pass.")
        print(f"\n  ⏱ FORCE TIMEOUT: {str(path)[:70]}")
        _log_audit(path, reason, ["force_timeout"], [])
        return ScoringResult(
            idx=idx, file_path=path, scores=None, confidence=0,
            failure_reason=reason, error_types=["force_timeout"],
        )

    if exc_box[0]:
        reason = f"Unhandled exception: {exc_box[0]}"
        _log_audit(path, reason, ["exception"], [])
        return ScoringResult(
            idx=idx, file_path=path, scores=None, confidence=0,
            failure_reason=reason, error_types=["exception"],
        )

    return result_box[0]

# ══════════════════════════════════════════════════════════════════════════════
#  SOURCE CODE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_source_code_from_disk(df: pd.DataFrame) -> pd.DataFrame:
    print(f"  ↳ Loading source code from: {MANUAL_FILES_DIR}")
    if not MANUAL_FILES_DIR.exists():
        print(f"  ✗ Directory not found: {MANUAL_FILES_DIR}")
        df["source_code"] = None
        return df

    def _read(stored_path: str) -> Optional[str]:
        fname = Path(stored_path).name
        full  = MANUAL_FILES_DIR / fname
        if not full.exists():
            print(f"  ⚠ Not found: {full}")
            return None
        try:
            return full.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            print(f"  ⚠ Could not read {full}: {exc}")
            return None

    df = df.copy()
    df["source_code"] = df["file_path"].apply(_read)
    found   = df["source_code"].notna().sum()
    missing = df["source_code"].isna().sum()
    print(f"  ↳ Loaded {found}/{len(df)} files  ({missing} missing)")
    return df.dropna(subset=["source_code"]).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
#  BUILD OUTPUT  (35 clean columns — see header for full schema)
# ══════════════════════════════════════════════════════════════════════════════

# Columns to keep from the miner parquet (always present)
MINER_KEEP = [
    "file_path",
    "loc", "cyclomatic_complexity", "num_functions",
    "avg_cc_per_function", "max_cc",
]

# vf_ columns to save (11 features with unique validation value)
VF_KEEP = [
    "num_docstrings", "num_type_hints",
    "num_try_except", "num_raise", "num_bare_except",
    "num_magic_numbers",
    "num_global", "has_hardcoded_secret",
    "has_md5_password",          # NEW in v4: validates llm_security
    "liz_max_params", "liz_long_funcs",
]

# val_ columns to save (4 active checks + summary + filter flag + framework)
VAL_KEEP = [
    "val_error_handling_consistent",
    "val_security_consistent",        # replaces val_magic_values_consistent
    "val_anti_patterns_consistent",
    "val_maintainability_consistent", # NEW in v4
    "val_consistency_score",
    "val_high_confidence",
    "val_detected_framework",
]


def build_output(df: pd.DataFrame, results: list) -> pd.DataFrame:
    # Start from miner columns only — drop everything else
    keep_from_miner = [c for c in MINER_KEEP if c in df.columns]
    out = df[keep_from_miner].copy()

    # LLM scores (12 dimensions)
    for d in DIMENSIONS:
        out[f"llm_{d}"] = [
            (r.scores[d] if r and r.scores else None) for r in results]

    # Composite score
    out["llm_quality_score_12d"] = (
        out[[f"llm_{d}" for d in DIMENSIONS]]
        .mean(axis=1, skipna=True).round(3))

    # vf_ validation features (10 selected columns)
    blank_feat = asdict(ValidationFeatures())
    for k in VF_KEEP:
        out[f"vf_{k}"] = [
            (r.val_features.get(k) if r and r.val_features else None)
            for r in results]

    # val_ consistency (3 active + summary + filter + framework)
    # ConsistencyResult fields are stored WITH the val_ prefix —
    # look up col directly, do NOT strip "val_"
    for col in VAL_KEEP:
        out[col] = [
            (r.consistency.get(col) if r and r.consistency else None)
            for r in results]

    return out

# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(df: pd.DataFrame, results: list, total: int):
    with open(CHECKPOINT, "wb") as f:
        pickle.dump({"results": results}, f)
    completed = sum(1 for r in results if r is not None)
    succeeded = sum(1 for r in results if r is not None and r.scores is not None)
    PROGRESS_FILE.write_text(json.dumps({
        "completed": completed, "succeeded": succeeded,
        "failed":    completed - succeeded, "total": total,
        "percent":   round(completed / total * 100, 2),
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

    # All rows scored (llm_quality_score_12d is present when scored)
    scored   = out[out["llm_quality_score_12d"].notna()].copy()
    n_scored = len(scored)
    n_total  = len(out)

    print(f"\n  Files total  : {n_total:,}")
    print(f"  Files scored : {n_scored:,}  ({n_scored/n_total*100:.1f}%)")
    print(f"  Files failed : {n_total - n_scored:,}")

    if n_scored == 0:
        print("\n  ✗ No scored rows.")
        print("═" * 80)
        return

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
            issues.append(f"{col}: out-of-range values")
            range_ok = False
    if range_ok:
        passes.append("All scores in valid range [0.0, 10.0]")

    comp = scored["llm_quality_score_12d"].dropna()
    if comp.mean() > 9.2:
        issues.append(f"Composite mean={comp.mean():.2f} — suspiciously HIGH")
    elif comp.mean() < 2.0:
        issues.append(f"Composite mean={comp.mean():.2f} — suspiciously LOW")
    else:
        passes.append(f"Composite mean={comp.mean():.2f} — healthy range")

    print(f"\n  {'Dimension':<22} {'mean':>6} {'std':>6} "
          f"{'min':>5} {'max':>5}  verdict")
    print("  " + "─" * 62)
    for d in DIMENSIONS:
        col  = scored[f"llm_{d}"].dropna()
        mean = col.mean()
        std  = col.std()
        v    = ("⚠ HIGH" if mean > 9.2 else
                "⚠ LOW"  if mean < 1.5 else "✓ ok")
        print(f"  {d:<22} {mean:>6.2f} {std:>6.2f} "
              f"{col.min():>5.1f} {col.max():>5.1f}  {v}")

    # Consistency report
    if "val_consistency_score" in scored.columns:
        print(f"\n  CONSISTENCY VALIDATION (scores not modified — audit only):")
        print(f"  {'File':<40} {'cons':>5}  {'HiConf':>7}  "
              f"{'framework':>10}  inconsistencies")
        print("  " + "─" * 88)

        active_checks = [
            ("val_error_handling_consistent",  "err"),
            ("val_security_consistent",        "sec"),
            ("val_anti_patterns_consistent",   "anti"),
            ("val_maintainability_consistent", "maint"),
        ]

        for _, r in scored.sort_values(
                "llm_quality_score_12d", ascending=False).iterrows():
            fname = str(r["file_path"]).replace("manual_files/", "")[:38]
            sc    = r.get("val_consistency_score", "?")
            hc    = "✓" if r.get("val_high_confidence") else "✗"
            fw    = str(r.get("val_detected_framework", "?"))[:10]
            bad   = [label for col, label in active_checks
                     if not r.get(col, True)]
            incon = ", ".join(bad) if bad else "—"
            print(f"  {fname:<40} {sc:>5}/4  {hc:>7}  {fw:>10}  {incon}")

        hi_conf = int(scored["val_high_confidence"].sum())
        mean_cs = scored["val_consistency_score"].mean()
        print(f"\n  High-confidence files  : {hi_conf}/{n_scored}")
        print(f"  Mean consistency score : {mean_cs:.2f}/3")

    print(f"\n  CHECKS PASSED ({len(passes)}):")
    for p in passes:
        print(f"    ✓  {p}")
    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for i in issues:
            print(f"    ⚠  {i}")
    else:
        print(f"\n  Overall verdict: ✓ SCORES LOOK VALID AND HEALTHY")
    print("═" * 80)

# ══════════════════════════════════════════════════════════════════════════════
#  FAILURE RE-RUN
# ══════════════════════════════════════════════════════════════════════════════

def rerun_failures(df: pd.DataFrame, results: list, total: int) -> list:
    timeout_idx = [
        i for i, r in enumerate(results)
        if r is not None and r.scores is None
        and r.error_types and "force_timeout" in r.error_types
    ]
    retry_idx = [
        i for i, r in enumerate(results)
        if (r is None or r.scores is None)
        and not (r is not None and r.error_types
                 and "force_timeout" in r.error_types)
    ]

    if timeout_idx:
        print(f"\n  ⏩ Skipping {len(timeout_idx)} timed-out file(s).")
    if not retry_idx:
        print("\n  ♻  No retryable failures.")
        return results

    print(f"\n  ♻  RE-RUN: {len(retry_idx)} retryable files…")
    pbar = tqdm(
        total=len(retry_idx), desc="  Re-scoring", unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )
    save_counter = 0
    for idx in retry_idx:
        row  = df.iloc[idx]
        path = str(row["file_path"])
        code = str(row["source_code"])
        try:
            results[idx] = score_single_file_with_timeout(
                idx, path, code,
                miner_cc=miner_cc,
                miner_max_cc=miner_max_cc,
            )
        except Exception:
            reason = f"Re-run exception: {traceback.format_exc()}"
            _log_audit(path, reason, [], [])
            results[idx] = ScoringResult(
                idx=idx, file_path=path, scores=None,
                confidence=0, failure_reason=reason)
        pbar.update(1)
        save_counter += 1
        if save_counter % SAVE_EVERY == 0:
            save_checkpoint(df, results, total)

    pbar.close()
    recovered    = sum(1 for i in retry_idx
                       if results[i] and results[i].scores)
    still_failed = len(retry_idx) - recovered + len(timeout_idx)
    print(f"\n  ♻  Re-run: {recovered} recovered | {still_failed} still failed")
    return results

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 80)
    print("  LLM CODE QUALITY SCORER — MANUAL FILES EDITION  v4.0")
    print(f"  Model                : {GROQ_MODEL}  (single model)")
    print(f"  Keys loaded          : {len(GROQ_API_KEYS)} / 5")
    for name, key in GROQ_API_KEYS:
        print(f"    {name:<20} {key[:8]}…{key[-4:]}")
    print(f"  Max code length      : {MAX_CODE_LEN:,} chars")
    print(f"  Rate limit           : {REQUESTS_PER_MINUTE} req/min")
    print(f"  Output columns       : 35  (down from 64 in v2)")
    print(f"  Validation checks    : 4 active (err_handling, security, anti_patterns, maintainability)")
    print(f"  val_high_confidence  : True only when ALL 4 checks pass (score = 4/4)")
    print(f"  LLM scores modified  : NEVER")
    print(f"  Input                : {INPUT_PARQUET}")
    print(f"  Dimensions changed   : framework_bp→maintainability | magic_values→security")
    print(f"  Output               : {OUTPUT_PARQUET}")
    print(f"  Started              : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("═" * 80)

    if not INPUT_PARQUET.exists():
        sys.exit(
            f"\n[FATAL] Input parquet not found: {INPUT_PARQUET}\n"
            f"  Run manual_files_miner.py first.\n"
        )
    if not MANUAL_FILES_DIR.exists():
        sys.exit(
            f"\n[FATAL] Manual files dir not found: {MANUAL_FILES_DIR}\n"
        )

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"\n  ↳ Parquet: {len(df):,} rows | {len(df.columns)} columns")
    df = load_source_code_from_disk(df)
    total = len(df)

    if total == 0:
        sys.exit("\n[FATAL] No source files loaded.\n")

    required_cols = ["loc", "cyclomatic_complexity", "num_functions", "max_cc"]
    missing_cols  = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        sys.exit(
            f"\n[FATAL] Miner columns missing: {missing_cols}\n"
            f"  Re-run manual_files_miner.py.\n"
        )

    # Resume from checkpoint
    results: list[Optional[ScoringResult]] = [None] * total
    if CHECKPOINT.exists():
        try:
            state  = pickle.load(open(CHECKPOINT, "rb"))
            loaded = state.get("results", [])
            if len(loaded) == total:
                results = loaded
                done = sum(1 for r in results if r is not None)
                print(f"  ↳ Resumed: {done}/{total} already scored.")
            else:
                print(f"  ↳ Checkpoint size mismatch — fresh start.")
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

    load_llm_cache()

    print("\n[STARTUP] Testing API keys…")
    if not test_api_connection():
        sys.exit("\n[FATAL] No working Groq API keys found.\n")

    interrupt_flag = [False]

    def _signal_handler(sig, frame):
        print("\n\n  ⚡ Ctrl+C — saving checkpoint…")
        interrupt_flag[0] = True

    signal.signal(signal.SIGINT, _signal_handler)
    _log_daily("session_started", {
        "keys": [n for n, _ in GROQ_API_KEYS], "files": total})

    if remaining == 0:
        print("  ✓ All files already scored — writing outputs.")
    else:
        est = remaining / REQUESTS_PER_MINUTE
        print(f"\n  ↳ Files to score : {remaining}")
        print(f"  ↳ Est. time      : ~{est:.1f} min at {REQUESTS_PER_MINUTE} req/min")

    save_counter = 0
    pbar = tqdm(
        total=remaining, desc="  Scoring", unit="file",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                   "[{elapsed}<{remaining}, {rate_fmt}]"
    )

    for idx, path, code, miner_cc, miner_max_cc in work:
        if interrupt_flag[0]:
            save_checkpoint(df, results, total)
            pbar.close()
            return

        try:
            results[idx] = score_single_file_with_timeout(
                idx, path, code,
                miner_cc=miner_cc,
                miner_max_cc=miner_max_cc,
            )
        except Exception:
            fp     = str(path)
            reason = f"Unhandled exception: {traceback.format_exc()}"
            _log_audit(fp, reason, [], [])
            results[idx] = ScoringResult(
                idx=idx, file_path=fp, scores=None,
                confidence=0, failure_reason=reason)

        pbar.update(1)
        save_counter += 1
        if save_counter % SAVE_EVERY == 0:
            save_checkpoint(df, results, total)

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
        results = rerun_failures(df, results, total)
    elif n_timed_out > 0:
        print(f"\n  ⏩ {n_timed_out} timed-out file(s) — skipping re-run.")

    save_checkpoint(df, results, total)

    # Write outputs
    print(f"\n  Writing final outputs…")
    write_done: list[Optional[pd.DataFrame]] = [None]

    def _write():
        try:
            out_local = build_output(df, results)
            out_local.to_parquet(OUTPUT_PARQUET, index=False, compression="snappy")
            out_local.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            write_done[0] = out_local
        except Exception as e:
            print(f"  ⚠ Output write error: {e}")

    wt = threading.Thread(target=_write, daemon=True)
    wt.start()
    wt.join(timeout=120)

    if write_done[0] is None:
        print("  ⚠ Output write failed — results in checkpoint.")
        return

    out: pd.DataFrame = write_done[0]  # type: ignore[assignment]
    succeeded = int(out["llm_quality_score_12d"].notna().sum())
    overall   = out["llm_quality_score_12d"].dropna()

    print(f"  ↳ Parquet : {out.shape[0]:,} rows × {out.shape[1]} cols "
          f"→ {OUTPUT_PARQUET.name}")
    print(f"  ↳ CSV     : {out.shape[0]:,} rows × {out.shape[1]} cols "
          f"→ {OUTPUT_CSV.name}")
    print(f"  ↳ Scored  : {succeeded}/{total}  |  "
          f"mean={overall.mean():.3f}  std={overall.std():.3f}")

    validate_scores(out)
    save_llm_cache()

    _log_daily("session_finished", {
        "total":      total,
        "succeeded":  succeeded,
        "failed":     total - succeeded,
        "mean":       round(float(overall.mean()), 3) if len(overall) else None,
        "cache_size": len(LLM_CACHE_DATA),
    })

    print(f"\n{'═' * 80}")
    print(f"  DONE")
    print(f"  Finished  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Cache     : {len(LLM_CACHE_DATA):,} total cached responses")
    print(f"  Log       : {DAILY_LOG}")
    print(f"{'═' * 80}\n")


if __name__ == "__main__":
    main()