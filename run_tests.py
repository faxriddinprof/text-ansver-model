"""
run_tests.py

Evaluation framework for the Green Project Evaluation System.

Usage:
    python3 run_tests.py
    python3 run_tests.py --quiet      # suppress extractor debug output
    python3 run_tests.py --file checks/simple.txt  # single file
"""

import json
import os
import sys

# Suppress extractor debug prints when --quiet is passed
QUIET = "--quiet" in sys.argv
SINGLE_FILE = None
for i, arg in enumerate(sys.argv):
    if arg == "--file" and i + 1 < len(sys.argv):
        SINGLE_FILE = sys.argv[i + 1]

if QUIET:
    import io
    sys.stdout = io.StringIO()

from src.utils.parser import read_txt
from src.utils.extractor import extract_data, _extract_with_keywords, _validate, _keyword_fallback
from src.utils.engine import evaluate

if QUIET:
    sys.stdout = sys.__stdout__

# Files larger than this limit use keyword-only extraction (LLM would time out)
LLM_CHAR_LIMIT = 20_000


def extract(text: str) -> dict:
    if len(text) > LLM_CHAR_LIMIT:
        print(f"  [warn] File too large for LLM ({len(text):,} chars) — using keyword fallback")
        data = _extract_with_keywords(text)
        return data
    return extract_data(text)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RULES_FILE   = os.path.join(BASE_DIR, "src", "utils", "green_rules.json")
EXPECTED_FILE = os.path.join(BASE_DIR, "checks", "expected_results.json")

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

with open(RULES_FILE, encoding="utf-8") as f:
    rules_json = json.load(f)

with open(EXPECTED_FILE, encoding="utf-8") as f:
    expected_list = json.load(f)

if SINGLE_FILE:
    expected_list = [e for e in expected_list if e["file"] == SINGLE_FILE]
    if not expected_list:
        print(f"[ERROR] '{SINGLE_FILE}' not found in expected_results.json")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

passed_count = 0
failed_count = 0
failed_details = []

SEPARATOR = "=" * 60

print(f"\n{SEPARATOR}")
print("  GREEN PROJECT EVALUATION — TEST SUITE")
print(f"{SEPARATOR}\n")

for entry in expected_list:
    rel_path  = entry["file"]
    expected  = entry["expected"]
    note      = entry.get("note", "")
    file_path = os.path.join(BASE_DIR, rel_path)

    file_label = os.path.basename(rel_path)

    print(f"[TEST] {file_label}")

    # --- File existence check ---
    if not os.path.exists(file_path):
        print(f"  Expected : {expected}")
        print(f"  Actual   : FILE NOT FOUND")
        print(f"  Result   : ⚠️  SKIP\n")
        failed_count += 1
        continue

    # --- Run pipeline (suppress debug output in quiet mode) ---
    if QUIET:
        sys.stdout = io.StringIO()

    text   = read_txt(file_path)
    data   = extract(text)
    result = evaluate(data, rules_json)

    if QUIET:
        sys.stdout = sys.__stdout__

    actual  = result["status"]
    score   = result["score"]
    reasons = result.get("decision_reasons", {})

    passed_rules      = reasons.get("passed_rules", [])
    failed_rules      = reasons.get("failed_rules", [])
    exclusions        = reasons.get("exclusions_triggered", [])
    dependent_rules   = reasons.get("dependent_rules_triggered", [])

    is_pass = (actual == expected)

    if is_pass:
        passed_count += 1
        status_icon = "✅ PASS"
    else:
        failed_count += 1
        status_icon = "❌ FAIL"
        failed_details.append({
            "file": rel_path,
            "expected": expected,
            "actual": actual,
            "score": score,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "exclusions_triggered": exclusions,
            "dependent_rules_triggered": dependent_rules,
        })

    print(f"  Expected : {expected}")
    print(f"  Actual   : {actual}  (score: {score})")
    print(f"  Result   : {status_icon}")
    if note:
        print(f"  Note     : {note}")
    print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total    = passed_count + failed_count
accuracy = (passed_count / total * 100) if total > 0 else 0.0

print(SEPARATOR)
print("  SUMMARY")
print(SEPARATOR)
print(f"  Total tests : {total}")
print(f"  Passed      : {passed_count}")
print(f"  Failed      : {failed_count}")
print(f"  Accuracy    : {accuracy:.1f}%")

if accuracy >= 85:
    grade = "🟢 EXCELLENT (≥ 85%)"
elif accuracy >= 70:
    grade = "🟡 ACCEPTABLE (≥ 70%)"
else:
    grade = "🔴 NEEDS IMPROVEMENT (< 70%)"

print(f"  Grade       : {grade}")
print()

# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

if failed_details:
    print(SEPARATOR)
    print("  ERROR ANALYSIS — FAILED CASES")
    print(SEPARATOR)

    for detail in failed_details:
        print(f"\n  File     : {detail['file']}")
        print(f"  Expected : {detail['expected']}")
        print(f"  Actual   : {detail['actual']}  (score: {detail['score']})")

        if detail["exclusions_triggered"]:
            print("  Exclusions triggered:")
            for ex in detail["exclusions_triggered"]:
                print(f"    - {ex}")

        if detail["dependent_rules_triggered"]:
            print("  Dependent rules triggered:")
            for dr in detail["dependent_rules_triggered"]:
                print(f"    - {dr}")

        if detail["passed_rules"]:
            print("  Passed rules:")
            for r in detail["passed_rules"]:
                print(f"    ✓ {r}")

        if detail["failed_rules"]:
            print("  Failed rules:")
            for r in detail["failed_rules"]:
                print(f"    ✗ {r}")

    print()

print(SEPARATOR)
print()

# Exit code: 0 = all passed, 1 = some failed
sys.exit(0 if failed_count == 0 else 1)
