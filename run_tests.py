"""
run_tests.py

Hybrid evaluation framework for the Green Project Evaluation System.

Supports two pipelines:
  - .json files → evaluate_from_esg_json()  (PRIMARY, high accuracy)
  - .txt files  → extract_data() + evaluate() (FALLBACK)

Usage:
    python3 run_tests.py                          # all tests, balanced mode
    python3 run_tests.py --quiet                  # suppress extractor debug
    python3 run_tests.py --mode strict            # strict negation-aware mode
    python3 run_tests.py --file checks/simple.txt # single file
"""

import json
import os
import sys

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

QUIET       = "--quiet" in sys.argv
MODE        = "strict" if "--mode" in sys.argv and sys.argv[sys.argv.index("--mode") + 1] == "strict" else "balanced"
SINGLE_FILE = None

for i, arg in enumerate(sys.argv):
    if arg == "--file" and i + 1 < len(sys.argv):
        SINGLE_FILE = sys.argv[i + 1]

# ---------------------------------------------------------------------------
# Imports (suppress noisy extractor prints in quiet mode)
# ---------------------------------------------------------------------------

if QUIET:
    import io
    sys.stdout = io.StringIO()

from src.utils.parser import read_txt
from src.utils.extractor import (
    extract_data,
    _extract_with_keywords,
    _ollama_available,
    analyze_esg_holistic,
)
from src.utils.engine import evaluate, evaluate_from_esg_json

if QUIET:
    sys.stdout = sys.__stdout__

# Files larger than this limit skip LLM (too slow)
LLM_CHAR_LIMIT = 20_000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
RULES_FILE    = os.path.join(BASE_DIR, "src", "utils", "green_rules.json")
EXPECTED_FILE = os.path.join(BASE_DIR, "checks", "expected_results.json")

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
# Pipeline runners
# ---------------------------------------------------------------------------

def run_txt_pipeline(file_path: str) -> dict:
    """TXT pipeline: holistic ESG analyst (primary) → rule engine (fallback)."""
    text = read_txt(file_path)

    # ── PRIMARY: context-aware holistic analysis ──────────────────────────
    if _ollama_available():
        esg = analyze_esg_holistic(text)
        criteria = esg.get("green_criteria", {})

        # Validate that we got the expected nested structure
        if esg and criteria and any(
            isinstance(v, dict) and "value" in v for v in criteria.values()
        ):
            stop  = esg.get("stop_factors", {})
            score = sum(
                1 for v in criteria.values()
                if isinstance(v, dict) and v.get("value") is True
            )

            # Python enforces the decision — never trust LLM's final_decision
            if stop.get("triggered", False):
                final_status = "NOT GREEN"
            elif score >= 3:
                final_status = "GREEN"
            else:
                final_status = "NOT GREEN"

            # Build evidence-annotated passed / failed lists
            passed = [
                f"{k}: {v.get('evidence', '')[:100]}"
                for k, v in criteria.items()
                if isinstance(v, dict) and v.get("value") is True
            ]
            failed = [
                f"{k}: {v.get('evidence', '')[:100]}"
                for k, v in criteria.items()
                if isinstance(v, dict) and v.get("value") is False
            ]

            return {
                "status":          final_status,
                "score":           score,
                "pipeline":        "txt_esg",
                "confidence":      esg.get("confidence", 50) / 100,
                "reason":          esg.get("reasoning", ""),
                "rejected_flags":  esg.get("rejected_flags", []),
                "validation_notes": esg.get("validation_notes", []),
                "decision_reasons": {
                    "project_summary":           esg.get("project_summary", {}),
                    "stop_factors":              stop,
                    "green_criteria":            criteria,
                    "passed_rules":              passed,
                    "failed_rules":              failed,
                    "exclusions_triggered":      stop.get("details", []),
                    "dependent_rules_triggered": [],
                },
            }

    # ── FALLBACK: keyword extraction + rule engine ────────────────────────
    print(f"  [info] Holistic unavailable — rule engine fallback (mode={MODE})")
    if len(text) > LLM_CHAR_LIMIT:
        # Large docs: always strict to avoid keyword false positives
        data   = _extract_with_keywords(text, mode="strict")
        result = evaluate(data, rules_json, mode="strict")
    else:
        data   = extract_data(text, mode=MODE)
        result = evaluate(data, rules_json, mode=MODE)
    result["pipeline"] = "txt"
    return result


def run_json_pipeline(file_path: str) -> dict:
    """JSON pipeline: structured ESG JSON → evaluate_from_esg_json()."""
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)
    result = evaluate_from_esg_json(data)
    return result


def run_pipeline(file_path: str) -> dict:
    """Auto-select pipeline based on file extension."""
    if file_path.endswith(".json"):
        return run_json_pipeline(file_path)
    return run_txt_pipeline(file_path)

# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

passed_count  = 0
failed_count  = 0
failed_details = []
SEPARATOR = "=" * 62

print(f"\n{SEPARATOR}")
print(f"  GREEN PROJECT EVALUATION — TEST SUITE  [mode: {MODE}]")
print(f"{SEPARATOR}\n")

for entry in expected_list:
    rel_path   = entry["file"]
    expected   = entry["expected"]
    note       = entry.get("note", "")
    file_path  = os.path.join(BASE_DIR, rel_path)
    file_label = os.path.basename(rel_path)

    print(f"[TEST] {file_label}")

    if not os.path.exists(file_path):
        print(f"  Expected : {expected}")
        print(f"  Actual   : FILE NOT FOUND")
        print(f"  Result   : ⚠️  SKIP\n")
        failed_count += 1
        continue

    # Run pipeline, suppress debug in quiet mode
    if QUIET:
        import io as _io
        _buf = _io.StringIO()
        sys.stdout = _buf

    try:
        result = run_pipeline(file_path)
    except Exception as e:
        if QUIET:
            sys.stdout = sys.__stdout__
        print(f"  ERROR    : {e}")
        print(f"  Result   : ⚠️  SKIP\n")
        failed_count += 1
        continue

    if QUIET:
        sys.stdout = sys.__stdout__

    actual     = result["status"]
    score      = result.get("score", 0)
    confidence = result.get("confidence")
    pipeline   = result.get("pipeline", "txt")
    reason     = result.get("reason", "")
    rejected_flags   = result.get("rejected_flags", [])
    validation_notes = result.get("validation_notes", [])

    # TXT pipeline extras
    reasons          = result.get("decision_reasons", {})
    passed_rules     = reasons.get("passed_rules", [])
    failed_rules     = reasons.get("failed_rules", [])
    exclusions       = reasons.get("exclusions_triggered", [])
    dependent_rules  = reasons.get("dependent_rules_triggered", [])

    is_pass = (actual == expected)

    if is_pass:
        passed_count += 1
        icon = "✅ PASS"
    else:
        failed_count += 1
        icon = "❌ FAIL"
        failed_details.append({
            "file": rel_path,
            "pipeline": pipeline,
            "expected": expected,
            "actual": actual,
            "score": score,
            "confidence": confidence,
            "reason": reason,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "exclusions_triggered": exclusions,
            "dependent_rules_triggered": dependent_rules,
        })

    conf_str = f"  confidence: {confidence:.0%}" if confidence is not None else ""
    print(f"  Pipeline : [{pipeline}]")
    print(f"  Expected : {expected}")
    print(f"  Actual   : {actual}  (score: {score}{conf_str})")
    if reason:
        print(f"  Reason   : {reason}")
    if rejected_flags:
        for rf in rejected_flags:
            print(f"  Corrected: {rf}")
    if validation_notes:
        for vn in validation_notes:
            print(f"  [audit]  : {vn}")
    print(f"  Result   : {icon}")
    if note:
        print(f"  Note     : {note}")
    print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total    = passed_count + failed_count
accuracy = (passed_count / total * 100) if total > 0 else 0.0

# Split accuracy by pipeline type
txt_tests  = [e for e in expected_list if not e["file"].endswith(".json")]
json_tests = [e for e in expected_list if e["file"].endswith(".json")]
fail_files = {d["file"] for d in failed_details}
txt_pass   = sum(1 for e in txt_tests  if e["file"] not in fail_files)
json_pass  = sum(1 for e in json_tests if e["file"] not in fail_files)

print(SEPARATOR)
print("  SUMMARY")
print(SEPARATOR)
print(f"  Total tests : {total}")
print(f"  Passed      : {passed_count}")
print(f"  Failed      : {failed_count}")
print(f"  Accuracy    : {accuracy:.1f}%")
if txt_tests:
    print(f"  TXT  tests  : {txt_pass}/{len(txt_tests)} passed ({txt_pass/len(txt_tests)*100:.0f}%)")
if json_tests:
    print(f"  JSON tests  : {json_pass}/{len(json_tests)} passed ({json_pass/len(json_tests)*100:.0f}%)")

if accuracy >= 85:
    grade = "🟢 EXCELLENT  (≥ 85%)"
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
        print(f"\n  File       : {detail['file']}  [{detail['pipeline']}]")
        print(f"  Expected   : {detail['expected']}")
        print(f"  Actual     : {detail['actual']}  (score: {detail['score']})")
        if detail.get("confidence") is not None:
            print(f"  Confidence : {detail['confidence']:.0%}")
        if detail.get("reason"):
            print(f"  Reason     : {detail['reason']}")

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

sys.exit(0 if failed_count == 0 else 1)
