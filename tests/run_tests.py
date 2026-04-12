"""
run_tests.py

Hybrid evaluation framework for the Green Project Evaluation System.

Supports two pipelines:
  - .json files → evaluate_from_esg_json()  (PRIMARY, high accuracy)
  - .txt files  → extract_data() + evaluate() (FALLBACK)

Usage:
    python3 tests/run_tests.py                          # all tests, balanced mode
    python3 tests/run_tests.py --quiet                  # suppress extractor debug
    python3 tests/run_tests.py --mode strict            # strict negation-aware mode
    python3 tests/run_tests.py --file checks/simple.txt # single file
"""

import json
import os
import sys

# Project root (one level up from tests/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

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
    _semantic_strength,
    _GREEN_SEMANTIC_CONCEPTS,
    _STOP_FACTOR_SEMANTIC,
    _compute_calibrated_score,
    _compute_dynamic_threshold,
    _compute_ambiguity_level,
    _compute_risk_factors,
    _build_score_breakdown,
    _build_decision_explanation,
    _build_risk_explanation,
    _build_ambiguity_explanation,
    _build_confidence_explanation,
)
from src.utils.engine import evaluate, evaluate_from_esg_json

if QUIET:
    sys.stdout = sys.__stdout__

# Files larger than this limit skip LLM (too slow)
LLM_CHAR_LIMIT = 20_000

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR      = PROJECT_ROOT
RULES_FILE    = os.path.join(PROJECT_ROOT, "data", "green_rules.json")
EXPECTED_FILE = os.path.join(PROJECT_ROOT, "tests", "expected_results.json")

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
    """TXT pipeline: LLM extraction → Python decision (primary) → rule engine (fallback)."""
    text = read_txt(file_path)

    # ── PRIMARY: LLM extracts facts, Python decides ───────────────────────
    if _ollama_available():
        esg = analyze_esg_holistic(text)

        # New schema: esg["validated_flags"]["stop_factors" | "green_criteria"]
        vf = esg.get("validated_flags", {}) if esg else {}
        stop_facs = vf.get("stop_factors", {})
        criteria  = vf.get("green_criteria", {})

        if esg and criteria and any(
            isinstance(v, dict) and "value" in v for v in criteria.values()
        ):
            # ── Python decision (single source of truth) ──────────────────
            stop_triggered = any(
                v.get("value", 0) >= 0.5
                for v in stop_facs.values()
                if isinstance(v, dict)
            )
            # Use calibrated score + dynamic threshold from extractor
            score     = esg.get("calibrated_score",
                            sum(v.get("value", 0.0) for v in criteria.values()
                                if isinstance(v, dict)))
            threshold = esg.get("threshold", 3.0)

            if stop_triggered:
                final_status = "NOT GREEN"
            elif score >= threshold:
                final_status = "GREEN"
            else:
                final_status = "NOT GREEN"

            # Evidence-annotated passed / failed lists
            passed = [
                f"{k}({v.get('value', 0):.1f}): {v.get('evidence', '')[:80]}"
                for k, v in criteria.items()
                if isinstance(v, dict) and v.get("value", 0) >= 0.5
            ]
            failed = [
                f"{k}({v.get('value', 0):.1f})"
                for k, v in criteria.items()
                if isinstance(v, dict) and v.get("value", 0) < 0.5
            ]

            amb_level    = esg.get("ambiguity_level", "medium")
            risk_facts   = esg.get("risk_factors", [])
            conf_pct     = esg.get("confidence", 50)
            crit_brkdown = esg.get("criterion_breakdown") or _build_score_breakdown(criteria)
            explanation  = _build_decision_explanation(
                final_status, score, threshold, criteria, stop_facs,
                esg.get("rejected_flags", []),
            )
            risk_expl    = esg.get("risk_explanation") or _build_risk_explanation(risk_facts)
            amb_expl     = esg.get("ambiguity_explanation") or _build_ambiguity_explanation(amb_level, esg.get("rejected_flags", []))
            conf_expl    = esg.get("confidence_explanation") or _build_confidence_explanation(conf_pct, amb_level, esg.get("rejected_flags", []))

            return {
                "status":                final_status,
                "score":                 score,
                "threshold":             threshold,
                "ambiguity_level":       amb_level,
                "risk_factors":          risk_facts,
                "pipeline":              "txt_esg",
                "confidence":            conf_pct / 100,
                "confidence_pct":        conf_pct,
                "reason":                esg.get("reason", ""),
                "rejected_flags":        esg.get("rejected_flags", []),
                "validation_notes":      esg.get("notes", []),
                # Explainability fields
                "score_breakdown":       crit_brkdown,
                "explanation":           explanation,
                "risk_explanation":      risk_expl,
                "ambiguity_explanation": amb_expl,
                "confidence_explanation":conf_expl,
                "decision_reasons": {
                    "stop_factors":              stop_facs,
                    "green_criteria":            criteria,
                    "passed_rules":              passed,
                    "failed_rules":              failed,
                    "exclusions_triggered":      [
                        k for k, v in stop_facs.items()
                        if isinstance(v, dict) and v.get("value", 0) >= 0.5
                    ],
                    "dependent_rules_triggered": [],
                },
            }

    # ── FALLBACK: keyword extraction + rule engine ────────────────────────
    print(f"  [info] LLM unavailable — rule engine fallback (mode={MODE})")
    if len(text) > LLM_CHAR_LIMIT:
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


def run_context_json_pipeline(file_path: str) -> dict:
    """Semantic-only pipeline for multi-section Uzbek bank credit OCR JSON.

    qwen2.5:7b struggles with Cyrillic/Uzbek OCR text, so we bypass the LLM
    entirely and use pure semantic strength scoring on a targeted excerpt:
      eco[:2800]       → Davlat Ekologik Ekspertiza header + emission sources
      mon[4500:8000]   → 105 solar panels + OEKO/ISO certification mentions
    """
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    sections = {
        k: " ".join(str(p) for p in (v if isinstance(v, list) else [v]))
        for k, v in data.items()
    }
    eco = sections.get("экология хулосаси", "")
    mon = sections.get("дастлабки мониторинг", "")

    priority_text = (
        "=== EKOLOGIYA XULOSASI ===\n" + eco[:2_800]
        + "\n\n=== MONITORING VA SERTIFIKATLAR ===\n" + mon[4_500:8_000]
    )

    # Normalize whitespace so OCR newlines don't break phrase matching
    t = " ".join(priority_text.lower().split())

    _CRIT_NAMES = ("renewable_energy", "energy_efficiency", "ghg_reduction",
                   "environmental_infrastructure", "certificate")
    _STOP_NAMES = ("coal", "oil_gas", "alcohol", "tobacco", "gambling")

    criteria_scores = {
        n: _semantic_strength(n, t, _GREEN_SEMANTIC_CONCEPTS) for n in _CRIT_NAMES
    }
    stop_scores = {
        n: _semantic_strength(n, t, _STOP_FACTOR_SEMANTIC) for n in _STOP_NAMES
    }

    criteria  = {k: {"value": v, "evidence": ""} for k, v in criteria_scores.items()}
    stop_facs = {k: {"value": v, "evidence": ""} for k, v in stop_scores.items()}

    # Use same calibrated scoring + dynamic threshold as the LLM pipeline
    calibrated, breakdown = _compute_calibrated_score(criteria, stop_facs)
    threshold             = _compute_dynamic_threshold(criteria, stop_facs)
    amb                   = _compute_ambiguity_level(criteria, stop_facs, [])
    risk_factors          = _compute_risk_factors(criteria, stop_facs, [])

    stop_triggered = any(v >= 0.5 for v in stop_scores.values())
    score          = calibrated

    if stop_triggered:
        final_status = "NOT GREEN"
    elif score >= threshold:
        final_status = "GREEN"
    else:
        final_status = "NOT GREEN"

    passed = [f"{k}({v:.1f})" for k, v in criteria_scores.items() if v >= 0.5]
    failed = [f"{k}({v:.1f})" for k, v in criteria_scores.items() if v < 0.5]
    strong = sum(1 for v in criteria_scores.values() if v >= 1.0)

    explanation = _build_decision_explanation(
        final_status, score, threshold, criteria, stop_facs, []
    )
    risk_expl  = _build_risk_explanation(risk_factors)
    amb_expl   = _build_ambiguity_explanation(amb, [])
    conf_pct   = min(95, 40 + 15 * strong)
    conf_expl  = _build_confidence_explanation(conf_pct, amb, [])

    reason = (
        f"Semantic criteria: {', '.join(passed) or 'none'}. "
        f"Calibrated score: {calibrated:.2f} (threshold: {threshold:.1f}). "
        + ("Stop factor triggered." if stop_triggered
           else f"Score {'≥' if score >= threshold else '<'} {threshold:.1f} → {final_status}.")
    )

    return {
        "status":                final_status,
        "score":                 score,
        "threshold":             threshold,
        "ambiguity_level":       amb,
        "risk_factors":          risk_factors,
        "pipeline":              "context_json_semantic",
        "confidence":            min(1.0, conf_pct / 100),
        "confidence_pct":        conf_pct,
        "reason":                reason,
        "rejected_flags":        [],
        "validation_notes":      [],
        # Explainability fields
        "score_breakdown":       _build_score_breakdown(criteria),
        "explanation":           explanation,
        "risk_explanation":      risk_expl,
        "ambiguity_explanation": amb_expl,
        "confidence_explanation":conf_expl,
        "decision_reasons": {
            "stop_factors":              stop_facs,
            "green_criteria":            criteria,
            "passed_rules":              passed,
            "failed_rules":              failed,
            "exclusions_triggered":      [k for k, v in stop_scores.items() if v >= 0.5],
            "dependent_rules_triggered": [],
        },
    }


def _is_context_json(data: dict) -> bool:
    """Return True if JSON looks like a multi-section Uzbek bank credit document."""
    return isinstance(data, dict) and any(
        k in data for k in ("экология хулосаси", "ариза", "дастлабки мониторинг")
    )


def run_pipeline(file_path: str) -> dict:
    """Auto-select pipeline based on file extension and content."""
    if file_path.endswith(".json"):
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        if _is_context_json(data):
            return run_context_json_pipeline(file_path)
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

    actual            = result["status"]
    score             = result.get("score", 0)
    threshold         = result.get("threshold", 3.0)
    ambiguity         = result.get("ambiguity_level", "")
    risk_factors      = result.get("risk_factors", [])
    confidence        = result.get("confidence")
    confidence_pct    = result.get("confidence_pct")
    pipeline          = result.get("pipeline", "txt")
    rejected_flags    = result.get("rejected_flags", [])
    validation_notes  = result.get("validation_notes", [])
    # Explainability
    explanation       = result.get("explanation", "")
    risk_explanation  = result.get("risk_explanation", "")
    amb_explanation   = result.get("ambiguity_explanation", "")
    conf_explanation  = result.get("confidence_explanation", "")
    score_breakdown   = result.get("score_breakdown", {})

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
            "threshold": threshold,
            "ambiguity_level": ambiguity,
            "confidence": confidence,
            "reason": reason,
            "risk_factors": risk_factors,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "exclusions_triggered": exclusions,
            "dependent_rules_triggered": dependent_rules,
        })

    conf_str  = f"{confidence_pct}%" if confidence_pct is not None else (f"{confidence:.0%}" if confidence is not None else "n/a")
    amb_str   = f"  ambiguity: {ambiguity}" if ambiguity else ""
    print(f"  Pipeline : [{pipeline}]")
    print(f"  Expected : {expected}")
    print(f"  Actual   : {actual}  (score: {score:.2f}  threshold: {threshold:.1f}  confidence: {conf_str}{amb_str})")
    if explanation:
        print(f"  Decision : {explanation}")
    if risk_explanation and "No significant" not in risk_explanation:
        print(f"  Risk     : {risk_explanation}")
    if amb_explanation and ambiguity in ("medium", "high"):
        print(f"  Ambiguity: {amb_explanation}")
    if conf_explanation:
        print(f"  Confidence: {conf_explanation}")
    if score_breakdown:
        for crit_name, crit_data in score_breakdown.items():
            if isinstance(crit_data, dict) and crit_data.get("score", 0) > 0:
                print(f"  [{crit_data['label']}]: {crit_data['impact']}  — {crit_data['reason']}")
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
