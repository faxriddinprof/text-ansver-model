"""
ESG evaluation pipeline helpers — kept separate from Django view logic.
"""

import json


def read_file_content(django_file) -> str:
    """Decode an InMemoryUploadedFile to a plain string."""
    raw = django_file.read()
    return raw.decode("utf-8", errors="replace")


def run_esg_pipeline(text: str) -> dict:
    """
    Run the full ESG evaluation on *text* using the extractor engine.
    Returns the raw result dict produced by analyze_esg_holistic.
    Raises RuntimeError if the engine fails.
    """
    from src.utils.extractor import analyze_esg_holistic, normalize_text  # noqa: PLC0415

    normalized = normalize_text(text)
    result = analyze_esg_holistic(normalized)
    if result is None:
        raise RuntimeError("ESG engine returned no result.")
    return result


def parse_result(result: dict) -> dict:
    """
    Flatten the raw engine result into the fields the template needs.
    Safe: always returns a complete dict even if some keys are absent.
    """
    # Inline imports — avoids circular dependency at module load time
    from src.utils.extractor import (  # noqa: PLC0415
        _build_decision_explanation,
        _build_missing_criteria_explanation,
    )

    vf       = result.get("validated_flags", {})
    criteria = vf.get("green_criteria", {})
    stops    = vf.get("stop_factors", {})

    status    = result.get("status", "UNKNOWN")
    score     = result.get("calibrated_score", result.get("score", 0.0))
    threshold = result.get("threshold", 3.0)
    rejected  = result.get("rejected_flags", [])
    eq_scores = result.get("evidence_quality", {})

    # Use post-penalty adjusted criteria (closer to what was actually scored)
    adj_crit = (
        result.get("_adj_crit_for_explanation")
        or result.get("penalty_breakdown", {}).get("adjusted_criteria")
        or criteria
    )
    gw_report = result.get("_gw_report_for_explanation") or {}

    # Build explanation — analyze_esg_holistic does NOT include this key
    explanation_raw = _build_decision_explanation(
        status, score, threshold, adj_crit, stops, rejected,
        greenwashing=gw_report or None,
    )
    explanation_sections = _parse_explanation_sections(explanation_raw)

    # Score breakdown list  [{name, score, score_pct, label, reason, impact}]
    breakdown_raw = result.get("criterion_breakdown") or {}
    breakdown = [
        {
            "name":      key,
            "score":     data.get("score", 0),
            "score_pct": int(round(data.get("score", 0) * 100)),
            "label":     data.get("label", key),
            "reason":    data.get("reason", ""),
            "impact":    data.get("impact", ""),
        }
        for key, data in breakdown_raw.items()
        if isinstance(data, dict)
    ]

    # Missing criteria — use engine value or build it ourselves
    missing_raw = result.get("missing_criteria") or {}
    if not missing_raw:
        missing_raw = _build_missing_criteria_explanation(
            adj_crit, stops, score, threshold, evidence_quality=eq_scores or None
        ) or {}
    missing = missing_raw.get("missing_criteria", [])

    # Risk / greenwashing
    risk_factors = result.get("risk_factors") or []
    gw_level     = result.get("greenwashing_risk_level", "low")
    gw_score     = result.get("greenwashing_risk_score", 0.0)
    gw_signals   = result.get("greenwashing_signals", [])
    notes        = result.get("notes", [])

    return {
        "verdict":           status,
        "score":             score,
        "threshold":         threshold,
        "confidence_pct":    result.get("confidence", 50),
        "ambiguity_level":   result.get("ambiguity_level", "medium"),
        "explanation_raw":   explanation_raw,
        "explanation":       explanation_sections,
        "breakdown":         breakdown,
        "missing":           missing,
        "missing_summary":   missing_raw.get("summary", ""),
        "risk_factors":      risk_factors,
        "risk_explanation":  result.get("risk_explanation", ""),
        "gw_level":          gw_level,
        "gw_score":          gw_score,
        "gw_signals":        gw_signals,
        "rejected_flags":    rejected,
        "audit_notes":       notes,
        "evidence_quality":  eq_scores,
    }


def _parse_explanation_sections(text: str) -> dict:
    """Split the multi-section explanation string into named parts."""
    sections = {"verdict": "", "why": "", "missing": "", "fix": ""}
    if not text:
        return sections

    mapping = {
        "VERDICT:":         "verdict",
        "WHY:":             "why",
        "SPECIFICALLY:":    "why",       # merge into why
        "WHAT IS MISSING:": "missing",
        "WHAT WOULD MAKE IT GREEN:": "fix",
    }

    current_key = "verdict"
    for line in text.splitlines():
        matched = False
        for marker, key in mapping.items():
            if line.startswith(marker):
                current_key = key
                body = line[len(marker):].strip()
                if body:
                    sections[current_key] = (sections[current_key] + " " + body).strip()
                matched = True
                break
        if not matched and line.strip():
            sections[current_key] = (sections[current_key] + " " + line.strip()).strip()

    return sections
