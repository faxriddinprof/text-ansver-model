"""
extractor.py

Hybrid extractor: Local LLM (qwen2.5:7b via Ollama) + validation layer + keyword fallback.

Pipeline:
    text → split_text() → call_llm() → merge_results() → validate() → keyword_fallback() → data
"""

import re
import json
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"
CHUNK_SIZE   = 1500

FIELDS = [
    "uses_solar_energy",
    "is_coal_based_project",
    "involves_alcohol_or_tobacco",
    "building_energy_or_carbon_reduction_percent",
    "hydropower_capacity_mw",
    "hydropower_co2_emission_g_per_kwh",
    "has_compliance_certificate_from_authorized_body",
    "improves_water_supply_quality_or_efficiency",
    "reduces_ghg_emissions_in_production",
    "installs_dust_gas_filter_products",
]

BOOLEAN_FIELDS = {
    "uses_solar_energy",
    "is_coal_based_project",
    "involves_alcohol_or_tobacco",
    "has_compliance_certificate_from_authorized_body",
    "improves_water_supply_quality_or_efficiency",
    "reduces_ghg_emissions_in_production",
    "installs_dust_gas_filter_products",
}

NUMERIC_FIELDS = {
    "building_energy_or_carbon_reduction_percent",
    "hydropower_capacity_mw",
    "hydropower_co2_emission_g_per_kwh",
}

# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "")


def _ollama_available() -> bool:
    try:
        requests.get("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_text(text: str, max_length: int = CHUNK_SIZE) -> list:
    words = text.split()
    chunks, current, current_len = [], [], 0
    for word in words:
        wl = len(word) + 1
        if current_len + wl > max_length and current:
            chunks.append(" ".join(current))
            current, current_len = [word], wl
        else:
            current.append(word)
            current_len += wl
    if current:
        chunks.append(" ".join(current))
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """You are an information extraction system.

Extract structured data from the given text.

STRICT RULES:
- Do NOT assume based on keywords alone
- If the statement is NEGATIVE (e.g. "emas", "yo'q", "не", "нет", "не соответствует", "javob bermaydi") -> return false
- If information is unclear or missing -> return null
- Only return true if explicitly and positively confirmed in text
- Language may be Uzbek or Russian
- Text may be long and complex

FIELDS TO EXTRACT:
- uses_solar_energy: project uses solar panels or solar energy
- reduces_ghg_emissions_in_production: project reduces greenhouse gas emissions
- installs_dust_gas_filter_products: dust-gas filters are installed
- improves_water_supply_quality_or_efficiency: water supply or water efficiency is improved
- has_compliance_certificate_from_authorized_body: has official green/compliance certificate
- building_energy_or_carbon_reduction_percent: numeric % of energy/carbon reduction in buildings
- hydropower_capacity_mw: numeric MW capacity of hydropower plant
- hydropower_co2_emission_g_per_kwh: numeric CO2 emission in g/kWh for hydro
- is_coal_based_project: project uses or is based on coal
- involves_alcohol_or_tobacco: project involves alcohol or tobacco production/sales

OUTPUT FORMAT (return JSON ONLY, no explanation):
{
  "uses_solar_energy": true/false/null,
  "reduces_ghg_emissions_in_production": true/false/null,
  "installs_dust_gas_filter_products": true/false/null,
  "improves_water_supply_quality_or_efficiency": true/false/null,
  "has_compliance_certificate_from_authorized_body": true/false/null,
  "building_energy_or_carbon_reduction_percent": number or null,
  "hydropower_capacity_mw": number or null,
  "hydropower_co2_emission_g_per_kwh": number or null,
  "is_coal_based_project": true/false/null,
  "involves_alcohol_or_tobacco": true/false/null
}

Text:
---
{chunk}
---
"""


def _build_prompt(chunk: str) -> str:
    return PROMPT_TEMPLATE.replace("{chunk}", chunk)


# ---------------------------------------------------------------------------
# Parse LLM JSON response
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str) -> dict:
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Merge chunk results
# ---------------------------------------------------------------------------

def merge_results(results: list) -> dict:
    merged = {}

    for field in BOOLEAN_FIELDS:
        values = [r.get(field) for r in results if field in r]
        if any(v is True for v in values):
            merged[field] = True
        elif all(v is False for v in values if v is not None):
            merged[field] = False
        else:
            merged[field] = None  # unknown — goes to fallback

    for field in NUMERIC_FIELDS:
        merged[field] = None
        for r in results:
            val = r.get(field)
            if val is not None:
                try:
                    merged[field] = float(val)
                    break
                except (TypeError, ValueError):
                    pass

    return merged


# ---------------------------------------------------------------------------
# Validation layer — hard overrides (runs AFTER LLM)
# ---------------------------------------------------------------------------

def _validate(data: dict, text: str) -> dict:
    t = text.lower()

    # Critical exclusions: always force True if explicit keywords present
    if "ko'mir" in t or "ko\u02BBmir" in t or "\u0443\u0433\u043e\u043b\u044c" in t:
        data["is_coal_based_project"] = True

    if "alkogol" in t or "tamaki" in t or "\u0442\u0430\u0431\u0430\u043a" in t or "\u0430\u043b\u043a\u043e\u0433\u043e\u043b\u044c" in t:
        data["involves_alcohol_or_tobacco"] = True

    if "sertifikat" in t or "vakolatli organ" in t or "\u0441\u0435\u0440\u0442\u0438\u0444\u0438\u043a\u0430\u0442" in t:
        if data.get("has_compliance_certificate_from_authorized_body") is None:
            data["has_compliance_certificate_from_authorized_body"] = True

    return data


# ---------------------------------------------------------------------------
# Negation detection helper
# ---------------------------------------------------------------------------

# Negation words that can appear BEFORE or AFTER a keyword within a window
_NEGATION_WORDS = [
    "emas", "yo'q", "yo\u02BBq", "mavjud emas", "qilinmagan", "joriy etilmagan",
    "yo'naltirilmagan", "nazarda tutilmagan", "taqdim etilmagan",
    "not", "no ", "нет", "не ", "не соответствует", "не используется",
    "javob bermaydi", "mos kelmaydi",
]

def _negated(text: str, keyword_pos: int, window: int = 80) -> bool:
    """
    Return True if a negation word appears within `window` chars of the keyword position.
    Checks both before and after the keyword.
    """
    start = max(0, keyword_pos - window)
    end   = min(len(text), keyword_pos + window)
    snippet = text[start:end].lower()
    return any(neg in snippet for neg in _NEGATION_WORDS)


def _safe_keyword(text: str, keyword: str) -> bool:
    """
    Returns True only if keyword is present AND no negation is nearby.
    Uses strict mode: requires multiple independent signals for short keywords.
    """
    t = text.lower()
    pos = t.find(keyword.lower())
    if pos == -1:
        return False
    return not _negated(t, pos)


# ---------------------------------------------------------------------------
# Keyword fallback — fills in None fields
# ---------------------------------------------------------------------------

def _keyword_fallback(data: dict, text: str, mode: str = "balanced") -> dict:
    """
    Fill None fields using keyword heuristics.

    mode="balanced" : single keyword hit is enough (legacy)
    mode="strict"   : requires keyword present AND no negation nearby
    """
    t = text.lower()

    def check(keyword: str, *extra_keywords) -> bool:
        all_kws = [keyword] + list(extra_keywords)
        if mode == "strict":
            return any(_safe_keyword(t, kw) for kw in all_kws)
        else:
            return any(kw in t for kw in all_kws)

    fallbacks = {
        "uses_solar_energy":
            check("quyosh", "солнечн"),
        "installs_dust_gas_filter_products":
            check("filtr", "фильтр"),
        "improves_water_supply_quality_or_efficiency":
            (check("suv", "вод")) and check("samarad", "tejam", "qayta ishlash", "эффектив"),
        "reduces_ghg_emissions_in_production":
            check("issiqxona gaz", "парников") or check("chiqindi kamaytir", "выброс"),
        "is_coal_based_project":
            check("ko'mir", "ko\u02BBmir", "уголь"),
        "involves_alcohol_or_tobacco":
            check("alkogol", "tamaki", "табак", "алкоголь"),
        "has_compliance_certificate_from_authorized_body":
            check("sertifikat", "сертификат") and check("vakolatli", "муваффиқлик", "authorized"),
    }

    for field, result in fallbacks.items():
        if data.get(field) is None:
            data[field] = result

    if data.get("building_energy_or_carbon_reduction_percent") is None:
        m = re.search(r"(\d+)\s*%", t)
        if m:
            pct = int(m.group(1))
            # strict: ignore stray % values, require >= 5
            data["building_energy_or_carbon_reduction_percent"] = pct if pct >= 5 else None
        else:
            data["building_energy_or_carbon_reduction_percent"] = None

    if data.get("hydropower_capacity_mw") is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*mw", t)
        data["hydropower_capacity_mw"] = float(m.group(1)) if m else None

    if data.get("hydropower_co2_emission_g_per_kwh") is None:
        m = re.search(r"(\d+(?:\.\d+)?)\s*g(?:/|\s)kvt", t)
        data["hydropower_co2_emission_g_per_kwh"] = float(m.group(1)) if m else None

    return data


# ---------------------------------------------------------------------------
# LLM extraction pipeline
# ---------------------------------------------------------------------------

def _extract_with_llm(text: str) -> dict:
    chunks = split_text(text)
    chunk_results = []

    for i, chunk in enumerate(chunks):
        prompt = _build_prompt(chunk)
        try:
            raw = call_llm(prompt)
            parsed = _parse_llm_response(raw)
            print(f"[extractor] Raw LLM output (chunk {i+1}/{len(chunks)}): {parsed}")
            if parsed:
                chunk_results.append(parsed)
        except Exception as e:
            print(f"[extractor] Chunk {i+1} failed: {e}")

    if not chunk_results:
        return {f: None for f in FIELDS}

    return merge_results(chunk_results)


# ---------------------------------------------------------------------------
# Keyword-only pipeline (Ollama unavailable)
# ---------------------------------------------------------------------------

def _extract_with_keywords(text: str, mode: str = "balanced") -> dict:
    data = {f: None for f in FIELDS}
    data = _validate(data, text)
    return _keyword_fallback(data, text, mode=mode)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_data(text: str, mode: str = "balanced") -> dict:
    """
    Main extraction entry point.
    1. Try LLM (qwen2.5:7b via Ollama)
    2. Apply validation layer  — hard overrides
    3. Apply keyword fallback  — fill remaining None fields

    mode="balanced" : lenient keyword matching (default)
    mode="strict"   : negation-aware, reduces false positives
    """
    if _ollama_available():
        print(f"[extractor] Using LLM ({OLLAMA_MODEL})")
        data = _extract_with_llm(text)
        data = _validate(data, text)
        data = _keyword_fallback(data, text, mode=mode)
    else:
        print("[extractor] Ollama unavailable — using keyword fallback")
        data = _extract_with_keywords(text, mode=mode)

    print(f"[extractor] Final data after validation: {data}")
    return data


# ---------------------------------------------------------------------------
# ESG Holistic Analyst — SENIOR mode (primary TXT pipeline)
# ---------------------------------------------------------------------------

# Only the first N characters are sent to the LLM.
# Project purpose is always in the opening paragraphs; tails are legal
# boilerplate, governance text, and collateral lists that pollute the context.
ESG_ANALYST_MAX_CHARS = 6_000

ESG_ANALYST_PROMPT = """CRITICAL: Your ENTIRE response must be a single valid JSON object.
Do NOT write any explanation, summary, or text outside the JSON.
Start immediately with { and end with }.

You are a SENIOR ESG ANALYST working at a development bank.

Your task is to analyze a raw document (TXT, possibly noisy OCR) and determine
whether the project is a GREEN PROJECT or NOT GREEN.

This is NOT a simple keyword task. You must behave like a human ESG expert and
produce a justified, evidence-based conclusion.

---

### STEP 1 — IDENTIFY PROJECT CORE

Understand:
- What is the project?
- What is the loan used for?
- What industry does it belong to?

IGNORE: legal statutes, governance text, collateral descriptions, boilerplate.
FOCUS ONLY on: project purpose, investment description, technical implementation.

---

### STEP 2 — CHECK STOP FACTORS (STRICT)

Mark TRUE only if it is the MAIN activity of the project:
- coal-related activity
- oil & gas extraction
- alcohol production or trade
- tobacco production or trade
- gambling, weapons, radioactive materials

If just mentioned somewhere → IGNORE.

---

### STEP 3 — CHECK GREEN CRITERIA (STRICT & EVIDENCE-BASED)

Evaluate ONLY if clearly and explicitly stated:

1. renewable_energy: solar / wind / hydro / geothermal
2. energy_efficiency: ≥20% improvement OR clearly stated
3. ghg_reduction: explicit CO2 / GHG reduction statement
4. environmental_infrastructure: water supply improvement, recycling systems,
   or pollution control — REAL industrial usage only
5. certificate: EDGE / LEED / BREEAM or official equivalent

RULES:
- If unclear → FALSE
- If generic mention → FALSE
- If not directly part of project purpose → FALSE

For each criterion, provide a short evidence quote or explanation.

---

### STEP 4 — FINAL DECISION

IF any stop factor = TRUE:
    → NOT GREEN

ELSE:
    count criteria where value = true
    IF count >= 3: → GREEN
    ELSE: → NOT GREEN

---

### STEP 5 — OUTPUT (STRICT JSON ONLY)

{
  "project_summary": {
    "what_is_project": "...",
    "loan_purpose": "...",
    "industry": "..."
  },
  "stop_factors": {
    "triggered": false,
    "details": []
  },
  "green_criteria": {
    "renewable_energy": {
      "value": false,
      "evidence": ""
    },
    "energy_efficiency": {
      "value": false,
      "evidence": ""
    },
    "ghg_reduction": {
      "value": false,
      "evidence": ""
    },
    "environmental_infrastructure": {
      "value": false,
      "evidence": ""
    },
    "certificate": {
      "value": false,
      "evidence": ""
    }
  },
  "final_decision": "GREEN or NOT GREEN",
  "confidence": 0,
  "reasoning": "..."
}

---

CRITICAL RULES:
- Be conservative. If evidence is weak → FALSE.
- Do NOT rely on keywords alone.
- Do NOT guess or hallucinate.
- Base every decision ONLY on clear evidence from the text.
- Output ONLY the JSON. No text before {. No text after }.

---

Now analyze this document:

{{TEXT}}
"""


def _parse_esg_response(raw: str) -> dict:
    """Extract the outermost JSON object from a holistic ESG analyst response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# ESG Validation Layer — trust-but-verify (runs after LLM parsing)
# ---------------------------------------------------------------------------

# Keywords that MUST appear in text for a stop-factor claim to be accepted.
# Conservative: short, unambiguous tokens only.
_STOP_FACTOR_KEYWORDS = {
    "coal":     ["ko'mir", "ko\u02BBmir", "\u0443\u0433\u043e\u043b\u044c", "coal", "koks"],
    "oil_gas":  ["neft qazib", "neft-gaz", "oil extraction", "gas extraction"],
    "alcohol":  ["alkogol", "spirt", "\u0441\u043f\u0438\u0440\u0442",
                 "\u0430\u043b\u043a\u043e\u0433\u043e\u043b\u044c",
                 "\u043f\u0438\u0432\u043e\u0432\u0430\u0440", "brewery", "distill",
                 "wine produc", "beer produc"],
    "tobacco":  ["tamaki", "\u0442\u0430\u0431\u0430\u043a", "tobacco"],
    "gambling": ["qimor", "\u043a\u0430\u0437\u0438\u043d\u043e", "gambling", "casino"],
    "weapons":  ["qurol-yarog", "qurol ishlab", "\u043e\u0440\u0443\u0436\u0438\u0435", "weapon"],
}

# Keywords confirming each green criterion.
# Designed to avoid keyword fallback FP while not rejecting valid LLM findings.
_GREEN_CRITERION_KEYWORDS = {
    "renewable_energy": [
        "quyosh", "shamol", "solar", "wind", "gidro", "geotherm",
        "qayta tiklanuvchi", "\u0432\u043e\u0437\u043e\u0431\u043d\u043e\u0432\u043b\u044f\u0435\u043c", "renewable",
    ],
    "energy_efficiency": [
        "samaradorli", "tejamkorlik", "tejam", "efficiency",
        "\u044d\u043d\u0435\u0440\u0433\u043e\u044d\u0444\u0444\u0435\u043a\u0442\u0438\u0432\u043d",
        "energosberezhen",
    ],
    "ghg_reduction": [
        "co2", "issiqxona", "\u043f\u0430\u0440\u043d\u0438\u043a\u043e\u0432",
        "ghg", "greenhouse", "emissiya",
        "\u0432\u044b\u0431\u0440\u043e\u0441",
    ],
    "environmental_infrastructure": [
        "qayta ishlash", "suv t", "suv resur", "chiqindi suv",
        "chang-gaz filtr", "chang gaz filtr",
        "water treat", "recycl", "pollution control",
    ],
    "certificate": [
        "leed", "edge", "breeam", "sertifikat",
        "\u0441\u0435\u0440\u0442\u0438\u0444\u0438\u043a\u0430\u0442", "certificate",
    ],
}


def _build_safe_reasoning(esg: dict, inconsistencies=None) -> str:
    """Build a factual reasoning string from validated criteria."""
    criteria = esg.get("green_criteria", {})
    stop     = esg.get("stop_factors", {})
    summary  = esg.get("project_summary", {})
    lines    = []

    if summary.get("industry"):
        lines.append(f"Industry: {summary['industry']}.")

    if stop.get("triggered"):
        lines.append(f"Stop factors confirmed: {stop.get('details', [])}.")
        lines.append("Decision: NOT GREEN.")
    else:
        confirmed = [
            k for k, v in criteria.items()
            if isinstance(v, dict) and v.get("value") is True
        ]
        lines.append(
            f"Confirmed green criteria: {', '.join(confirmed)}."
            if confirmed else "No green criteria confirmed."
        )
        n = len(confirmed)
        lines.append(
            f"{n} criteria met (\u22653 required) \u2192 GREEN."
            if n >= 3 else
            f"Only {n} criteria met (<3 required) \u2192 NOT GREEN."
        )

    if inconsistencies:
        lines.append(
            f"[LLM reasoning was corrected: {'; '.join(inconsistencies)}]"
        )

    return " ".join(lines)


def _validate_esg_response(esg: dict, text: str) -> dict:
    """
    Trust-but-verify layer:  validate every LLM-claimed flag against the raw text.

    Actions:
    1. Stop-factor claims are verified by keyword presence → override if unsupported.
    2. Green-criteria TRUE claims are verified by domain keywords → override if absent.
    3. Reasoning text is checked for words that contradict the text → regenerated.
    4. Confidence replaced with a rule-based 0-100 score.

    Returns a deep copy of `esg` enriched with:
      llm_raw_flags     — original LLM outputs before any override
      rejected_flags    — list of overridden claims
      validation_notes  — human-readable explanation of each correction
      confidence        — rule-based score
    """
    import copy as _copy
    esg = _copy.deepcopy(esg)
    t   = text.lower()

    rejected_flags   = []
    validation_notes = []

    # ── Snapshot raw LLM flags ────────────────────────────────────────────
    raw_stop = esg.get("stop_factors", {}).get("triggered", False)
    raw_crit = {
        k: (v.get("value") if isinstance(v, dict) else v)
        for k, v in esg.get("green_criteria", {}).items()
    }
    esg["llm_raw_flags"] = {"stop_triggered": raw_stop, "criteria": raw_crit}

    # ── Validate stop factors ─────────────────────────────────────────────
    stop = esg.setdefault("stop_factors", {})
    if stop.get("triggered", False):
        llm_details    = list(stop.get("details", []))
        confirmed_cats = [
            cat for cat, kws in _STOP_FACTOR_KEYWORDS.items()
            if any(kw in t for kw in kws)
        ]
        if not confirmed_cats:
            stop["triggered"] = False
            stop["details"]   = []
            rejected_flags.append(f"stop_factors:{llm_details}")
            validation_notes.append(
                f"LLM flagged stop factors {llm_details} but no supporting "
                f"keywords found in text \u2192 overridden to FALSE"
            )
        else:
            stop["details"] = confirmed_cats

    # ── Validate green criteria ───────────────────────────────────────────
    criteria = esg.setdefault("green_criteria", {})
    for crit_name, crit_obj in criteria.items():
        if not (isinstance(crit_obj, dict) and crit_obj.get("value") is True):
            continue
        required_kws = _GREEN_CRITERION_KEYWORDS.get(crit_name, [])
        if required_kws and not any(kw in t for kw in required_kws):
            criteria[crit_name]["value"]    = False
            criteria[crit_name]["evidence"] = (
                f"[REJECTED: no '{crit_name}' domain keywords found in text]"
            )
            rejected_flags.append(crit_name)
            validation_notes.append(
                f"LLM claimed {crit_name}=True but no supporting "
                f"keywords found in text \u2192 overridden to FALSE"
            )

    # ── Reasoning consistency check ───────────────────────────────────────
    reason_lower   = esg.get("reasoning", "").lower()
    inconsistencies = [
        f"'{cat}' mentioned in reasoning but not in text"
        for cat, kws in _STOP_FACTOR_KEYWORDS.items()
        if cat in reason_lower and not any(kw in t for kw in kws)
    ]
    if inconsistencies:
        esg["reasoning"] = _build_safe_reasoning(esg, inconsistencies)
        validation_notes.append(
            f"Reasoning inconsistency detected \u2192 regenerated from facts. "
            f"Issues: {inconsistencies}"
        )

    # ── Rule-based confidence (replaces meaningless LLM 0-1% value) ──────
    n_confirmed = sum(
        1 for v in criteria.values()
        if isinstance(v, dict) and v.get("value") is True
    )
    confidence = 50 + n_confirmed * 10

    # Strong-signal bonus
    strong = {"renewable_energy", "ghg_reduction", "certificate"}
    for sc in strong:
        cv = criteria.get(sc, {})
        if isinstance(cv, dict) and cv.get("value") is True:
            confidence += 5

    if stop.get("triggered", False):
        confidence = max(confidence, 85)   # very clear NOT GREEN

    confidence -= len(rejected_flags) * 10
    if inconsistencies:
        confidence -= 15

    esg["confidence"]       = max(0, min(100, confidence))
    esg["rejected_flags"]   = rejected_flags
    esg["validation_notes"] = validation_notes

    return esg


def analyze_esg_holistic(text: str) -> dict:
    """
    Send the first ESG_ANALYST_MAX_CHARS characters to the LLM for a holistic,
    context-aware ESG analysis.  Returns the parsed JSON dict, or {} on failure.

    Only the opening of the document is used because:
    - Project purpose is always described first
    - Tails contain legal boilerplate / bylaws that confuse the model
    """
    body   = text[:ESG_ANALYST_MAX_CHARS]
    prompt = ESG_ANALYST_PROMPT.replace("{{TEXT}}", body)

    for attempt in range(1, 3):
        try:
            raw    = call_llm(prompt)
            parsed = _parse_esg_response(raw)

            # Validate: must have green_criteria with the expected nested structure
            criteria = parsed.get("green_criteria", {})
            if parsed and criteria and any(
                isinstance(v, dict) and "value" in v for v in criteria.values()
            ):
                # ── Validation layer ──────────────────────────────────────
                validated = _validate_esg_response(parsed, text)
                n_conf    = sum(
                    1 for v in validated.get("green_criteria", {}).values()
                    if isinstance(v, dict) and v.get("value") is True
                )
                rejected  = validated.get("rejected_flags", [])
                print(
                    f"[extractor] ESG analyst (attempt {attempt}): "
                    f"decision={validated.get('final_decision')}, "
                    f"score={n_conf}/5, confidence={validated.get('confidence')}"
                    + (f", rejected={rejected}" if rejected else "")
                )
                return validated

            print(f"[extractor] Attempt {attempt}: invalid response structure — retrying...")
            prompt += "\n\nREMINDER: Output ONLY the JSON object. No text before or after."

        except Exception as e:
            print(f"[extractor] ESG analyst attempt {attempt} error: {e}")

    print("[extractor] ESG holistic analysis failed — will fall back to rule engine")
    return {}
