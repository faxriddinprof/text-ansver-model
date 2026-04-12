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
Do NOT write ANY explanation, summary, reasoning, or text outside the JSON.
Start immediately with { and end with }.

You are a structured INFORMATION EXTRACTION system for ESG documents.

YOUR ROLE: Extract facts ONLY.
DO NOT decide GREEN / NOT GREEN.
DO NOT compute score.
DO NOT write reasoning.
DO NOT assign confidence.

---

TASK: Read the document and fill in the JSON below.

For each field set "value" to true or false based ONLY on clear, explicit evidence.
- If unclear → false
- If not mentioned → false
- If mentioned in irrelevant context → false
Write a short "evidence" quote (max 120 chars) from the text. Empty string if false.

---

STOP FACTORS — mark true ONLY if this is the project's MAIN activity:

- coal:    project is primarily coal mining, coal power, or coal processing
- oil_gas: project is primarily oil/gas extraction or refining
- alcohol: project is primarily alcohol production, brewing, or distillery
- tobacco: project is primarily tobacco production or sales
- gambling: project is primarily gambling, casino, or betting

---

GREEN CRITERIA — mark true ONLY if explicitly and clearly stated as part of the project:

- renewable_energy:               solar / wind / hydro / geothermal used as main energy source
- energy_efficiency:              energy efficiency improvement ≥20% OR clearly stated improvement
- ghg_reduction:                  explicit CO2 / greenhouse gas reduction target or result
- environmental_infrastructure:   water treatment, recycling system, or industrial pollution control
- certificate:                    EDGE / LEED / BREEAM or official environmental certificate obtained

---

OUTPUT FORMAT — return ONLY this JSON, nothing else:

{
  "stop_factors": {
    "coal":     {"value": false, "evidence": ""},
    "oil_gas":  {"value": false, "evidence": ""},
    "alcohol":  {"value": false, "evidence": ""},
    "tobacco":  {"value": false, "evidence": ""},
    "gambling": {"value": false, "evidence": ""}
  },
  "green_criteria": {
    "renewable_energy":              {"value": false, "evidence": ""},
    "energy_efficiency":             {"value": false, "evidence": ""},
    "ghg_reduction":                 {"value": false, "evidence": ""},
    "environmental_infrastructure":  {"value": false, "evidence": ""},
    "certificate":                   {"value": false, "evidence": ""}
  }
}

---

Document to analyze:

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

# Synonyms / keywords that CONFIRM a stop-factor claim.
# A claim is accepted ONLY if at least one keyword is found in the text.
_STOP_FACTOR_KEYWORDS: dict[str, list[str]] = {
    "coal":     ["ko'mir", "ko\u02BBmir", "\u0443\u0433\u043e\u043b\u044c", "coal", "koks"],
    "oil_gas":  ["neft qazib", "neft-gaz", "oil extraction", "gas extraction",
                 "\u043d\u0435\u0444\u0442\u044c", "neft"],
    "alcohol":  ["alkogol", "spirt", "\u0441\u043f\u0438\u0440\u0442",
                 "\u0430\u043b\u043a\u043e\u0433\u043e\u043b\u044c",
                 "\u043f\u0438\u0432\u043e\u0432\u0430\u0440",
                 "brewery", "distill", "wine produc", "beer produc"],
    "tobacco":  ["tamaki", "\u0442\u0430\u0431\u0430\u043a", "tobacco"],
    "gambling": ["qimor", "\u043a\u0430\u0437\u0438\u043d\u043e", "gambling", "casino"],
}

# Synonyms / keywords that CONFIRM a green-criterion claim.
# A TRUE claim is accepted ONLY if at least one keyword group matches.
_GREEN_CRITERION_KEYWORDS: dict[str, list[str]] = {
    "renewable_energy": [
        # Uzbek
        "quyosh", "shamol", "gidro", "geotherm", "qayta tiklanuvchi",
        # Russian
        "\u0432\u043e\u0437\u043e\u0431\u043d\u043e\u0432\u043b\u044f\u0435\u043c",
        "\u0441\u043e\u043b\u043d\u0435\u0447\u043d",
        # English
        "solar", "wind", "hydro", "geotherm", "renewable",
    ],
    "energy_efficiency": [
        # Uzbek
        "samaradorli", "tejamkorlik", "tejam", "energiya tejam",
        # Russian
        "\u044d\u043d\u0435\u0440\u0433\u043e\u044d\u0444\u0444\u0435\u043a\u0442\u0438\u0432",
        "\u044d\u043d\u0435\u0440\u0433\u043e\u0441\u0431\u0435\u0440\u0435\u0436\u0435\u043d",
        # English
        "energy efficiency", "energy saving", "efficiency improvement",
    ],
    "ghg_reduction": [
        # Universal
        "co2", "ghg", "greenhouse",
        # Uzbek
        "issiqxona gaz", "emissiya", "chiqindi kamaytir",
        # Russian
        "\u043f\u0430\u0440\u043d\u0438\u043a\u043e\u0432",
        "\u0432\u044b\u0431\u0440\u043e\u0441",
        # English
        "emission reduction", "carbon reduction",
    ],
    "environmental_infrastructure": [
        # Uzbek
        "qayta ishlash", "suv tozalash", "suv resur", "chiqindi suv",
        "chang-gaz filtr", "chang gaz filtr", "kanalizatsiya",
        # Russian
        "\u0432\u043e\u0434\u043e\u043e\u0447\u0438\u0441\u0442\u043a",
        "\u0440\u0435\u0446\u0438\u043a\u043b\u0438\u043d\u0433",
        # English
        "water treat", "recycl", "pollution control", "wastewater",
    ],
    "certificate": [
        "leed", "edge", "breeam",
        # Uzbek (only if followed by "vakolatli" or "ekologik" to avoid bank certs)
        # — handled in logic below
        # English
        "environmental certificate", "green certificate",
        "iso 14001",
    ],
}

# For "certificate": plain "sertifikat" is too generic (bank docs have many).
# Accept it ONLY when accompanied by an ESG qualifier.
_CERT_ESG_QUALIFIERS = [
    "leed", "edge", "breeam", "ekologik", "yashil",
    "iso 14001", "vakolatli organ tomonidan", "environmental",
]


def _criterion_confirmed(crit_name: str, text_lower: str) -> bool:
    """Return True if domain keywords for `crit_name` are present in the text."""
    if crit_name == "certificate":
        # Strict: require an ESG-specific qualifier alongside any cert keyword
        has_cert = any(kw in text_lower for kw in ["sertifikat", "certificate", "сертификат"])
        has_esg  = any(q in text_lower for q in _CERT_ESG_QUALIFIERS)
        return has_cert and has_esg
    kws = _GREEN_CRITERION_KEYWORDS.get(crit_name, [])
    return bool(kws) and any(kw in text_lower for kw in kws)


def _build_safe_reasoning(validated_flags: dict, rejected: list, notes: list) -> str:
    """
    Build a 100% Python-generated reasoning string.
    Never uses LLM text.

    Parameters
    ----------
    validated_flags : dict  — after validation, keyed as:
        {"stop_factors": {"coal": {"value": bool}, ...},
         "green_criteria": {"renewable_energy": {"value": bool}, ...}}
    rejected        : list  — names of overridden fields
    notes           : list  — human-readable correction notes
    """
    lines = []

    # Stop factors
    triggered = [
        name for name, obj in validated_flags.get("stop_factors", {}).items()
        if isinstance(obj, dict) and obj.get("value") is True
    ]
    if triggered:
        lines.append(f"Stop factors confirmed: {', '.join(triggered)}.")
        lines.append("Decision: NOT GREEN (stop factor rule).")
        if rejected:
            lines.append(f"Corrections applied: {', '.join(rejected)}.")
        return " ".join(lines)

    # Green criteria
    confirmed = [
        name for name, obj in validated_flags.get("green_criteria", {}).items()
        if isinstance(obj, dict) and obj.get("value") is True
    ]
    n = len(confirmed)
    if confirmed:
        lines.append(f"Confirmed green criteria ({n}/5): {', '.join(confirmed)}.")
    else:
        lines.append("No green criteria confirmed.")

    if n >= 3:
        lines.append(f"{n} \u22653 required \u2192 GREEN.")
    else:
        lines.append(f"Only {n} of 3 required criteria met \u2192 NOT GREEN.")

    if rejected:
        lines.append(f"Overridden LLM claims: {', '.join(rejected)}.")

    return " ".join(lines)


def _compute_confidence(validated_flags: dict, rejected: list) -> int:
    """
    Rule-based confidence score (0–100).

    Base: 50
    +10  per confirmed green criterion
    +10  if stop factor confirmed (very certain NOT GREEN)
    +5   per strong-signal criterion (renewable_energy, ghg_reduction, certificate)
    -15  per rejected (overridden) LLM claim
    Clamp to [0, 100].
    """
    criteria  = validated_flags.get("green_criteria", {})
    stops     = validated_flags.get("stop_factors", {})
    n_conf    = sum(1 for v in criteria.values() if isinstance(v, dict) and v.get("value"))
    any_stop  = any(v.get("value") for v in stops.values() if isinstance(v, dict))

    confidence = 50 + n_conf * 10
    if any_stop:
        confidence += 10   # very certain

    strong = {"renewable_energy", "ghg_reduction", "certificate"}
    for sc in strong:
        cv = criteria.get(sc, {})
        if isinstance(cv, dict) and cv.get("value"):
            confidence += 5

    confidence -= len(rejected) * 15
    return max(0, min(100, confidence))


def _validate_esg_response(raw_esg: dict, text: str) -> dict:
    """
    Trust-but-verify layer.

    Validates every LLM-claimed TRUE flag against the raw document text
    using keyword/synonym checks.  Overrides unsupported claims to FALSE.

    Returns a new dict:
    {
        "validated_flags": {
            "stop_factors":   {name: {"value": bool, "evidence": str}, ...},
            "green_criteria": {name: {"value": bool, "evidence": str}, ...}
        },
        "llm_raw_output": <original parsed LLM dict>,
        "rejected_flags": [str, ...],
        "notes":          [str, ...],
    }
    """
    import copy as _copy
    t = text.lower()

    rejected: list[str] = []
    notes:    list[str] = []

    validated: dict = {
        "stop_factors":   {},
        "green_criteria": {},
    }

    # ── Validate stop factors ─────────────────────────────────────────────
    for name, obj in raw_esg.get("stop_factors", {}).items():
        if not isinstance(obj, dict):
            validated["stop_factors"][name] = {"value": False, "evidence": ""}
            continue

        val  = obj.get("value", False)
        evid = obj.get("evidence", "")

        if val:
            kws = _STOP_FACTOR_KEYWORDS.get(name, [])
            if kws and not any(kw in t for kw in kws):
                val  = False
                evid = f"[REJECTED: no '{name}' keywords in text]"
                rejected.append(f"stop:{name}")
                notes.append(
                    f"LLM claimed stop_factor '{name}'=True but no supporting "
                    f"keywords found \u2192 overridden to False"
                )

        validated["stop_factors"][name] = {"value": val, "evidence": evid}

    # ── Validate green criteria ───────────────────────────────────────────
    for name, obj in raw_esg.get("green_criteria", {}).items():
        if not isinstance(obj, dict):
            validated["green_criteria"][name] = {"value": False, "evidence": ""}
            continue

        val  = obj.get("value", False)
        evid = obj.get("evidence", "")

        if val and not _criterion_confirmed(name, t):
            val  = False
            evid = f"[REJECTED: no '{name}' domain keywords in text]"
            rejected.append(f"criteria:{name}")
            notes.append(
                f"LLM claimed '{name}'=True but no supporting keywords "
                f"found in text \u2192 overridden to False"
            )

        validated["green_criteria"][name] = {"value": val, "evidence": evid}

    return {
        "validated_flags": validated,
        "llm_raw_output":  _copy.deepcopy(raw_esg),
        "rejected_flags":  rejected,
        "notes":           notes,
    }


def analyze_esg_holistic(text: str) -> dict:
    """
    Call the LLM for structured EXTRACTION ONLY (no decision).
    Then pass the result through the validation layer.

    Returns the validated dict from _validate_esg_response, or {} on failure.
    """
    body   = text[:ESG_ANALYST_MAX_CHARS]
    prompt = ESG_ANALYST_PROMPT.replace("{{TEXT}}", body)

    for attempt in range(1, 3):
        try:
            raw    = call_llm(prompt)
            parsed = _parse_esg_response(raw)

            # Require both sections with the correct nested structure
            criteria  = parsed.get("green_criteria", {})
            stop_facs = parsed.get("stop_factors", {})
            if (parsed
                    and any(isinstance(v, dict) and "value" in v for v in criteria.values())
                    and any(isinstance(v, dict) and "value" in v for v in stop_facs.values())):

                result   = _validate_esg_response(parsed, text)
                rejected = result.get("rejected_flags", [])
                notes    = result.get("notes", [])
                vf       = result["validated_flags"]

                # Python computes confidence + reasoning (never from LLM)
                result["confidence"] = _compute_confidence(vf, rejected)
                result["reason"]     = _build_safe_reasoning(vf, rejected, notes)

                n_conf = sum(
                    1 for v in vf["green_criteria"].values()
                    if isinstance(v, dict) and v.get("value")
                )
                print(
                    f"[extractor] LLM extraction (attempt {attempt}): "
                    f"green_criteria_true={n_conf}/5, "
                    f"confidence={result['confidence']}"
                    + (f", overridden={rejected}" if rejected else "")
                )
                return result

            print(
                f"[extractor] Attempt {attempt}: unexpected JSON structure — retrying...\n"
                f"  keys found: stop_factors={list(stop_facs.keys())}, "
                f"criteria={list(criteria.keys())}"
            )
            prompt += "\n\nREMINDER: Output ONLY the JSON. No text before or after."

        except Exception as e:
            print(f"[extractor] Attempt {attempt} error: {e}")

    print("[extractor] LLM extraction failed after retries — falling back to rule engine")
    return {}
