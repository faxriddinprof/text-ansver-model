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
