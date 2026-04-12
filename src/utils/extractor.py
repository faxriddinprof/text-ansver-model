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

You are a structured SEMANTIC INFORMATION EXTRACTION system for ESG documents.

YOUR ROLE: Extract facts ONLY using SEMANTIC UNDERSTANDING.
DO NOT decide GREEN / NOT GREEN.
DO NOT compute score.
DO NOT write reasoning.
DO NOT assign confidence.

---

TASK: Read the document and fill in the JSON using MEANING, not just keywords.

Recognize semantic equivalents and paraphrases:
- "photovoltaic generation" = solar energy → renewable_energy: true
- "low-carbon electricity from natural sources" = renewable energy → renewable_energy: true
- "CO2 emission reduction target of 30%" = GHG reduction → ghg_reduction: true
- "energy consumption optimized by 25%" = energy efficiency → energy_efficiency: true
- "wastewater recycling facility" = environmental infrastructure → environmental_infrastructure: true

For each field set "value" to true or false.
- If unclear or not mentioned → false
- If CO2 is only mentioned as a measurement level (not reduced) → ghg_reduction: false
- If certificate/sertifikat is bank or quality only — NOT environmental → certificate: false
- If food fermentation/brewing is not the project's main activity → alcohol: false

Write a short "evidence" quote (max 120 chars) from the text. Empty string if false.

---

STOP FACTORS — true ONLY if this is the project's PRIMARY CORE activity:

- coal:    primary business = coal mining, coal power plant, or coal processing
- oil_gas: primary business = oil/gas extraction or refining
- alcohol: primary business = alcohol production, brewery, or distillery
- tobacco: primary business = tobacco production or cigarette manufacturing
- gambling: primary business = casino, gambling facility, or betting operations

DISAMBIGUATION:
- Renewable energy project near coal region → coal: false
- Oil used as auxiliary fuel for non-oil project → oil_gas: false
- Juice/food factory with fermentation step → alcohol: false
- Brewing company → alcohol: true

---

GREEN CRITERIA — true ONLY if explicitly and clearly part of the project:

- renewable_energy:              solar / wind / hydro / geothermal / photovoltaic as MAIN power source
                                 "low-carbon electricity generation", "clean energy generation" also qualify
- energy_efficiency:             energy efficiency improvement ≥20% OR clearly stated energy optimization
                                 "energy-saving reconstruction", "thermal insulation upgrade" also qualify
- ghg_reduction:                 explicit CO2/GHG REDUCTION target or achieved result — not just emission levels
                                 "emission reduction", "carbon footprint reduced", "decarbonization" qualify
- environmental_infrastructure:  water treatment facility, recycling plant, industrial pollution control
                                 "dust-gas filters installed", "wastewater management system" also qualify
- certificate:                   EDGE / LEED / BREEAM / ISO 14001 or official ENVIRONMENTAL certificate
                                 Generic bank, quality, or compliance certificates do NOT qualify

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


# ---------------------------------------------------------------------------
# Semantic Concept Maps  (3 tiers: strong → 1.0 | partial → 0.6 | weak → 0.3)
# ---------------------------------------------------------------------------
# Each entry is a list of SYNONYM GROUPS.  ANY phrase match within a group
# counts as a hit for that tier.  Highest tier wins.
# New maps use actual UTF-8 Cyrillic chars for readability.

_GREEN_SEMANTIC_CONCEPTS: dict[str, dict[str, list[list[str]]]] = {
    "renewable_energy": {
        "strong": [
            ["quyosh panellari", "quyosh paneli", "solar panel", "photovoltaic",
             "қуёш панеллари", "қуёш панели"],
            ["қуёш панеллари", "қуёш панели", "солнечные панели", "солнечная панель"],
            ["shamol turbina", "wind turbine", "ветрогенератор", "ветроустановка"],
            ["gidroelektrostantsiya", "hydroelectric station", "гидроэлектростанция", "гэс"],
            ["geothermal", "геотермальн"],
            ["solar farm", "solar park", "wind farm", "wind power plant"],
        ],
        "partial": [
            ["quyosh energiya", "solar energy", "солнечная энергия", "solar power",
             "қуёш энергия", "қуёш энергиясидан"],
            ["қуёш энергия", "қуёш энергияси"],
            ["shamol energiya", "wind energy", "ветровая энергия", "wind power"],
            ["qayta tiklanuvchi energiya", "renewable energy", "возобновляемая энергия"],
            ["past karbon elektr", "low-carbon electricity", "low carbon generation",
             "low-carbon generation"],
            ["toza energiya", "clean energy", "чистая энергия", "green energy"],
            ["gidroelektr", "гидроэнергетик", "гидроэлектр", "gidroenergiya"],
            ["alternativ energiya", "alternative energy source"],
        ],
        "weak": [
            ["quyosh", "solar", "солнечн"],
            ["shamol", "ветер"],
            ["gidro", "hydro"],
            ["yashil energiya", "green electricity"],
        ],
    },
    "energy_efficiency": {
        "strong": [
            ["energiya tejamkorlik darajasi 2", "energiya tejamkorlik 2",
             "energy efficiency 20%", "20% energy saving",
             "energiya samaradorligini 2", "energiya samaradorligi 2",
             "energiya samaradorligi 3", "energy efficiency improvement"],
            ["iso 50001"],
            ["energiya sinfi", "energy class", "энергетический класс"],
        ],
        "partial": [
            ["energiya tejamkorlik", "energy saving", "энергосбережен"],
            ["energiya samaradorligi", "energy efficiency", "энергоэффективн"],
            ["issiqlik izolyatsiya", "thermal insulation", "теплоизоляц"],
            ["led yoritish", "led lighting", "светодиодн"],
            ["energiya iste'molini kamaytirish", "reduce energy consumption",
             "снижение энергопотребления"],
            ["past energiya iste'moli", "low energy consumption"],
            ["aqlli energiya", "smart energy management", "smart grid"],
            ["rekonstruksiya qilish", "energy-saving reconstruction"],
        ],
        "weak": [
            ["tejamkorlik", "tejam"],
            ["samaradorlik oshirish", "efficiency improvement"],
            ["issiqlik tejash", "heat saving", "теплосбережен"],
        ],
    },
    "ghg_reduction": {
        "strong": [
            ["co2 kamaytirish", "co2 reduction", "снижение co2", "reduce co2 emissions"],
            ["issiqxona gazi kamaytirish", "issiqxona gazlar kamaytirish",
             "greenhouse gas reduction", "сокращение парниковых газов"],
            ["karbon izini kamaytirish", "carbon footprint reduction",
             "снижение углеродного следа"],
            ["net zero", "carbon neutral", "net-zero", "decarbonization",
             "dekarbonizatsiya"],
            ["co2 chiqindilari kamaytir", "co2 chiqindilari kamay"],
        ],
        "partial": [
            ["co2", "co\u2082"],
            ["karbon emissiya", "carbon emission", "выброс углерода"],
            ["issiqxona gaz", "greenhouse gas", "парниковый газ"],
            ["past karbon", "low-carbon", "низкоуглеродн"],
            ["emissiyani kamaytirish", "emission reduction", "снижение выбросов"],
            ["uglerod kamaytirish", "декарбонизац"],
        ],
        "weak": [
            ["emissiya", "emission", "выброс"],
            ["iqlim o'zgarish", "climate change", "изменение климата"],
        ],
    },
    "environmental_infrastructure": {
        "strong": [
            ["tashkil qilingan manba"],
            ["chang-gaz filtr", "chang gaz filtr", "dust-gas filter",
             "пылегазоулавлив"],
            ["suv tozalash inshoot", "water treatment facility",
             "сооружения водоочист"],
            ["chiqindi qayta ishlash zavod", "waste recycling plant",
             "завод по переработке отход"],
            ["oqova suv tozalash", "oqava suv tozalash", "wastewater treatment",
             "очистка сточных вод"],
            ["sanoat ifloslanishini nazorat", "industrial pollution control"],
            ["tashkil qilingan manba"],
        ],
        "partial": [
            ["suv tozalash", "water treatment", "водоочист"],
            ["chiqindi qayta ishlash", "waste recycling", "рециклинг",
             "recycling system"],
            ["chiqindi suv", "chiqindi suvlar", "chiqindi suvni"],
            ["oqova suv", "oqava suv", "wastewater", "сточные воды"],
            ["filtr qurilma", "filtration system", "система фильтрации"],
            ["kanalizatsiya tozalash", "sewage treatment", "очистка канализации"],
            ["ifloslanishni nazorat", "pollution control", "контроль загрязнения"],
            ["chiqindilarni boshqarish", "waste management", "управление отходами"],
            ["qayta ishlash texnologiya", "recycling technology"],
        ],
        "weak": [
            ["filtr", "filter"],
            ["qayta ishlash", "recycling"],
            ["ekologik himoya", "environmental protection facility"],
        ],
    },
    "certificate": {
        "strong": [
            ["leed sertifikat", "leed certified", "leed certification", "leed-certified"],
            ["edge sertifikat", "edge certified", "edge certification"],
            ["breeam sertifikat", "breeam certified", "breeam certification"],
            ["iso 14001"],
            ["davlat ekologik ekspertiza xulosasi", "ekologik ekspertiza xulosasi"],
            ["давлат экологик экспертиза", "экологик экспертиза хулосаси"],
        ],
        "partial": [
            ["leed"],
            ["breeam"],
            ["ekologik sertifikat", "environmental certificate",
             "экологический сертификат"],
            ["yashil bino sertifikat", "green building certificate"],
            ["vakolatli organ tomonidan"],
            ["atrof-muhit boshqaruv sertifikat",
             "environmental management certificate"],
            ["оеко", "oeko-tex", "oeko tex"],
        ],
        "weak": [],  # plain "sertifikat" alone never qualifies
    },
}

_STOP_FACTOR_SEMANTIC: dict[str, dict[str, list[list[str]]]] = {
    "coal": {
        "strong": [
            ["ko'mir qazib olish", "ko\u02BBmir qazib", "coal mining",
             "добыча угля", "угледобыча"],
            ["ko'mir elektr stansiya", "ko\u02BBmir elektr", "coal power plant",
             "coal-fired power", "угольная электростанция"],
            ["ko'mirni qayta ishlash", "coal processing", "переработка угля"],
            ["koks ishlab chiqarish", "coke production", "коксохим"],
        ],
        "partial": [
            ["ko'mir", "ko\u02BBmir", "coal", "уголь"],
        ],
        "weak": [],
    },
    "oil_gas": {
        "strong": [
            ["neft qazib olish", "oil extraction", "добыча нефти", "нефтедобыча"],
            ["gaz qazib olish", "gas extraction", "добыча газа", "газодобыча"],
            ["neftni rafinirlash", "oil refining", "нефтепереработ"],
            ["neft-gaz kompaniya", "oil and gas company", "нефтегазовая компания"],
        ],
        "partial": [
            ["neft qazib", "petroleum extraction", "oil well"],
            ["gaz qazib", "natural gas extraction"],
            ["neft-gaz", "oil-gas", "нефтегаз"],
        ],
        "weak": [
            ["neft", "oil", "нефть"],
        ],
    },
    "alcohol": {
        "strong": [
            ["alkogol ishlab chiqarish", "alcohol production",
             "производство алкоголя"],
            ["spirt zavodi", "distillery", "спиртозавод", "спиртовой завод"],
            ["pivo zavodi", "brewery", "пивоварня", "пивоваренный завод"],
            ["vino zavodi", "winery", "винодельня"],
            ["araq ishlab chiqarish", "vodka production"],
        ],
        "partial": [
            ["alkogol", "alcohol", "алкоголь"],
            ["spirt", "ethanol", "спирт"],
            ["pivo ishlab chiqar", "beer production", "пивоварение"],
            ["vino ishlab chiqar", "wine production", "виноделие"],
        ],
        "weak": [
            ["fermentatsiya", "fermentation", "брожение"],
        ],
    },
    "tobacco": {
        "strong": [
            ["tamaki ishlab chiqarish", "tobacco production",
             "производство табака"],
            ["sigaret ishlab chiqarish", "cigarette manufacturing"],
            ["tamaki kompaniya", "tobacco company", "табачная компания"],
        ],
        "partial": [
            ["tamaki", "tobacco", "табак"],
            ["sigaret", "cigarette", "сигарет"],
        ],
        "weak": [],
    },
    "gambling": {
        "strong": [
            ["kazino", "casino", "казино"],
            ["qimor uyi", "gambling house", "игорный дом"],
            ["tikish kompaniya", "betting company", "букмекерская компания"],
        ],
        "partial": [
            ["qimor", "gambling", "азартн"],
            ["tikish", "betting", "ставки"],
            ["lotereya", "lottery", "лотерея"],
        ],
        "weak": [],
    },
}


def _semantic_strength(name: str, text_lower: str, concept_map: dict) -> float:
    """
    Return evidence strength (0.0 | 0.3 | 0.6 | 1.0) via tiered semantic matching.
    Checks concept tiers strong → partial → weak.
    Each tier is a list of synonym groups; any single phrase hit within a group
    counts as a match for that tier.  Highest tier wins.
    """
    entry = concept_map.get(name, {})
    for phrase_group in entry.get("strong", []):
        if any(phrase in text_lower for phrase in phrase_group):
            return 1.0
    for phrase_group in entry.get("partial", []):
        if any(phrase in text_lower for phrase in phrase_group):
            return 0.6
    for phrase_group in entry.get("weak", []):
        if any(phrase in text_lower for phrase in phrase_group):
            return 0.3
    return 0.0


def _build_safe_reasoning(validated_flags: dict, rejected: list, notes: list) -> str:
    """
    Build a 100% Python-generated reasoning string from float-scored flags.
    Never uses LLM text.
    """
    lines = []
    stops = validated_flags.get("stop_factors", {})
    crits = validated_flags.get("green_criteria", {})

    # Stop factors (threshold >= 0.5 = partial or strong evidence)
    triggered = [
        f"{name}({obj['value']:.1f})"
        for name, obj in stops.items()
        if isinstance(obj, dict) and obj.get("value", 0) >= 0.5
    ]
    if triggered:
        lines.append(f"Stop factors triggered: {', '.join(triggered)}.")
        lines.append("Decision: NOT GREEN (exclusion rule).")
        if rejected:
            lines.append(f"Overridden claims: {', '.join(rejected)}.")
        return " ".join(lines)

    # Green criteria
    total_score = sum(
        obj.get("value", 0.0) for obj in crits.values() if isinstance(obj, dict)
    )
    confirmed = [
        f"{name}({obj['value']:.1f})"
        for name, obj in crits.items()
        if isinstance(obj, dict) and obj.get("value", 0) >= 0.5
    ]
    weak_sigs = [
        f"{name}({obj['value']:.1f})"
        for name, obj in crits.items()
        if isinstance(obj, dict) and 0 < obj.get("value", 0) < 0.5
    ]
    if confirmed:
        lines.append(f"Confirmed criteria: {', '.join(confirmed)}.")
    if weak_sigs:
        lines.append(f"Weak signals: {', '.join(weak_sigs)}.")
    lines.append(f"Total evidence score: {total_score:.2f}.")
    if total_score >= 3.0:
        lines.append("Score \u2265 3.0 \u2192 GREEN.")
    else:
        lines.append(f"Score {total_score:.2f} < 3.0 \u2192 NOT GREEN.")

    if rejected:
        lines.append(f"Overridden LLM claims: {', '.join(rejected)}.")
    return " ".join(lines)


# ---------------------------------------------------------------------------
# Calibrated Scoring Engine
# ---------------------------------------------------------------------------

def _compute_dynamic_threshold(criteria: dict, stops: dict) -> float:
    """
    Project-type-aware GREEN decision threshold [2.7 – 3.3].

    certified renewable  → 2.7   (clean credentials, lower bar)
    oil/coal adjacent    → 3.3   (higher scrutiny required)
    pure industrial      → 3.2   (no clean-energy evidence at all)
    standard             → 3.0
    """
    r  = criteria.get("renewable_energy", {})
    c  = criteria.get("certificate", {})
    oi = stops.get("oil_gas", {})
    co = stops.get("coal", {})

    r_val  = r.get("value",  0.0) if isinstance(r,  dict) else 0.0
    c_val  = c.get("value",  0.0) if isinstance(c,  dict) else 0.0
    oi_val = oi.get("value", 0.0) if isinstance(oi, dict) else 0.0
    co_val = co.get("value", 0.0) if isinstance(co, dict) else 0.0

    if r_val >= 1.0 and c_val >= 0.6:
        return 2.7   # certified renewable project — easier bar
    if 0.3 <= oi_val < 0.5 or 0.3 <= co_val < 0.5:
        return 3.3   # oil/coal adjacency — higher scrutiny
    if r_val == 0.0 and c_val == 0.0:
        return 3.2   # pure industrial, no clean-energy evidence
    return 3.0       # standard


def _compute_calibrated_score(criteria: dict, stops: dict) -> tuple:
    """
    Calibrated scoring with penalties for weak-only evidence and soft contradictions.

    ambiguity_penalty    — when ALL signals are weak (< 0.5), each adds 0.2 penalty
    contradiction_penalty — soft stop signals (0.3–0.49) indicate unresolved industry risk

    Score clamped to [0, 5]. Returns (final_score: float, breakdown: dict).
    """
    raw = sum(v.get("value", 0.0) for v in criteria.values() if isinstance(v, dict))

    confirmed = sum(
        1 for v in criteria.values()
        if isinstance(v, dict) and v.get("value", 0) >= 0.5
    )
    weak = sum(
        1 for v in criteria.values()
        if isinstance(v, dict) and 0.0 < v.get("value", 0) < 0.5
    )

    # Ambiguity penalty: only weak signals present, no confirmed evidence
    ambiguity_penalty = 0.2 * weak if confirmed == 0 else 0.0

    # Contradiction penalty: soft stop signals present (concerning but not blocking)
    soft_stop_sum = sum(
        v.get("value", 0.0) for v in stops.values()
        if isinstance(v, dict) and 0.3 <= v.get("value", 0) < 0.5
    )
    contradiction_penalty = 0.3 if soft_stop_sum > 0 else 0.0

    final = max(0.0, min(5.0, raw - ambiguity_penalty - contradiction_penalty))
    return final, {
        "raw":                   round(raw, 3),
        "ambiguity_penalty":     round(ambiguity_penalty, 3),
        "contradiction_penalty": round(contradiction_penalty, 3),
        "final":                 round(final, 3),
    }


def _compute_ambiguity_level(criteria: dict, stops: dict, rejected: list) -> str:
    """
    Returns 'low' | 'medium' | 'high' based on evidence quality.

    high   — many rejected claims, soft stops present, or only weak criteria signals
    medium — one rejected claim or one weak criterion
    low    — confirmed evidence, no rejections, no soft stops
    """
    confirmed  = sum(
        1 for v in criteria.values()
        if isinstance(v, dict) and v.get("value", 0) >= 0.5
    )
    weak       = sum(
        1 for v in criteria.values()
        if isinstance(v, dict) and 0.0 < v.get("value", 0) < 0.5
    )
    soft_stops = sum(
        1 for v in stops.values()
        if isinstance(v, dict) and 0.3 <= v.get("value", 0) < 0.5
    )
    n_rejected = len(rejected)

    if n_rejected >= 2 or soft_stops >= 1 or (confirmed == 0 and weak >= 2):
        return "high"
    if n_rejected == 1 or weak == 1:
        return "medium"
    return "low"


def _compute_risk_factors(criteria: dict, stops: dict, rejected: list) -> list:
    """Build an ordered list of human-readable risk signals for audit trail."""
    factors = []

    for k, v in stops.items():
        if isinstance(v, dict) and v.get("value", 0) >= 0.5:
            factors.append(f"STOP:{k}(score={v['value']:.1f})")
        elif isinstance(v, dict) and v.get("value", 0) >= 0.3:
            factors.append(f"SOFT_RISK:{k}(score={v['value']:.1f})")

    for r in rejected:
        factors.append(f"REJECTED:{r}")

    for k, v in criteria.items():
        if isinstance(v, dict) and 0.0 < v.get("value", 0) < 0.5:
            factors.append(f"WEAK_EVIDENCE:{k}(score={v['value']:.1f})")

    return factors


# ---------------------------------------------------------------------------
# Explainability Layer
# ---------------------------------------------------------------------------

_CRITERION_DISPLAY: dict[str, str] = {
    "renewable_energy":             "Renewable Energy",
    "energy_efficiency":            "Energy Efficiency",
    "ghg_reduction":                "GHG / Emissions Reduction",
    "environmental_infrastructure": "Environmental Infrastructure",
    "certificate":                  "Environmental Certification",
}

_CRITERION_REASONS: dict[str, dict] = {
    "renewable_energy": {
        1.0: "Strong evidence of renewable energy systems (solar panels, wind turbines, or hydropower) confirmed in the document.",
        0.6: "Partial evidence of renewable energy use identified, but details are incomplete.",
        0.3: "Indirect references to renewable energy found, but not sufficiently substantiated.",
        0.0: "No renewable energy systems identified in this project.",
    },
    "energy_efficiency": {
        1.0: "Documented energy efficiency improvements with measurable targets confirmed.",
        0.6: "Energy efficiency measures mentioned, but full implementation details are missing.",
        0.3: "General references to energy savings found, but not substantiated with data.",
        0.0: "No measurable energy efficiency improvements identified.",
    },
    "ghg_reduction": {
        1.0: "Quantified greenhouse gas or emissions reduction commitments clearly documented.",
        0.6: "Emissions reduction measures mentioned, but supporting data is incomplete.",
        0.3: "Indirect references to emissions reduction found.",
        0.0: "No greenhouse gas or emissions reduction evidence found.",
    },
    "environmental_infrastructure": {
        1.0: "Environmental control systems (filters, treatment facilities, emission controls) confirmed.",
        0.6: "Environmental management systems evidenced, but not fully detailed.",
        0.3: "General environmental management references found, without specific infrastructure.",
        0.0: "No environmental infrastructure or control systems identified.",
    },
    "certificate": {
        1.0: "Valid environmental certification (state eco-approval, ISO 14001, LEED, or equivalent) confirmed.",
        0.6: "Environmental certification indicators found, but not fully verified.",
        0.3: "References to certification found, but type and validity are unclear.",
        0.0: "No environmental certification found. Note: quality certificates (ISO 9001) do not qualify as ESG certification.",
    },
}

_STOP_RISK_MESSAGES: dict[str, str] = {
    "coal":     "The project has direct involvement in coal-related activities, which is a critical environmental exclusion under green financing standards.",
    "oil_gas":  "The project has exposure to oil and gas operations. This significantly increases environmental risk and requires stronger evidence of mitigating measures.",
    "alcohol":  "The project involves alcohol production, which is excluded from green project classification.",
    "tobacco":  "The project involves tobacco-related activities, which are excluded from green financing.",
    "gambling": "The project involves gambling activities, which are excluded from green financing.",
}


def _criterion_reason(name: str, score: float) -> str:
    """Return a human-readable sentence explaining why a criterion received its score."""
    reasons = _CRITERION_REASONS.get(name)
    if not reasons:
        if score >= 1.0:
            return "Strong evidence confirmed in the document."
        if score >= 0.5:
            return "Partial evidence found in the document."
        if score > 0.0:
            return "Weak or indirect evidence only."
        return "No supporting evidence found."
    if score >= 1.0:
        return reasons[1.0]
    if score >= 0.5:
        return reasons[0.6]
    if score > 0.0:
        return reasons[0.3]
    return reasons[0.0]


def _build_score_breakdown(criteria: dict) -> dict:
    """
    Per-criterion breakdown for business users.

    Each entry: {"label", "score", "impact", "reason"}
    """
    breakdown = {}
    for name, obj in criteria.items():
        if not isinstance(obj, dict):
            continue
        score   = obj.get("value", 0.0)
        label   = _CRITERION_DISPLAY.get(name, name.replace("_", " ").title())
        impact  = f"+{score:.1f}" if score > 0 else "0.0"
        reason  = _criterion_reason(name, score)
        breakdown[name] = {
            "label":  label,
            "score":  round(score, 2),
            "impact": impact,
            "reason": reason,
        }
    return breakdown


def _build_decision_explanation(
    decision: str,
    score: float,
    threshold: float,
    criteria: dict,
    stops: dict,
    rejected: list,
) -> str:
    """
    Plain-language paragraph explaining the final classification.
    Written for non-technical managers, auditors, and regulators.
    """
    confirmed = [
        _CRITERION_DISPLAY.get(k, k.replace("_", " ").title())
        for k, v in criteria.items()
        if isinstance(v, dict) and v.get("value", 0) >= 0.5
    ]
    missing = [
        _CRITERION_DISPLAY.get(k, k.replace("_", " ").title())
        for k, v in criteria.items()
        if isinstance(v, dict) and v.get("value", 0) < 0.5
    ]
    stop_triggered = [
        k for k, v in stops.items()
        if isinstance(v, dict) and v.get("value", 0) >= 0.5
    ]

    if stop_triggered:
        stop_names = " and ".join(
            _STOP_RISK_MESSAGES.get(k, k.replace("_", " ")) for k in stop_triggered[:2]
        )
        return (
            f"This project is classified as NOT GREEN due to the presence of "
            f"activities that are excluded from green financing. "
            f"Specifically: {stop_names} "
            f"ESG criteria cannot override an exclusion trigger."
        )

    if decision == "GREEN":
        conf_list = ", ".join(confirmed) if confirmed else "multiple criteria"
        return (
            f"This project qualifies as GREEN. "
            f"The following environmental criteria are confirmed: {conf_list}. "
            f"The total evidence score ({score:.2f}) meets the required threshold "
            f"({threshold:.1f}) for a project of this type."
        )

    # NOT GREEN — insufficient score
    parts = []
    if confirmed:
        parts.append(
            f"Although {', '.join(confirmed)} {'is' if len(confirmed)==1 else 'are'} "
            f"present, the evidence is insufficient on its own."
        )
    if missing:
        parts.append(
            f"No supporting evidence was found for: {', '.join(missing)}."
        )
    if rejected:
        parts.append(
            f"Additionally, {len(rejected)} claim(s) made in the document could "
            f"not be verified against the document content and were excluded from scoring."
        )
    score_sentence = (
        f"The total ESG score ({score:.2f}) is below the required "
        f"threshold ({threshold:.1f}) for a project of this type."
    )
    base = (
        "This project is classified as NOT GREEN because it lacks sufficient "
        "environmental impact evidence. "
    )
    return base + " ".join(parts) + " " + score_sentence


def _build_risk_explanation(risk_factors: list) -> str:
    """Convert machine risk flags into natural language risk assessment."""
    if not risk_factors:
        return "No significant risk factors identified."

    sentences = []
    for flag in risk_factors:
        if flag.startswith("STOP:"):
            key = flag.split(":")[1].split("(")[0]
            msg = _STOP_RISK_MESSAGES.get(key, f"Critical exclusion triggered: {key}.")
            sentences.append(msg)
        elif flag.startswith("SOFT_RISK:"):
            key = flag.split(":")[1].split("(")[0]
            base = _STOP_RISK_MESSAGES.get(key, f"Potential risk: {key}.")
            sentences.append(
                base.replace("has direct involvement", "has indirect exposure")
                    .replace("has exposure", "shows indicators of exposure")
                    .replace("significantly increases", "may increase")
            )
        elif flag.startswith("REJECTED:"):
            part = flag.replace("REJECTED:criteria:", "").replace("REJECTED:stop:", "")
            name = _CRITERION_DISPLAY.get(part, part.replace("_", " ").title())
            sentences.append(
                f"A claim regarding '{name}' in the document could not be confirmed "
                f"by independent evidence verification and was excluded from scoring."
            )
        elif flag.startswith("WEAK_EVIDENCE:"):
            key = flag.split(":")[1].split("(")[0]
            name = _CRITERION_DISPLAY.get(key, key.replace("_", " ").title())
            sentences.append(
                f"The evidence for '{name}' is weak and inconclusive."
            )

    if not sentences:
        return "No significant risk factors identified."
    return " ".join(sentences)


def _build_ambiguity_explanation(ambiguity_level: str, rejected: list) -> str:
    """Plain-language explanation of the evidence ambiguity level."""
    if ambiguity_level == "low":
        return (
            "Evidence quality is good. The document contains clear, verifiable "
            "information that supports the evaluation."
        )
    if ambiguity_level == "medium":
        if rejected:
            return (
                "The evaluation contains moderate uncertainty. One or more statements "
                "in the document could not be independently verified, which reduces "
                "confidence in the result."
            )
        return (
            "The evaluation contains moderate uncertainty due to limited or "
            "indirect evidence in the document."
        )
    # high
    if rejected:
        return (
            "The evaluation has high uncertainty. Multiple claims in the document "
            "were not supported by verifiable content, or conflicting signals are "
            "present. Independent due diligence is strongly recommended before "
            "making a financing decision."
        )
    return (
        "The evaluation has high uncertainty due to weak, indirect, or conflicting "
        "evidence. The document does not provide sufficient clarity for a "
        "high-confidence assessment. Independent review is recommended."
    )


def _build_confidence_explanation(
    confidence: int, ambiguity_level: str, rejected: list
) -> str:
    """Plain-language explanation of the confidence score."""
    pct = f"{confidence}%"
    if confidence >= 75:
        level = "high"
        base  = f"Confidence is high ({pct}). The decision is well-supported by strong, verifiable evidence."
    elif confidence >= 50:
        level = "moderate"
        base  = f"Confidence is moderate ({pct}). The decision is based on partial evidence."
    else:
        level = "low"
        base  = f"Confidence is low ({pct}). The available evidence is limited, weak, or inconsistent."

    reasons = []
    if ambiguity_level == "high":
        reasons.append("high ambiguity in the source document")
    elif ambiguity_level == "medium":
        reasons.append("moderate ambiguity in the source document")
    if rejected:
        n = len(rejected)
        reasons.append(f"{n} unverified claim{'s' if n > 1 else ''} rejected during validation")

    if reasons and level != "high":
        base += f" Contributing factors: {'; '.join(reasons)}."
    return base


def _compute_confidence(validated_flags: dict, rejected: list) -> int:
    """
    Evidence-based confidence (0–100).

    Considers: signal strength, contradiction level, data completeness,
    ambiguity level, and number of rejected LLM claims — NOT just score.
    Clamped to [10, 95].
    """
    crits = validated_flags.get("green_criteria", {})
    stops = validated_flags.get("stop_factors",  {})

    strong_count    = sum(
        1 for v in crits.values() if isinstance(v, dict) and v.get("value", 0) >= 1.0
    )
    confirmed_count = sum(
        1 for v in crits.values() if isinstance(v, dict) and v.get("value", 0) >= 0.5
    )
    any_stop  = any(v.get("value", 0) >= 0.5 for v in stops.values() if isinstance(v, dict))
    soft_stop = any(
        0.3 <= v.get("value", 0) < 0.5 for v in stops.values() if isinstance(v, dict)
    )

    amb         = _compute_ambiguity_level(crits, stops, rejected)
    amb_penalty = {"low": 0, "medium": 8, "high": 18}[amb]

    confidence  = 35
    confidence += 15 * strong_count
    confidence +=  8 * confirmed_count
    if any_stop:
        confidence += 12
    if soft_stop:
        confidence -= 5
    confidence -= 12 * len(rejected)
    confidence -= amb_penalty

    return max(10, min(95, confidence))


def _validate_esg_response(raw_esg: dict, text: str) -> dict:
    """
    Semantic trust-but-verify layer  (replaces keyword-only validation).

    For every LLM claim this layer:
      1. Computes an independent semantic strength score (0.0–1.0) from the text.
      2. LLM True  + semantic > 0.0 → use semantic score (validated).
      3. LLM True  + semantic = 0.0 → override to 0.0 (rejected).
      4. LLM False + semantic = 1.0 → promote to 0.7
             (LLM missed a strong semantic signal).
      5. Otherwise 0.0.

    Returns:
    {
        "validated_flags": {
            "stop_factors":   {name: {"value": float, "evidence": str}, ...},
            "green_criteria": {name: {"value": float, "evidence": str}, ...}
        },
        "llm_raw_output": <original parsed LLM dict>,
        "rejected_flags": [str, ...],
        "notes":          [str, ...],
    }
    """
    import copy as _copy
    # Normalize whitespace so OCR line-breaks don't break phrase matching
    t = ' '.join(text.lower().split())

    # Canonical names the LLM must use; map common deviations back
    _CRIT_ALIASES: dict[str, str] = {
        "sustainability_certificates": "certificate",
        "sustainability_certificate":  "certificate",
        "green_certificate":           "certificate",
        "environmental_certificate":   "certificate",
        "renewables":                  "renewable_energy",
        "solar_energy":                "renewable_energy",
        "clean_energy":                "renewable_energy",
        "energy_saving":               "energy_efficiency",
        "ghg":                         "ghg_reduction",
        "emission_reduction":          "ghg_reduction",
        "env_infrastructure":          "environmental_infrastructure",
    }

    rejected: list[str] = []
    notes:    list[str] = []
    validated: dict = {"stop_factors": {}, "green_criteria": {}}

    # ── Validate stop factors ─────────────────────────────────────────────
    for name, obj in raw_esg.get("stop_factors", {}).items():
        if not isinstance(obj, dict):
            validated["stop_factors"][name] = {"value": 0.0, "evidence": ""}
            continue

        llm_val = obj.get("value", False)
        evid    = obj.get("evidence", "")
        sem_s   = _semantic_strength(name, t, _STOP_FACTOR_SEMANTIC)

        if llm_val and sem_s == 0.0:
            evid = f"[REJECTED: no semantic evidence for '{name}' stop factor]"
            rejected.append(f"stop:{name}")
            notes.append(
                f"LLM claimed stop_factor '{name}'=True but semantic search "
                f"found no supporting evidence → overridden to 0.0"
            )
            final = 0.0
        elif llm_val:
            final = sem_s
        elif sem_s >= 1.0:
            final = 0.7
            notes.append(
                f"stop_factor '{name}': strong semantic evidence (score=1.0) "
                f"without LLM confirmation → promoted to 0.7"
            )
        else:
            final = 0.0

        validated["stop_factors"][name] = {"value": final, "evidence": evid}

    # ── Validate green criteria ───────────────────────────────────────────
    for raw_name, obj in raw_esg.get("green_criteria", {}).items():
        name = _CRIT_ALIASES.get(raw_name, raw_name)  # normalise LLM key
        if not isinstance(obj, dict):
            validated["green_criteria"][name] = {"value": 0.0, "evidence": ""}
            continue

        llm_val = obj.get("value", False)
        evid    = obj.get("evidence", "")
        sem_s   = _semantic_strength(name, t, _GREEN_SEMANTIC_CONCEPTS)

        if llm_val and sem_s == 0.0:
            evid = f"[REJECTED: no semantic evidence for '{name}']"
            rejected.append(f"criteria:{name}")
            notes.append(
                f"LLM claimed '{name}'=True but semantic search found no "
                f"supporting evidence → overridden to 0.0"
            )
            final = 0.0
        elif llm_val:
            final = sem_s
        elif sem_s >= 1.0:
            final = 0.7
            notes.append(
                f"'{name}': strong semantic evidence (score=1.0) without "
                f"LLM confirmation → promoted to 0.7"
            )
        else:
            final = 0.0

        validated["green_criteria"][name] = {"value": final, "evidence": evid}

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

                # Calibrated scoring + risk profiling (new fields)
                vfc = vf["green_criteria"]
                vfs = vf["stop_factors"]
                calibrated, breakdown = _compute_calibrated_score(vfc, vfs)
                amb        = _compute_ambiguity_level(vfc, vfs, rejected)
                conf       = _compute_confidence(vf, rejected)
                thresh     = _compute_dynamic_threshold(vfc, vfs)
                risk_facts = _compute_risk_factors(vfc, vfs, rejected)

                result["calibrated_score"]  = calibrated
                result["penalty_breakdown"] = breakdown
                result["threshold"]         = thresh
                result["ambiguity_level"]   = amb
                result["risk_factors"]      = risk_facts
                result["semantic_signals"]  = {
                    k: v.get("value", 0.0)
                    for k, v in vfc.items() if isinstance(v, dict)
                }
                # Explainability fields (decision-independent — built here)
                result["criterion_breakdown"]     = _build_score_breakdown(vfc)
                result["ambiguity_explanation"]   = _build_ambiguity_explanation(amb, rejected)
                result["confidence_explanation"]  = _build_confidence_explanation(conf, amb, rejected)
                result["risk_explanation"]        = _build_risk_explanation(risk_facts)

                n_conf = sum(
                    1 for v in vf["green_criteria"].values()
                    if isinstance(v, dict) and v.get("value", 0) >= 0.5
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
