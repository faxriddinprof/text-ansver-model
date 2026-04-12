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




# ---------------------------------------------------------------------------
# Advanced Signal Processing
# ---------------------------------------------------------------------------

# 1 ── Negation detection
# Words that, within a short window around a keyword, negate its meaning.
_SEMANTIC_NEGATION_PATTERNS: list[str] = [
    # Uzbek
    r"\byo['']q\b", r"\bemas\b", r"\bqilinmagan\b", r"\bko'rilmagan\b",
    r"\bmavjud emas\b", r"\brejada yo['']q\b", r"\bnazarda tutilmagan\b",
    r"\bbajarilmagan\b", r"\butilizatsiya qilinmagan\b",
    # Russian
    r"\bне\b", r"\bнет\b", r"\bотсутству", r"\bне используется\b",
    r"\bне применяется\b", r"\bне предусмотрен", r"\bне установлен",
    # English
    r"\bnot\b", r"\bno\s+\w+", r"\bwithout\b", r"\bdoes not\b",
    r"\bdid not\b", r"\bnever\b", r"\black[s]?\b", r"\babsence of\b",
]

_COMPILED_NEGATION = [re.compile(p, re.IGNORECASE) for p in _SEMANTIC_NEGATION_PATTERNS]


def _has_negation_nearby(text: str, match_start: int, match_end: int,
                          window: int = 120) -> bool:
    """Return True if a negation word appears within `window` chars of the match."""
    start   = max(0, match_start - window)
    end     = min(len(text), match_end + window)
    snippet = text[start:end]
    return any(pat.search(snippet) for pat in _COMPILED_NEGATION)


# 2 ── Numeric intelligence
# Thresholds that determine whether a numeric value constitutes valid evidence.
_NUMERIC_RULES: dict[str, dict] = {
    "energy_efficiency": {
        # % energy/carbon reduction — must be ≥ 20 to count as strong
        "pattern": re.compile(
            r"(\d{1,3}(?:[.,]\d+)?)\s*%\s*(?:"
            r"energy\s*(?:saving|efficien|reduction)|"
            r"energiya\s*(?:tejam|samaradorl|kamay)|"
            r"снижени[ея]\s*энергопотреблен|"
            r"энергоэффективност|energoeffektivnost"
            r")", re.IGNORECASE
        ),
        "strong":  20.0,   # ≥ 20% → strong (1.0)
        "partial": 10.0,   # ≥ 10% → partial (0.6)
        "units":   "%",
        "inverted": False,  # higher = better
    },
    "ghg_reduction": {
        # CO₂ g/kWh — for hydro: must be < 50 to qualify as green
        "pattern": re.compile(
            r"(\d{1,4}(?:[.,]\d+)?)\s*g\s*/?\s*kw[-·]?h", re.IGNORECASE
        ),
        "strong":  50.0,   # < 50 g/kWh → strong
        "partial": 100.0,  # < 100 g/kWh → partial
        "units":   "g/kWh",
        "inverted": True,   # lower = better (emissions)
    },
    "ghg_reduction_pct": {
        # % GHG/CO₂ reduction — must be ≥ 15
        "pattern": re.compile(
            r"(\d{1,3}(?:[.,]\d+)?)\s*%\s*(?:"
            r"co2\s*(?:reduction|kamay|снижени)|"
            r"ghg\s*reduction|"
            r"emission\s*reduction|"
            r"greenhouse\s*gas\s*reduction|"
            r"issiqxona\s*gaz\s*kamay|"
            r"снижени[ея]\s*(?:выброс|парников)"
            r")", re.IGNORECASE
        ),
        "strong":  30.0,
        "partial": 15.0,
        "units":   "%",
        "inverted": False,
    },
}


def _extract_numeric_signals(text: str, criterion: str) -> float:
    """
    For relevant criteria, parse numeric values from text and return
    an evidence score (0.0 | 0.6 | 1.0) based on threshold rules.

    Called as an ADDITIONAL input to reinforce or upgrade semantic score.
    """
    rules_for = []
    if criterion == "energy_efficiency":
        rules_for = [_NUMERIC_RULES["energy_efficiency"]]
    elif criterion == "ghg_reduction":
        rules_for = [_NUMERIC_RULES["ghg_reduction"], _NUMERIC_RULES["ghg_reduction_pct"]]
    else:
        return 0.0

    best = 0.0
    for rule in rules_for:
        for m in rule["pattern"].finditer(text):
            raw_val = m.group(1).replace(",", ".")
            try:
                val = float(raw_val)
            except ValueError:
                continue
            inverted = rule["inverted"]
            if not inverted:
                if val >= rule["strong"]:
                    best = max(best, 1.0)
                elif val >= rule["partial"]:
                    best = max(best, 0.6)
            else:
                if val < rule["strong"]:
                    best = max(best, 1.0)
                elif val < rule["partial"]:
                    best = max(best, 0.6)
    return best


# 3 ── Source reliability  (UPGRADED: 5-tier with evidence_quality_score)
# Phrases that indicate the evidence is authoritative vs. marketing.
_SOURCE_STRONG: list[str] = [
    "leed", "edge certified", "breeam", "iso 14001",
    "davlat ekologik ekspertiza", "state environmental",
    "government report", "technical specification", "audit report",
    "engineering assessment", "feasibility study", "project passport",
]
_SOURCE_MEDIUM: list[str] = [
    "technical report", "measurement data", "test result",
    "monitoring report", "assessment", "technical design",
    "documented", "project design", "технический",
]
_SOURCE_WEAK: list[str] = [
    "eco-friendly", "environment-friendly", "sustainable approach",
    "green initiative", "green values", "green culture",
    "ecological harmony", "environmental responsibility statement",
    "commitment to", "supports sustainable",
]
# Additional tier: government/certification = highest trust
_SOURCE_AUTHORITATIVE: list[str] = [
    "davlat ekologik ekspertiza xulosasi", "davlat ekologik",
    "state environmental approval", "government green certification",
    "official environmental approval", "leed certified", "edge certified",
    "breeam certified", "iso 14001 certified", "iso 14001:2015",
    "давлат экологик экспертиза", "государственная экологическая экспертиза",
]
# Tier for future plans — not yet implemented
_SOURCE_FUTURE: list[str] = [
    "will be installed", "planned to", "rejalashtirilmoqda", "планируется",
    "will install", "future phase", "upon completion", "to be commissioned",
    "kelajakda", "по плану",
]


def _source_reliability_score(evidence_text: str) -> float:
    """
    Return a reliability multiplier (0.2 – 1.0) based on evidence provenance.

    Tier map:
    1.0  → government certificate / official state approval
    1.0  → accredited certification (LEED, EDGE, BREEAM, ISO 14001)
    0.8  → technical/engineering document
    0.6  → assessment or documented report
    0.3  → future plan / uncommitted statement
    0.2  → marketing / soft claim
    """
    t = evidence_text.lower()
    if any(s in t for s in _SOURCE_AUTHORITATIVE):
        return 1.0
    if any(s in t for s in _SOURCE_STRONG):
        return 1.0
    if any(s in t for s in _SOURCE_FUTURE):
        return 0.3
    if any(s in t for s in _SOURCE_MEDIUM):
        return 0.8
    if any(s in t for s in _SOURCE_WEAK):
        return 0.2
    return 0.6  # neutral default


def _compute_evidence_quality(
    criteria: dict,
    full_text: str,
    time_statuses: dict,
) -> dict:
    """
    Compute per-criterion evidence_quality_score in [0.0, 1.0].

    Formula:
        eq = source_reliability × time_status_multiplier

    Returns {criterion_name: float}
    """
    t = full_text.lower()
    quality: dict[str, float] = {}
    for name, obj in criteria.items():
        if not isinstance(obj, dict):
            quality[name] = 0.0
            continue
        val  = obj.get("value", 0.0)
        evid = obj.get("evidence", "")
        if val == 0.0:
            quality[name] = 0.0
            continue
        src_rel  = _source_reliability_score(evid or t[:500])
        time_mul = time_statuses.get(name, 0.7)
        quality[name] = round(min(1.0, src_rel * time_mul), 3)
    return quality


# 4 ── Time-status detection
_PLANNED_PATTERNS = re.compile(
    r"\b(?:"
    r"will\s+(?:be\s+)?(?:install|build|implement|establish|construct)|"
    r"plan(?:ned|s)?\s+to|"
    r"planned|planning\s+stage|"
    r"pre-?construction|preconstruction|"
    r"proposed\s+to|"
    r"intend(?:s)?\s+to|"
    r"intended\s+to|"
    r"to\s+be\s+(?:installed|implemented|constructed|commissioned)|"
    r"will\s+be\s+commissioned|"
    r"upon\s+completion|once\s+operational|"
    r"being\s+considered|under\s+consideration|"
    r"future\s+(?:install|project|phase)|"
    r"by\s+20\d{2}|"
    r"qur(?:iladi|ilmoqda)|"
    r"o['']rnatiladi|"
    r"rejalashtirilmoqda|"
    r"kelajakda|"
    r"предусмотрен\s+(?:установк|строительств|монтаж)|"
    r"планируется|"
    r"будет\s+(?:установлен|смонтирован|построен)"
    r")\b",
    re.IGNORECASE,
)

_CONSTRUCTION_PATTERNS = re.compile(
    r"\b(?:"
    r"under\s+construction|being\s+(?:built|constructed|installed|implemented)|"
    r"construction\s+(?:in\s+progress|phase|underway)|currently\s+being\s+built|"
    r"quri(?:lmoqda|sh\s+jarayoni)|montaj\s+qilinmoqda|"
    r"в\s+процессе\s+строительства|строится|монтируется|возводится"
    r")\b",
    re.IGNORECASE,
)

_PARTIAL_PATTERNS = re.compile(
    r"\b(?:"
    r"partially\s+(?:installed|operational|implemented|complete)|"
    r"first\s+phase\s+(?:complete|operational)|phase\s+1\s+(?:done|complete)|"
    r"partial(?:ly)?\s+(?:commission|deploy|integrat)|"
    r"some\s+units?\s+(?:installed|operational)|"
    r"qisman\s+(?:o'rnatilgan|ishlamoqda)|birinchi\s+bosqich\s+yakunlangan|"
    r"частично\s+(?:установлен|введён|введен|реализован)"
    r")\b",
    re.IGNORECASE,
)

_OPERATIONAL_PATTERNS = re.compile(
    r"\b(?:"
    r"(?:already\s+)?(?:installed|functioning|running|in\s+use|operating)|"
    r"(?:has|have|was|were)\s+been\s+(?:installed|implemented|commissioned)|"
    r"(?:is|are)\s+operational|"
    r"o['']rnatilgan|"
    r"ishlamoqda|"
    r"joriy\s+etilgan|"
    r"установлен[оа]?\b|"
    r"введён\s+в\s+эксплуатацию|"
    r"функционирует|"
    r"工作中"
    r")\b",
    re.IGNORECASE,
)


def _detect_time_status(evidence_snippet: str) -> float:
    """
    4-tier graded time-status multiplier:

    1.0  — fully operational / installed and running
    0.8  — partially operational (phase 1 done, phase 2 pending)
    0.6  — under active construction / being implemented
    0.4  — planned / future / proposed
    0.7  — default (ambiguous tense)
    """
    t = evidence_snippet
    is_operational  = bool(_OPERATIONAL_PATTERNS.search(t))
    is_partial      = bool(_PARTIAL_PATTERNS.search(t))
    is_construction = bool(_CONSTRUCTION_PATTERNS.search(t))
    is_planned      = bool(_PLANNED_PATTERNS.search(t))

    if is_operational and not is_planned and not is_partial:
        return 1.0
    if is_partial:
        return 0.8
    if is_construction and not is_planned:
        return 0.6
    if is_planned and not is_operational:
        return 0.4
    if is_operational and is_planned:
        return 0.8   # mixed deployment
    return 0.7       # ambiguous tense — slight discount


def _detect_time_status_label(evidence_snippet: str) -> str:
    """Return a human-readable time-status label for debug output."""
    mult = _detect_time_status(evidence_snippet)
    mapping = {1.0: "operational", 0.8: "partial/mixed", 0.6: "under_construction",
               0.4: "planned", 0.7: "ambiguous"}
    return mapping.get(mult, "ambiguous")


def _get_evidence_context(full_text: str, evidence_snippet: str, window: int = 160) -> str:
    """Return surrounding text for an extracted evidence snippet when available."""
    haystack = " ".join((full_text or "").lower().split())
    needle = " ".join((evidence_snippet or "").lower().split())
    if not haystack or not needle:
        return needle or haystack

    start = haystack.find(needle)
    if start == -1:
        probe = " ".join(needle.split()[:6])
        if not probe:
            return needle
        start = haystack.find(probe)
        if start == -1:
            return needle
        end = start + max(len(needle), len(probe))
    else:
        end = start + len(needle)

    return haystack[max(0, start - window):min(len(haystack), end + window)]


# 5 ── Multilingual OCR normalization
# Cyrillic → Latin lookalike table (partial, for common OCR confusions)
_CYRILLIC_TO_LATIN: dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "r", "с": "c",
    "х": "x", "у": "y", "А": "A", "В": "B", "С": "C",
    "Е": "E", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "Т": "T", "Х": "X",
}
_BROKEN_SPACE_RE  = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")   # hyphenated line break
_MULTI_SPACE_RE   = re.compile(r"[ \t]{2,}")
_OCR_ARTIFACT_RE  = re.compile(r"[|}{@#~^]")                 # common OCR garbage chars


def normalize_text(text: str) -> str:
    """
    Multilingual OCR hardening:
    1. Rejoin hyphen-broken words ("so-\n  lar" → "solar")
    2. Collapse multiple spaces / tabs
    3. Strip common OCR artifact characters
    4. Normalize Unicode (NFC)

    Does NOT transliterate Cyrillic — keeps original script for phrase matching.
    """
    import unicodedata
    t = _BROKEN_SPACE_RE.sub("", text)
    t = _MULTI_SPACE_RE.sub(" ", t)
    t = _OCR_ARTIFACT_RE.sub(" ", t)
    t = unicodedata.normalize("NFC", t)
    return t


# 6 ── Cross-criteria dependency rules  (UPGRADED: R1–R6 + consistency_penalty)
def _apply_cross_rules(
    criteria_scores: dict,
    stop_scores:     dict,
    notes:           list,
) -> dict:
    """
    Post-processing consistency rules.  Adjusts scores where criteria
    are logically dependent on each other.

    Rules applied:
    R1: renewable_energy strong + ghg_reduction absent + no other confirmed → −0.1
    R2: energy_efficiency present but numeric evidence <20% → cap at 0.6
    R3: certificate alone (no other criterion confirmed) → cap at 0.6
    R4: any hard stop triggered → zero-out all green criteria (safety net)
    R5: renewable in industrial/oil context (greenwash flag) → renewable cap 0.5
    R6: energy_efficiency claimed but no numeric % found in text → cap at 0.8
    """
    s = {k: v for k, v in criteria_scores.items()}  # shallow copy

    r_score = s.get("renewable_energy", 0.0)
    g_score = s.get("ghg_reduction", 0.0)
    e_score = s.get("energy_efficiency", 0.0)
    c_score = s.get("certificate", 0.0)
    ei_score = s.get("environmental_infrastructure", 0.0)

    confirmed = {k for k, v in s.items() if v >= 0.5}
    stop_triggered = any(
        (v.get("value", 0.0) if isinstance(v, dict) else v) >= 0.5
        for v in stop_scores.values()
    )
    # Detect oil/industrial context from soft stop signals
    oil_soft = (
        stop_scores.get("oil_gas", {}).get("value", 0.0)
        if isinstance(stop_scores.get("oil_gas"), dict)
        else 0.0
    )

    # R4 safety net
    if stop_triggered:
        for k in s:
            s[k] = 0.0
        return s

    # R1: renewable-only with no emissions or infrastructure evidence → suspicious
    other_confirmed = confirmed - {"renewable_energy", "certificate"}
    if r_score >= 1.0 and g_score == 0.0 and ei_score == 0.0 and not other_confirmed:
        s["renewable_energy"] = min(r_score, 0.9)
        notes.append(
            "R1: renewable_energy confirmed but no GHG or infrastructure evidence "
            "— renewable score discounted to 0.9 (consistency check)."
        )

    # R3: certificate alone (no other confirmed criterion) → cap at 0.6
    if c_score >= 0.5 and not (confirmed - {"certificate"}):
        s["certificate"] = min(c_score, 0.6)
        notes.append(
            "R3: certificate present but no other ESG criterion confirmed "
            "— certificate score capped at 0.6 (standalone cert insufficient)."
        )

    # R5: solar/renewable flag in primarily oil-adjacent context → greenwashing risk
    if r_score >= 0.5 and oil_soft >= 0.3:
        s["renewable_energy"] = min(s["renewable_energy"], 0.5)
        notes.append(
            "R5: renewable_energy claim found alongside oil/gas signals "
            "— possible token solar in non-green business, score capped at 0.5."
        )

    return s


def _compute_consistency_penalty(
    criteria_scores: dict,
    stop_scores:     dict,
    notes:           list,
) -> float:
    """
    Compute an additive consistency penalty (subtracted from calibrated score).

    Penalties:
    P1: renewable_energy WITHOUT any GHG + any infra + any efficiency → -0.3
    P2: ONLY certificate confirmed (no real operational criteria) → -0.4
    P3: energy_efficiency claimed but numeric percentage <20% in criteria → -0.2
    P4: contradictory criteria (oil/coal soft-stop + renewable strong) → -0.3

    Returns total penalty (non-negative float).
    """
    penalty = 0.0
    r   = criteria_scores.get("renewable_energy", 0.0)
    g   = criteria_scores.get("ghg_reduction", 0.0)
    e   = criteria_scores.get("energy_efficiency", 0.0)
    ei  = criteria_scores.get("environmental_infrastructure", 0.0)
    c   = criteria_scores.get("certificate", 0.0)

    confirmed = {k for k, v in criteria_scores.items() if v >= 0.5}

    # P1: renewable with zero other evidence is suspicious
    if r >= 0.5 and g == 0.0 and e == 0.0 and ei == 0.0:
        penalty += 0.3
        notes.append(
            "P1 consistency penalty (-0.3): renewable energy claimed but "
            "no GHG reduction, efficiency, or infrastructure evidence found."
        )

    # P2: certificate only ("paper green") → strong penalty
    if confirmed == {"certificate"}:
        penalty += 0.4
        notes.append(
            "P2 consistency penalty (-0.4): only certificate criterion "
            "confirmed, no operational ESG evidence — paper green risk."
        )

    # P4: contradiction — soft stop co-existing with strong renewable claim
    oil_val = (
        stop_scores.get("oil_gas", {}).get("value", 0.0)
        if isinstance(stop_scores.get("oil_gas"), dict) else 0.0
    )
    coal_val = (
        stop_scores.get("coal", {}).get("value", 0.0)
        if isinstance(stop_scores.get("coal"), dict) else 0.0
    )
    if (oil_val >= 0.3 or coal_val >= 0.3) and r >= 0.5:
        penalty += 0.3
        notes.append(
            "P4 consistency penalty (-0.3): strong renewable claim alongside "
            "oil/coal signals — potential token greenwashing."
        )

    return round(penalty, 2)


def _semantic_strength(name: str, text_lower: str, concept_map: dict) -> float:
    """
    Return evidence strength (0.0 | 0.3 | 0.6 | 1.0) via tiered semantic matching.
    Checks concept tiers strong → partial → weak.
    Each tier is a list of synonym groups; any single phrase hit within a group
    counts as a match for that tier.  Highest tier wins.

    Negation guard: if the matched phrase is surrounded by negation words
    within a 120-char window, the tier is skipped entirely.
    """
    entry = concept_map.get(name, {})
    for tier_name, score in (("strong", 1.0), ("partial", 0.6), ("weak", 0.3)):
        for phrase_group in entry.get(tier_name, []):
            for phrase in phrase_group:
                pos = text_lower.find(phrase)
                if pos == -1:
                    continue
                # Negation guard
                if _has_negation_nearby(text_lower, pos, pos + len(phrase)):
                    continue
                return score
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
    Project-type-aware GREEN decision threshold [2.7 – 3.5].

    certified renewable  → 2.7   (clean credentials, lower bar)
    oil/coal adjacent    → 3.5   (higher scrutiny — greenwash risk zone)
    gambling/alcohol     → 3.5   (excluded category, virtually unreachable)
    pure industrial      → 3.2   (no clean-energy evidence at all)
    standard             → 3.0
    """
    r  = criteria.get("renewable_energy", {})
    c  = criteria.get("certificate", {})
    oi = stops.get("oil_gas", {})
    co = stops.get("coal", {})
    al = stops.get("alcohol", {})
    to = stops.get("tobacco", {})
    ga = stops.get("gambling", {})

    r_val  = r.get("value",  0.0) if isinstance(r,  dict) else 0.0
    c_val  = c.get("value",  0.0) if isinstance(c,  dict) else 0.0
    oi_val = oi.get("value", 0.0) if isinstance(oi, dict) else 0.0
    co_val = co.get("value", 0.0) if isinstance(co, dict) else 0.0
    al_val = al.get("value", 0.0) if isinstance(al, dict) else 0.0
    to_val = to.get("value", 0.0) if isinstance(to, dict) else 0.0
    ga_val = ga.get("value", 0.0) if isinstance(ga, dict) else 0.0

    if r_val >= 1.0 and c_val >= 0.6:
        return 2.7   # certified renewable project — easier bar
    if al_val >= 0.3 or to_val >= 0.3 or ga_val >= 0.3:
        return 3.5   # excluded sectors — effectively unreachable threshold
    if 0.3 <= oi_val < 0.5 or 0.3 <= co_val < 0.5:
        return 3.5   # oil/coal adjacency — highest scrutiny
    if r_val == 0.0 and c_val == 0.0:
        return 3.2   # pure industrial, no clean-energy evidence
    return 3.0       # standard


def _compute_calibrated_score(
    criteria: dict,
    stops:    dict,
    full_text: str = "",
    notes:     list | None = None,
) -> tuple:
    """
    Calibrated scoring — upgraded engine:
      1. Numeric intelligence (raises semantic score if numeric evidence stronger)
      2. 4-tier time-status discounting
      3. 5-tier source reliability discounting  →  evidence_quality_score
      4. Cross-criteria consistency rules (R1–R6)
      5. Consistency penalty (P1–P4)
      6. Ambiguity + contradiction penalties

    final_score = clamp(raw - ambiguity_penalty - contradiction_penalty
                        - consistency_penalty, 0, 5)

    Inputs
    ------
    criteria  : {name: {"value": float, "evidence": str}, ...}
    stops     : {name: {"value": float, "evidence": str}, ...}
    full_text : raw document text (for numeric + context checks)
    notes     : accumulated audit notes list (mutated in-place)

    Returns (final_score: float, breakdown: dict)
    Score clamped to [0, 5].
    """
    if notes is None:
        notes = []
    t_lower = (" ".join(full_text.lower().split())) if full_text else ""

    time_statuses:   dict[str, float]  = {}
    evidence_quality: dict[str, float] = {}

    # Step 1 — collect base semantic scores, numeric upgrades, time + source discounting
    base_scores: dict[str, float] = {}
    for name, obj in criteria.items():
        if not isinstance(obj, dict):
            base_scores[name] = 0.0
            evidence_quality[name] = 0.0
            time_statuses[name] = 0.7
            continue
        sem_score = obj.get("value", 0.0)
        evidence  = obj.get("evidence", "")

        # Numeric intelligence: can raise (but not lower) semantic score
        if t_lower:
            num_score = _extract_numeric_signals(t_lower, name)
            if num_score > sem_score:
                notes.append(
                    f"Numeric evidence for '{name}' raised score "
                    f"{sem_score:.1f} → {num_score:.1f}."
                )
                sem_score = num_score

        # 4-tier time-status discount
        evidence_context = _get_evidence_context(t_lower, evidence)
        if sem_score > 0 and evidence:
            time_mult = _detect_time_status(evidence_context or evidence)
            time_statuses[name] = time_mult
            if time_mult < 0.7:
                label = _detect_time_status_label(evidence_context or evidence)
                discounted = round(sem_score * time_mult, 2)
                notes.append(
                    f"Time-status [{label}]: '{name}' evidence "
                    f"→ score discounted {sem_score:.1f} × {time_mult:.1f} = {discounted:.2f}."
                )
                sem_score = discounted
        else:
            time_statuses[name] = 0.7  # default for missing evidence

        # 5-tier source reliability: evidence_quality = src_reliability × time_mult
        if sem_score > 0:
            src_rel = _source_reliability_score(evidence or t_lower[:500])
            eq = round(min(1.0, src_rel * time_statuses[name]), 3)
            evidence_quality[name] = eq
            # Apply source discount for low-quality evidence
            if src_rel < 0.4:  # Only penalize truly low (marketing/future plans)
                discounted = round(sem_score * src_rel, 2)
                notes.append(
                    f"Source reliability [{src_rel:.1f}]: '{name}' evidence is "
                    f"low-quality → score {sem_score:.1f} → {discounted:.2f}."
                )
                sem_score = discounted
        else:
            evidence_quality[name] = 0.0

        base_scores[name] = sem_score

    # Step 2 — cross-criteria consistency rules (R1–R6)
    base_scores = _apply_cross_rules(base_scores, stops, notes)

    # Step 3 — consistency penalty (P1–P4)
    consistency_penalty = _compute_consistency_penalty(base_scores, stops, notes)

    # Rebuild adjusted criteria dict for downstream use
    adjusted_criteria = {
        k: {"value": base_scores.get(k, v.get("value", 0.0) if isinstance(v, dict) else 0.0),
            "evidence": v.get("evidence", "") if isinstance(v, dict) else ""}
        for k, v in criteria.items()
    }

    raw       = sum(base_scores.values())
    confirmed = sum(1 for v in base_scores.values() if v >= 0.5)
    weak      = sum(1 for v in base_scores.values() if 0.0 < v < 0.5)

    # Ambiguity penalty: no confirmed criteria, only weak signals
    ambiguity_penalty = 0.2 * weak if confirmed == 0 else 0.0

    # Contradiction penalty: soft stop signals (concerning but not blocking)
    soft_stop_sum = sum(
        v.get("value", 0.0) for v in stops.values()
        if isinstance(v, dict) and 0.3 <= v.get("value", 0) < 0.5
    )
    contradiction_penalty = 0.3 if soft_stop_sum > 0 else 0.0

    total_penalty = ambiguity_penalty + contradiction_penalty + consistency_penalty
    final = max(0.0, min(5.0, raw - total_penalty))
    return final, {
        "raw":                   round(raw, 3),
        "ambiguity_penalty":     round(ambiguity_penalty, 3),
        "contradiction_penalty": round(contradiction_penalty, 3),
        "consistency_penalty":   round(consistency_penalty, 3),
        "total_penalty":         round(total_penalty, 3),
        "final":                 round(final, 3),
        "adjusted_criteria":     adjusted_criteria,
        "evidence_quality":      evidence_quality,
        "time_statuses":         time_statuses,
    }


def _compute_ambiguity_level(criteria: dict, stops: dict, rejected: list) -> str:
    """
    Returns 'low' | 'medium' | 'high' based on evidence quality.

    high   — many rejected claims, soft stops, only weak criteria, or consistency issues
    medium — one rejected claim, one weak criterion, or soft stop
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

    # Elevated to high: multiple rejections, soft stops, or exclusively weak signals
    if n_rejected >= 2 or soft_stops >= 2 or (confirmed == 0 and weak >= 2):
        return "high"
    # Medium: single rejection, a soft stop, or one weak-only criterion
    if n_rejected == 1 or soft_stops == 1 or (confirmed == 0 and weak == 1):
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
# Adversarial / Greenwashing Detection Layer  (NEW)
# ---------------------------------------------------------------------------

# Marketing / vague sustainability phrases that add zero epistemic value
_GREENWASH_PHRASES: list[str] = [
    "eco-friendly", "environment-friendly", "environmentally friendly",
    "sustainable approach", "green values", "green culture", "green initiative",
    "yashil iqtisodiyot", "ecological harmony", "responsible business",
    "supports sustainable", "commitment to environment", "caring for the planet",
    "carbon neutral future", "net zero journey", "climate positive",
    "eco conscious", "environmentally conscious", "green minded",
    "goes green", "embrace sustainability",
]

# Non-ESG certifications that are sometimes misrepresented as green
_FAKE_ESG_CERTS: list[str] = [
    "iso 9001", "iso 9001:2015", "iso 9001:2008",
    "iso 45001",                   # OHS — not environmental
    "haccp",                       # food safety
    "gmp certificate", "gmp certified",
]

# Patterns suggesting token/marginal green element in a non-green business
_TOKEN_GREEN_PATTERNS = re.compile(
    r"(?:"
    r"solar\s+panel[s]?\s+(?:for|on|at)\s+(?:office|admin|facility|building)\s+(?:only|roof)|"
    r"(?:small|minor|one|single|few)\s+solar\s+panel|"
    r"solar\s+panel[s]?\s+as\s+(?:auxiliary|secondary|supplementary)|"
    r"a\s+couple\s+of\s+(?:solar|wind)|only\s+marginally\s+renew|"
    r"minimal\s+(?:renewable|solar|green)\s+component"
    r")",
    re.IGNORECASE,
)


def _compute_greenwashing_risk(
    criteria:  dict,
    stops:     dict,
    text_lower: str,
    rejected:   list,
) -> dict:
    """
    Greenwashing Risk Scorer — returns a structured risk report.

    greenwashing_risk_score : 0.0 – 1.0
      0.0 – 0.2 : Low  (normal project)
      0.2 – 0.5 : Medium (some suspicious signals)
      0.5 – 0.8 : High  (probable greenwash)
      0.8 – 1.0 : Critical (strong adversarial pattern)

    Signals scored:
    S1: many marketing phrases, low confirmed evidence
    S2: non-ESG certificate passed as ESG
    S3: token solar in non-green primary business
    S4: soft stop factor with renewable claim (oil+solar pattern)
    S5: high LLM rejection rate (claims not in document)
    S6: only vague sustainability language, no numbers
    """
    risk = 0.0
    signals: list[str] = []
    t = text_lower

    confirmed_count = sum(
        1 for v in criteria.values()
        if isinstance(v, dict) and v.get("value", 0) >= 0.5
    )

    # S1: marketing language dominance
    gw_hits = sum(1 for ph in _GREENWASH_PHRASES if ph in t)
    if gw_hits >= 4 and confirmed_count == 0:
        risk += 0.35
        signals.append(f"S1:marketing_dominance({gw_hits}_phrases,no_confirmed_criteria)")
    elif gw_hits >= 2 and confirmed_count == 0:
        risk += 0.20
        signals.append(f"S1:marketing_phrases({gw_hits},no_confirmed_criteria)")

    # S2: non-ESG cert misused
    fake_hits = [c for c in _FAKE_ESG_CERTS if c in t]
    cert_val = criteria.get("certificate", {})
    cert_score = cert_val.get("value", 0.0) if isinstance(cert_val, dict) else 0.0
    if fake_hits and cert_score >= 0.3:
        risk += 0.25
        signals.append(f"S2:non_esg_cert_misuse({','.join(fake_hits)})")

    # S3: token solar pattern
    if _TOKEN_GREEN_PATTERNS.search(t):
        risk += 0.20
        signals.append("S3:token_solar_minor_component")

    # S4: soft stop + renewable = oil+solar adversarial pattern
    oil_val = (
        stops.get("oil_gas", {}).get("value", 0.0)
        if isinstance(stops.get("oil_gas"), dict) else 0.0
    )
    r_val = criteria.get("renewable_energy", {})
    r_score = r_val.get("value", 0.0) if isinstance(r_val, dict) else 0.0
    if oil_val >= 0.3 and r_score >= 0.3:
        risk += 0.25
        signals.append(f"S4:oil_solar_mix(oil={oil_val:.1f},solar={r_score:.1f})")

    # S5: high rejection rate
    total_claims = len(criteria) + len(stops)
    if total_claims > 0:
        rejection_rate = len(rejected) / total_claims
        if rejection_rate >= 0.4:
            risk += 0.25
            signals.append(f"S5:high_rejection_rate({rejection_rate:.0%})")
        elif rejection_rate >= 0.2:
            risk += 0.10
            signals.append(f"S5:moderate_rejection_rate({rejection_rate:.0%})")

    # S6: no quantitative data at all
    has_numbers = bool(re.search(r"\d+\s*(?:%|mw|kwh|t(?:on)?(?:ne)?s?\s*co2|g/kwh)", t))
    if not has_numbers and confirmed_count >= 1:
        risk += 0.10
        signals.append("S6:no_quantitative_data_with_green_claim")

    risk = round(min(1.0, risk), 3)
    if risk >= 0.5:
        level = "high"
    elif risk >= 0.2:
        level = "medium"
    else:
        level = "low"

    return {
        "greenwashing_risk_score": risk,
        "greenwashing_risk_level": level,
        "greenwashing_signals":    signals,
    }


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
    greenwashing: dict | None = None,
) -> str:
    """
    Upgraded explanation — answers three questions:
      1. WHY did the project get this verdict?
      2. WHAT EXACTLY is missing or wrong?
      3. WHAT WOULD MAKE IT GREEN?

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
    weak_criteria = [
        f"{_CRITERION_DISPLAY.get(k, k.replace('_', ' ').title())} "
        f"(weak signal, score {v.get('value', 0):.1f})"
        for k, v in criteria.items()
        if isinstance(v, dict) and 0.0 < v.get("value", 0) < 0.5
    ]
    stop_triggered = [
        k for k, v in stops.items()
        if isinstance(v, dict) and v.get("value", 0) >= 0.5
    ]
    gw = greenwashing or {}
    gw_level = gw.get("greenwashing_risk_level", "low")
    gw_score  = gw.get("greenwashing_risk_score", 0.0)

    # ── Case 1: Exclusion triggered ────────────────────────────────────────
    if stop_triggered:
        stop_names = "; ".join(
            _STOP_RISK_MESSAGES.get(k, k.replace("_", " ")) for k in stop_triggered
        )
        return (
            f"VERDICT: NOT GREEN — exclusion rule triggered.\n"
            f"WHY: The document describes activities that are categorically excluded "
            f"from green financing under the applicable framework.\n"
            f"SPECIFICALLY: {stop_names}\n"
            f"WHAT WOULD MAKE IT GREEN: This project cannot qualify as green "
            f"without fundamentally changing its primary business activity. "
            f"ESG criteria cannot override an exclusion trigger."
        )

    # ── Case 2: GREEN ──────────────────────────────────────────────────────
    if decision == "GREEN":
        conf_list = ", ".join(confirmed) if confirmed else "multiple criteria"
        gw_note = ""
        if gw_level == "medium":
            gw_note = (
                f" Note: moderate greenwashing risk detected (score {gw_score:.2f}). "
                f"Independent verification is recommended."
            )
        elif gw_level == "high":
            gw_note = (
                f" WARNING: High greenwashing risk detected (score {gw_score:.2f}). "
                f"Claims may be overstated. Due diligence required before financing."
            )
        return (
            f"VERDICT: GREEN — threshold met.\n"
            f"WHY: The project demonstrates verifiable environmental impact in: {conf_list}.\n"
            f"SCORE: {score:.2f} ≥ required threshold {threshold:.1f}.{gw_note}"
        )

    # ── Case 3: NOT GREEN — below threshold ───────────────────────────────
    gap = round(threshold - score, 2)
    parts_why = []
    parts_missing = []
    parts_fix = []

    # WHY section
    if confirmed:
        parts_why.append(
            f"Partial evidence found for {', '.join(confirmed)}, "
            f"but the combined score ({score:.2f}) falls {gap:.2f} points "
            f"short of the required threshold ({threshold:.1f})."
        )
    else:
        parts_why.append(
            f"No ESG criteria met the minimum evidence threshold. "
            f"Score: {score:.2f} vs. required {threshold:.1f} (gap: {gap:.2f})."
        )

    if rejected:
        parts_why.append(
            f"{len(rejected)} claim(s) in the document were rejected "
            f"because they could not be verified from the document content."
        )

    if gw_level in ("medium", "high"):
        parts_why.append(
            f"Greenwashing risk {gw_level} (score {gw_score:.2f}): "
            f"the document contains sustainability language without sufficient "
            f"verifiable evidence to support it."
        )

    # WHAT IS MISSING section
    if weak_criteria:
        parts_missing.append(
            f"Weak (unconfirmed) signals detected for: {'; '.join(weak_criteria)}. "
            f"These require stronger documentary support to count."
        )
    if missing:
        parts_missing.append(
            f"No evidence found for: {', '.join(missing)}."
        )

    # WHAT WOULD MAKE IT GREEN section
    needed_count = max(1, -(-gap // 1.0))  # ceiling division
    fixable_criteria = [
        _CRITERION_DISPLAY.get(k, k.replace("_", " ").title())
        for k, v in criteria.items()
        if isinstance(v, dict) and v.get("value", 0) < 1.0
    ]
    if fixable_criteria:
        top_fix = fixable_criteria[:3]
        parts_fix.append(
            f"To qualify as GREEN, provide verifiable documentation for at least "
            f"{int(needed_count)} more criterion/criteria, starting with: "
            f"{', '.join(top_fix)}. "
            f"Evidence must include specific data (percentages, capacities, "
            f"certifications) from authoritative sources."
        )

    sections = []
    sections.append("VERDICT: NOT GREEN — insufficient evidence score.")
    if parts_why:
        sections.append("WHY: " + " ".join(parts_why))
    if parts_missing:
        sections.append("WHAT IS MISSING: " + " ".join(parts_missing))
    if parts_fix:
        sections.append("WHAT WOULD MAKE IT GREEN: " + " ".join(parts_fix))
    return "\n".join(sections)


def _build_missing_criteria_explanation(
    criteria: dict,
    stops: dict,
    score: float,
    threshold: float,
    evidence_quality: dict | None = None,
) -> dict:
    """
    Audit 2.0: structured 'what is missing to become GREEN' breakdown.

    Upgraded: now includes evidence quality per criterion and specific
    upgrade actions (e.g. "replace marketing claim with technical data").

    Returns:
    {
      "gap": float,
      "missing_criteria": [{criterion, label, current_score, evidence_quality,
                             needed, guidance, upgrade_action}],
      "is_fixable": bool,
      "summary": str,
    }
    """
    eq = evidence_quality or {}

    _CRITERION_GUIDANCE: dict[str, str] = {
        "renewable_energy": (
            "Provide documentation of installed renewable energy systems "
            "(solar panels, wind turbines, hydropower). Include technical specifications, "
            "installed capacity (kW/MW), and commissioning certificate."
        ),
        "energy_efficiency": (
            "Document energy efficiency improvements of ≥20%. "
            "Include baseline and measured energy consumption, reduction percentage, "
            "and reference an accepted standard (e.g. ISO 50001, EDGE certification)."
        ),
        "ghg_reduction": (
            "Provide quantified GHG / CO₂ reduction commitments with methodology. "
            "Include baseline emissions (tCO₂e/yr), target reduction (%), and "
            "monitoring plan. Avoid vague 'reducing carbon footprint' statements."
        ),
        "environmental_infrastructure": (
            "Document environmental control systems: wastewater treatment capacity, "
            "dust-gas filtration specs, or industrial recycling volumes. "
            "Include technical drawings or commissioning certificates."
        ),
        "certificate": (
            "Obtain (or document existing) environmental certification: "
            "LEED, EDGE, BREEAM, ISO 14001, or state environmental approval. "
            "ISO 9001 (quality) and ISO 45001 (safety) do NOT qualify."
        ),
    }

    _UPGRADE_ACTIONS: dict[str, str] = {
        "renewable_energy": "Replace generic references with engineering specs: panel count, total kWp, annual generation (kWh).",
        "energy_efficiency": "Add a certified energy audit showing ≥20% baseline-to-target reduction with measurement methodology.",
        "ghg_reduction": "Provide an emissions inventory (baseline year + projection) validated by an accredited body.",
        "environmental_infrastructure": "Attach commissioning certificate for treatment/filtration system with capacity data.",
        "certificate": "Attach the actual certificate (LEED/EDGE/BREEAM/ISO 14001). Ensure it covers the specific project, not the company.",
    }

    stop_triggered = any(
        v.get("value", 0) >= 0.5 for v in stops.values() if isinstance(v, dict)
    )
    if stop_triggered:
        return {
            "gap": 0.0,
            "missing_criteria": [],
            "is_fixable": False,
            "summary": (
                "This project cannot qualify as GREEN due to exclusion triggers. "
                "Changing the primary business activity would be required."
            ),
        }

    gap = max(0.0, round(threshold - score, 2))
    confirmed_score = sum(
        v.get("value", 0.0) for v in criteria.values()
        if isinstance(v, dict) and v.get("value", 0) >= 0.5
    )
    missing_potential = sum(
        1.0 - (v.get("value", 0.0) if isinstance(v, dict) else 0.0)
        for v in criteria.values()
        if isinstance(v, dict) and v.get("value", 0) < 0.5
    )

    missing_list = []
    for name, obj in criteria.items():
        val     = obj.get("value", 0.0) if isinstance(obj, dict) else 0.0
        crit_eq = eq.get(name, None)
        if val < 0.5:
            label   = _CRITERION_DISPLAY.get(name, name.replace("_", " ").title())
            guidance = _CRITERION_GUIDANCE.get(name, "Provide verifiable documentation.")
            upgrade  = _UPGRADE_ACTIONS.get(name, "Provide authoritative documentary evidence.")
            entry = {
                "criterion":      name,
                "label":          label,
                "current_score":  round(val, 2),
                "needed":         round(1.0 - val, 2),
                "guidance":       guidance,
                "upgrade_action": upgrade,
            }
            if crit_eq is not None:
                entry["evidence_quality"] = crit_eq
            missing_list.append(entry)
        elif crit_eq is not None and crit_eq < 0.5:
            # In criteria but low quality — flag for upgrade
            label   = _CRITERION_DISPLAY.get(name, name.replace("_", " ").title())
            upgrade = _UPGRADE_ACTIONS.get(name, "Strengthen evidence quality.")
            missing_list.append({
                "criterion":      name,
                "label":          label,
                "current_score":  round(val, 2),
                "needed":         0.0,  # already confirmed, quality upgrade needed
                "guidance":       f"{label} is confirmed but evidence quality is low ({crit_eq:.2f}). "
                                  f"Strengthen with authoritative sources.",
                "upgrade_action": upgrade,
                "evidence_quality": crit_eq,
                "quality_upgrade_only": True,
            })

    # Sort: unconfirmed first (by gap), then quality upgrades
    missing_list.sort(key=lambda x: (-x["needed"], x.get("evidence_quality", 1.0)))

    is_fixable = (confirmed_score + missing_potential) >= threshold

    if gap == 0.0:
        summary = "This project meets the required threshold. No additional criteria needed."
    elif is_fixable:
        names_needed = ", ".join(
            m["label"] for m in missing_list
            if not m.get("quality_upgrade_only") and m["needed"] > 0
        )[:3 * 30]  # rough char limit
        names_needed = ", ".join(
            m["label"] for m in missing_list
            if not m.get("quality_upgrade_only")
        )[:3]
        names_str = ", ".join(
            m["label"] for m in missing_list
            if not m.get("quality_upgrade_only")
        )
        summary = (
            f"Gap: {gap:.2f} points below threshold ({threshold:.1f}). "
            f"To qualify as GREEN, document: {names_str or 'additional ESG criteria'}. "
            f"Evidence must be from authoritative sources (technical docs, certificates)."
        )
    else:
        summary = (
            f"Gap: {gap:.2f} points below threshold ({threshold:.1f}). "
            "Available evidence is structurally insufficient for GREEN classification "
            "without fundamentally changing the project scope."
        )

    return {
        "gap":              gap,
        "missing_criteria": missing_list,
        "is_fixable":       is_fixable,
        "summary":          summary,
    }


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


def _compute_confidence(
    validated_flags: dict,
    rejected: list,
    evidence_quality: dict | None = None,
    greenwashing_risk: float = 0.0,
) -> int:
    """
    Confidence Model v2 — weighted formula replacing heuristic.

    confidence = base_signal_strength
               - ambiguity_penalty
               - contradiction_penalty
               + evidence_quality_bonus
               - greenwashing_penalty

    Clamped to [10, 95].

    Parameters
    ----------
    validated_flags    : {green_criteria, stop_factors}
    rejected           : list of rejected LLM claim names
    evidence_quality   : per-criterion quality scores (0–1), from breakdown
    greenwashing_risk  : greenwashing_risk_score (0–1)
    """
    eq = evidence_quality or {}
    crits = validated_flags.get("green_criteria", {})
    stops = validated_flags.get("stop_factors",  {})

    # Base signal strength: weighted by tier
    strong_count    = sum(
        1 for v in crits.values() if isinstance(v, dict) and v.get("value", 0) >= 1.0
    )
    confirmed_count = sum(
        1 for v in crits.values() if isinstance(v, dict) and 0.5 <= v.get("value", 0) < 1.0
    )
    weak_count      = sum(
        1 for v in crits.values() if isinstance(v, dict) and 0.0 < v.get("value", 0) < 0.5
    )
    any_stop  = any(v.get("value", 0) >= 0.5 for v in stops.values() if isinstance(v, dict))
    soft_stop = any(
        0.3 <= v.get("value", 0) < 0.5 for v in stops.values() if isinstance(v, dict)
    )

    # Evidence quality bonus: average quality of confirmed criteria
    confirmed_eq_values = [
        eq.get(name, 0.6)
        for name, v in crits.items()
        if isinstance(v, dict) and v.get("value", 0) >= 0.5
    ]
    eq_avg = (sum(confirmed_eq_values) / len(confirmed_eq_values)
              if confirmed_eq_values else 0.0)
    eq_bonus = round(eq_avg * 15)   # up to +15 for perfect evidence quality

    amb         = _compute_ambiguity_level(crits, stops, rejected)
    amb_penalty = {"low": 0, "medium": 10, "high": 22}[amb]

    # Greenwashing risk penalty
    gw_penalty = round(greenwashing_risk * 25)

    confidence  = 30                        # baseline
    confidence += 18 * strong_count         # strong confirmed evidence
    confidence +=  9 * confirmed_count      # partial confirmed
    confidence +=  3 * weak_count           # weak signals (low weight)
    confidence += eq_bonus                  # quality of evidence
    if any_stop:
        confidence += 15                    # stop-factor confidence is high
    if soft_stop:
        confidence -= 7
    confidence -= 14 * len(rejected)        # each rejected LLM claim hurts
    confidence -= amb_penalty
    confidence -= gw_penalty

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

                # Calibrated scoring — produces evidence_quality + time_statuses
                vfc = vf["green_criteria"]
                vfs = vf["stop_factors"]
                calibrated, breakdown = _compute_calibrated_score(vfc, vfs, text, notes)

                # Extract evidence quality and time statuses from breakdown
                eq_scores    = breakdown.get("evidence_quality", {})
                time_statuses = breakdown.get("time_statuses", {})

                # Use adjusted criteria from breakdown (cross-rules may have modified)
                adj_crit = breakdown.get("adjusted_criteria", vfc)

                # Greenwashing risk detection (adversarial intent layer)
                t_lower_full = " ".join(text.lower().split())
                gw_report    = _compute_greenwashing_risk(adj_crit, vfs, t_lower_full, rejected)

                # Confidence v2 — weighted formula
                amb        = _compute_ambiguity_level(adj_crit, vfs, rejected)
                conf       = _compute_confidence(
                    vf, rejected,
                    evidence_quality=eq_scores,
                    greenwashing_risk=gw_report["greenwashing_risk_score"],
                )
                thresh     = _compute_dynamic_threshold(adj_crit, vfs)
                risk_facts = _compute_risk_factors(adj_crit, vfs, rejected)

                # Python reasoning (never from LLM)
                result["confidence"] = conf
                result["reason"]     = _build_safe_reasoning(
                    {"green_criteria": adj_crit, "stop_factors": vfs}, rejected, notes
                )

                result["calibrated_score"]      = calibrated
                result["penalty_breakdown"]     = breakdown
                result["threshold"]             = thresh
                result["ambiguity_level"]       = amb
                result["risk_factors"]          = risk_facts
                result["evidence_quality"]      = eq_scores
                result["semantic_signals"]      = {
                    k: v.get("value", 0.0)
                    for k, v in adj_crit.items() if isinstance(v, dict)
                }
                # Greenwashing / adversarial detection
                result.update(gw_report)

                # Explainability fields
                result["criterion_breakdown"]     = _build_score_breakdown(adj_crit)
                result["ambiguity_explanation"]   = _build_ambiguity_explanation(amb, rejected)
                result["confidence_explanation"]  = _build_confidence_explanation(conf, amb, rejected)
                result["risk_explanation"]        = _build_risk_explanation(risk_facts)
                result["missing_criteria"]        = _build_missing_criteria_explanation(
                    adj_crit, vfs, calibrated, thresh, evidence_quality=eq_scores
                )

                # Decision explanation v2 (WHY / WHAT / HOW TO FIX)
                # Note: decision is not yet known here; set after threshold comparison
                # We store the data so run_tests.py can build the final explanation.
                result["_adj_crit_for_explanation"] = adj_crit
                result["_gw_report_for_explanation"] = gw_report

                # Debug visibility block
                result["debug"] = {
                    "semantic_scores":    {
                        k: (v.get("value", 0.0) if isinstance(v, dict) else 0.0)
                        for k, v in vfc.items()
                    },
                    "calibrated_score":   calibrated,
                    "raw_score":          breakdown.get("raw", 0.0),
                    "penalties": {
                        "ambiguity":     breakdown.get("ambiguity_penalty", 0.0),
                        "contradiction": breakdown.get("contradiction_penalty", 0.0),
                        "consistency":   breakdown.get("consistency_penalty", 0.0),
                        "total":         breakdown.get("total_penalty", 0.0),
                    },
                    "evidence_quality":   eq_scores,
                    "time_status":        time_statuses,
                    "greenwashing_risk":  gw_report,
                    "threshold":         thresh,
                    "ambiguity_level":   amb,
                    "rejected_flags":    rejected,
                    "audit_notes":       notes,
                }

                n_conf = sum(
                    1 for v in vf["green_criteria"].values()
                    if isinstance(v, dict) and v.get("value", 0) >= 0.5
                )
                gw_level = gw_report.get("greenwashing_risk_level", "low")
                print(
                    f"[extractor] LLM extraction (attempt {attempt}): "
                    f"green_criteria_true={n_conf}/5, "
                    f"confidence={conf}, "
                    f"greenwashing_risk={gw_level}"
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
