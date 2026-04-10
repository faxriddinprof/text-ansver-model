"""
extractor.py

Extracts structured features from raw text using keyword matching and regex.
"""

import re


def extract_data(text):
    """
    Convert text to lowercase and extract evaluation fields.
    Returns a flat dictionary of feature values.
    """
    t = text.lower()

    data = {}

    # --- EXCLUSIONS ---
    data["is_coal_based_project"] = "ko'mir" in t or "koʻmir" in t
    data["involves_alcohol_or_tobacco"] = "alkogol" in t or "tamaki" in t

    # --- BOOLEAN RULES ---
    data["uses_solar_energy"] = "quyosh" in t
    data["installs_dust_gas_filter_products"] = "filtr" in t
    data["improves_water_supply_quality_or_efficiency"] = "suv" in t and ("samarad" in t or "tejam" in t or "qayta ishlash" in t)
    data["reduces_ghg_emissions_in_production"] = "issiqxona gaz" in t or "chiqindi" in t

    # --- CERTIFICATE ---
    data["has_compliance_certificate_from_authorized_body"] = "sertifikat" in t

    # --- THRESHOLD: building energy efficiency % ---
    building_match = re.search(r"(\d+)\s*%", t)
    data["building_energy_or_carbon_reduction_percent"] = int(building_match.group(1)) if building_match else None

    # --- THRESHOLD: hydropower capacity (number before "mw") ---
    hydro_mw_match = re.search(r"(\d+(?:\.\d+)?)\s*mw", t)
    data["hydropower_capacity_mw"] = float(hydro_mw_match.group(1)) if hydro_mw_match else None

    # --- THRESHOLD: hydropower CO2 emission (number before "g/kvt" or "g kvt") ---
    hydro_co2_match = re.search(r"(\d+(?:\.\d+)?)\s*g(?:/|\s)kvt", t)
    data["hydropower_co2_emission_g_per_kwh"] = float(hydro_co2_match.group(1)) if hydro_co2_match else None

    return data

