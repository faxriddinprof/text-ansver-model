import json
import sys
import os

# Project root (one level up from tests/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.utils.parser import read_txt
from src.utils.extractor import extract_data
from src.utils.engine import evaluate

# --- Load rules ---
with open("src/utils/green_rules.json", encoding="utf-8") as f:
    rules_json = json.load(f)

# --- Read test file ---
text = read_txt("checks/simple.txt")

# --- Extract features ---
data = extract_data(text)

# --- Evaluate ---
result = evaluate(data, rules_json)

# --- Print output ---
print("=" * 55)
print("EXTRACTED DATA:")
print("=" * 55)
for key, value in data.items():
    print(f"  {key}: {value}")

print()
print("=" * 55)
print(f"STATUS : {result['status']}")
print(f"SCORE  : {result['score']} rule(s) passed")
print("=" * 55)

reasons = result["decision_reasons"]

if reasons["exclusions_triggered"]:
    print("\n[EXCLUSIONS TRIGGERED]")
    for r in reasons["exclusions_triggered"]:
        print(f"  ✗ {r}")

if reasons["dependent_rules_triggered"]:
    print("\n[DEPENDENT RULES TRIGGERED]")
    for r in reasons["dependent_rules_triggered"]:
        print(f"  ⚡ {r}")

if reasons["passed_rules"]:
    print("\n[PASSED RULES]")
    for r in reasons["passed_rules"]:
        print(f"  ✓ {r}")

if reasons["failed_rules"]:
    print("\n[FAILED RULES]")
    for r in reasons["failed_rules"]:
        print(f"  ✗ {r}")

