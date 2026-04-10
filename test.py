import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils.parser import read_txt
from utils.extractor import extract_data
from utils.engine import evaluate

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
print("=" * 50)
print("EXTRACTED DATA:")
print("=" * 50)
for key, value in data.items():
    print(f"  {key}: {value}")

print()
print("=" * 50)
print("RESULT:")
print("=" * 50)
print(f"  Status : {result['status']}")
print(f"  Reason : {result['reason']}")
print(f"  Passed : {result['passed_rules']}")
print(f"  Failed : {result['failed_rules']}")
