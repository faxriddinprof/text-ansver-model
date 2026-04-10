"""
engine.py

Rule-based evaluation engine for green project classification.
Applies exclusions, boolean rules, and threshold rules against extracted data.
"""

OPERATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def check_exclusions(data, exclusions):
    """
    Check if any exclusion condition is met.
    Returns (True, rule_id) if a disqualifying condition is found,
    otherwise (False, None).
    """
    for excl in exclusions:
        field = excl["field"]
        expected = excl["value"]
        if data.get(field) == expected:
            return True, excl["id"]
    return False, None


def check_rules(data, rules):
    """
    Evaluate all rules against extracted data.
    A rule is only evaluated if its field exists in data AND is not None.
    Supports 'boolean' and 'threshold' (>= / <=) rule types.
    Returns (passed_rules, failed_rules) as lists of rule IDs.
    """
    passed = []
    failed = []

    for rule in rules:
        field = rule["field"]

        # Skip entirely if field is missing or None
        if field not in data or data[field] is None:
            continue

        rule_type = rule["type"]
        result = False

        if rule_type == "boolean":
            result = data[field] == rule.get("expected", True)

        elif rule_type == "threshold":
            operator = rule.get("operator")
            threshold = rule.get("value")
            if operator in (">=", "<="):
                try:
                    result = OPERATORS[operator](float(data[field]), float(threshold))
                except (TypeError, ValueError):
                    pass

        if result:
            passed.append(rule["id"])
        else:
            failed.append(rule["id"])

    return passed, failed


def evaluate(data, rules_json):
    """
    Full evaluation pipeline:
    1. Check exclusions — any match → NOT GREEN immediately
    2. Check rules — len(passed) >= 3 → GREEN, else → NOT GREEN
    """
    exclusions = rules_json.get("exclusions", [])
    rules = rules_json.get("rules", [])

    # Step 1: exclusions
    is_excluded, excl_id = check_exclusions(data, exclusions)
    if is_excluded:
        return {
            "status": "NOT GREEN",
            "reason": f"Failed exclusion: {excl_id}",
            "passed_rules": [],
            "failed_rules": [],
        }

    # Step 2: rules
    passed_rules, failed_rules = check_rules(data, rules)

    if len(passed_rules) >= 3:
        return {
            "status": "GREEN",
            "reason": f"Passed {len(passed_rules)} rule(s)",
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
        }

    return {
        "status": "NOT GREEN",
        "reason": f"Only {len(passed_rules)} rule(s) passed (minimum 3 required)",
        "passed_rules": passed_rules,
        "failed_rules": failed_rules,
    }

