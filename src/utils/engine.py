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
    Supports 'boolean' and 'threshold' rule types.
    Returns (passed_count, list of passed rule ids).
    """
    passed = []

    for rule in rules:
        field = rule["field"]

        if field not in data or data[field] is None:
            continue

        rule_type = rule["type"]

        if rule_type == "boolean":
            if data[field] == rule.get("expected", True):
                passed.append(rule["id"])

        elif rule_type == "threshold":
            operator = rule.get("operator")
            threshold = rule.get("value")
            if operator in OPERATORS:
                try:
                    if OPERATORS[operator](float(data[field]), float(threshold)):
                        passed.append(rule["id"])
                except (TypeError, ValueError):
                    pass

    return len(passed), passed


def evaluate(data, rules_json):
    """
    Full evaluation pipeline:
    1. Check exclusions — any match → NOT GREEN
    2. Check rules — passed >= 3 → GREEN, else → NOT GREEN
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
            "passed_count": 0,
        }

    # Step 2: rules
    passed_count, passed_rules = check_rules(data, rules)

    if passed_count >= 3:
        return {
            "status": "GREEN",
            "reason": f"Passed {passed_count} rule(s)",
            "passed_rules": passed_rules,
            "passed_count": passed_count,
        }

    return {
        "status": "NOT GREEN",
        "reason": f"Only {passed_count} rule(s) passed (minimum 3 required)",
        "passed_rules": passed_rules,
        "passed_count": passed_count,
    }

