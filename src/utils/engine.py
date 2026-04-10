"""
engine.py

Explainable decision engine for green project classification.
Every decision is fully traceable: exclusions, rules, dependent rules, score.
"""

OPERATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">":  lambda a, b: a > b,
    "<":  lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _eval_condition(data, condition):
    """
    Evaluate a single {field, operator, value} condition against data.
    Returns True/False. Missing or None field → False.
    """
    field = condition.get("field")
    if field not in data or data[field] is None:
        return False
    operator = condition.get("operator")
    value = condition.get("value")
    if operator not in OPERATORS:
        return False
    try:
        return OPERATORS[operator](data[field], value)
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Step 1: Exclusions
# ---------------------------------------------------------------------------

def check_exclusions(data, exclusions):
    """
    Evaluate all exclusions.
    Returns a list of human-readable strings for every triggered exclusion.
    """
    triggered = []
    for excl in exclusions:
        field = excl["field"]
        expected = excl["value"]
        if data.get(field) == expected:
            triggered.append(f"{excl['id']} - {excl['description']}")
    return triggered


# ---------------------------------------------------------------------------
# Step 2: Normal rules
# ---------------------------------------------------------------------------

def check_rules(data, rules):
    """
    Evaluate all rules. Skip rules whose field is absent or None.
    Supports 'boolean' and 'threshold' (>= / <= only) types.
    Returns:
        passed_rules: list of explanation strings
        failed_rules: list of explanation strings
    """
    passed = []
    failed = []

    for rule in rules:
        field = rule["field"]
        desc = rule.get("description", rule.get("name", field))

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
            passed.append(f"{rule['id']} - PASSED - {desc}")
        else:
            failed.append(f"{rule['id']} - FAILED - {desc}")

    return passed, failed


# ---------------------------------------------------------------------------
# Step 3: Dependent rules
# ---------------------------------------------------------------------------

def check_dependent_rules(data, dependent_rules):
    """
    Evaluate dependent rules:
    - Evaluate IF condition
    - If IF matches → evaluate THEN (single condition or any_of)
    - If THEN fails → apply on_failure: "not_green" | "reclassify_required" | null

    Returns:
        triggered: list of explanation strings for rules whose IF fired
        override_status: "NOT GREEN" | "REQUIRES_REVIEW" | None
    """
    triggered = []
    override_status = None

    for dr in dependent_rules:
        dr_id = dr["id"]
        description = dr.get("description", dr_id)
        if_cond = dr.get("if", {})
        then_block = dr.get("then", {})
        on_failure = dr.get("on_failure")

        # Evaluate IF
        if not _eval_condition(data, if_cond):
            continue  # IF not triggered — skip entirely

        triggered.append(f"{dr_id} - {description}")

        # Evaluate THEN
        then_passed = False

        if "any_of" in then_block:
            then_passed = any(_eval_condition(data, c) for c in then_block["any_of"])
        else:
            then_passed = _eval_condition(data, then_block)

        if not then_passed and on_failure:
            if on_failure == "not_green":
                override_status = "NOT GREEN"
            elif on_failure == "reclassify_required":
                # Only upgrade to REQUIRES_REVIEW, never downgrade from NOT GREEN
                if override_status != "NOT GREEN":
                    override_status = "REQUIRES_REVIEW"

    return triggered, override_status


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(data, rules_json):
    """
    Evaluation order:
      1. Exclusions   → any match → NOT GREEN immediately
      2. Dependent rules → may override final status
      3. Normal rules → score = len(passed)
      4. Final status: override > score-based
    """
    exclusions      = rules_json.get("exclusions", [])
    rules           = rules_json.get("rules", [])
    dependent_rules = rules_json.get("dependent_rules", [])

    # --- Step 1: Exclusions ---
    exclusions_triggered = check_exclusions(data, exclusions)

    if exclusions_triggered:
        return {
            "status": "NOT GREEN",
            "score": 0,
            "decision_reasons": {
                "exclusions_triggered": exclusions_triggered,
                "passed_rules": [],
                "failed_rules": [],
                "dependent_rules_triggered": [],
            },
        }

    # --- Step 2: Dependent rules ---
    dep_triggered, override_status = check_dependent_rules(data, dependent_rules)

    # --- Step 3: Normal rules ---
    passed_rules, failed_rules = check_rules(data, rules)
    score = len(passed_rules)

    # --- Step 4: Final status ---
    if override_status:
        status = override_status
    elif score >= 3:
        status = "GREEN"
    else:
        status = "NOT GREEN"

    return {
        "status": status,
        "score": score,
        "decision_reasons": {
            "exclusions_triggered": [],
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "dependent_rules_triggered": dep_triggered,
        },
    }

