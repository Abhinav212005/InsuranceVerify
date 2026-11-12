"""Runs all rules and decides the review status. Pure functions: no database access."""
from dataclasses import dataclass

from app.validation.rules import RULES, Ctx, Issue

ORDER = {"error": 0, "warning": 1, "info": 2}
MISSING_RULE_CODES = {"MISSING_FIELD", "POLICY_NUMBER_MISSING", "CLAIM_NUMBER_MISSING"}


@dataclass
class Outcome:
    status: str  # valid | needs_review | incomplete
    issues: list[Issue]


def decide_status(issues: list[Issue]) -> str:
    if any(i.code in MISSING_RULE_CODES for i in issues):
        return "incomplete"
    if any(i.severity in ("error", "warning") for i in issues):
        return "needs_review"
    return "valid"


def validate(ctx: Ctx) -> Outcome:
    issues = [i for rule in RULES for i in rule(ctx)]
    issues.sort(key=lambda i: ORDER[i.severity])  # stable: keeps rule order inside a severity
    return Outcome(decide_status(issues), issues)
