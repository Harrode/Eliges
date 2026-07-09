"""EligES v2 — confidence-cascade screening (Tier 2.5 rule fast-path + Tier 3 LLM judge).

Per (criterion, patient) the cascade decides MET/NOT-MET by:
  1. Deterministic fast-path for *structured* constraints (numeric labs) when the
     patient has the relevant structured value — precise and free of LLM cost.
  2. Otherwise escalate to the generalizable LLM judge (Tier 3).

No dataset-specific rules are used: routing is decided from the LLM's own parse
of the criterion (lab vs. semantic), so the cascade generalizes to new criteria.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.llm_judge import extract_evidence, llm_judge

# Map free-text lab test names -> keys produced by extract_numeric_labs
_LAB_ALIASES = {
    "hba1c": "hba1c", "a1c": "hba1c", "hemoglobin a1c": "hba1c",
    "glycohemoglobin": "hba1c", "glycosylated hemoglobin": "hba1c",
    "creatinine": "creatinine", "creat": "creatinine", "serum creatinine": "creatinine",
}


def _lab_key(test_name: str) -> Optional[str]:
    t = str(test_name or "").lower().strip()
    for alias, key in _LAB_ALIASES.items():
        if alias in t:
            return key
    return None


def is_structured_lab_only(conds: Dict) -> bool:
    """True if the criterion reduces to numeric lab constraints only."""
    keys = {k for k, v in conds.items() if v not in (None, [], {}, "")}
    if "lab_tests" not in keys:
        return False
    return keys.issubset({"lab_tests", "age_min", "age_max", "gender"})


def _value_satisfies(value: float, lab: Dict) -> bool:
    op = (lab.get("op") or "").lower()
    vmin, vmax, v = lab.get("value_min"), lab.get("value_max"), lab.get("value")
    if op == "between" and vmin is not None and vmax is not None:
        return vmin <= value <= vmax
    if op in (">", "gt") and v is not None:
        return value > v
    if op in (">=", "gte") and v is not None:
        return value >= v
    if op in ("<", "lt") and v is not None:
        return value < v
    if op in ("<=", "lte") and v is not None:
        return value <= v
    if v is not None:  # default equality-ish tolerance
        return abs(value - v) < 1e-6
    return False


def deterministic_lab_verdict(conds: Dict, text: str, extract_numeric_labs) -> Tuple[bool, bool, str]:
    """Return (decided, met, reason). decided=False -> escalate to LLM judge."""
    labs = extract_numeric_labs(text)
    any_value_seen = False
    for lab in conds.get("lab_tests", []):
        if not isinstance(lab, dict):
            continue
        key = _lab_key(lab.get("test", ""))
        if not key:
            return (False, False, "unmapped lab -> escalate")
        values = labs.get(key, [])
        if values:
            any_value_seen = True
            if any(_value_satisfies(v, lab) for v in values):
                return (True, True, f"{key}={values} satisfies constraint")
    if any_value_seen:
        return (True, False, "lab value present but out of range")
    # No structured value found -> cannot decide deterministically
    return (False, False, "no structured lab value -> escalate")


def screen_one(criterion: str, conds: Dict, patient_text: str,
               llm_client, model: str, extract_numeric_labs, strip_thinking=None) -> Dict:
    """Cascade verdict for one (criterion, patient). Returns {met, source, reason}."""
    # Tier 2.5: deterministic fast-path for pure structured-lab criteria
    if is_structured_lab_only(conds):
        decided, met, reason = deterministic_lab_verdict(conds, patient_text, extract_numeric_labs)
        if decided:
            return {"met": met, "source": "rule", "reason": reason}
        # else fall through to judge

    # Tier 3: generalizable LLM judge
    evidence = extract_evidence(patient_text, criterion)
    verdict = llm_judge(criterion, evidence, llm_client, model, strip_thinking=strip_thinking)
    return {"met": bool(verdict["met"]), "source": verdict.get("source", "llm"),
            "reason": verdict.get("reason", "")}
