"""Hybrid pipeline routing — when to use L4 only vs LLM judge."""
from __future__ import annotations

from typing import Dict, Optional

from app.cohort_profile import get_criterion_route, load_profile


def is_lab_only(conds: Dict) -> bool:
    keys = {k for k, v in conds.items() if v not in (None, [], {}, "")}
    if "lab_tests" not in keys:
        return False
    return keys.issubset({"lab_tests", "age_min", "age_max", "gender"})


def needs_semantic_judge(
    conds: Dict,
    profile_id: str = "generic",
    criterion_id: Optional[str] = None,
    query: str = "",
) -> bool:
    """Return True if this (criterion, patient) path should call llm_judge after L4."""
    route = get_criterion_route(profile_id, criterion_id, conds, query)
    if "judge" in route:
        return bool(route["judge"])

    profile = load_profile(profile_id)
    routing = profile.get("routing") or {}

    if routing.get("judge_never_lab_only") and is_lab_only(conds):
        return False

    for field in routing.get("judge_when_fields", []):
        if field in conds and conds[field] not in (None, [], {}, ""):
            return True

    # Generic default: judge if not pure lab/age/gender
    if is_lab_only(conds):
        return False
    return True
