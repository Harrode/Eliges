"""Cohort profile loader — dataset-specific criterion catalogs (JSON, not Python if-else)."""
from __future__ import annotations

import fnmatch
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

PROFILE_DIR = Path(__file__).resolve().parent.parent / "config" / "cohort_profiles"
_PROFILE_CACHE: Dict[str, Dict] = {}


def list_profiles() -> List[Dict[str, str]]:
    out = []
    for path in sorted(PROFILE_DIR.glob("*.json")):
        data = load_profile(path.stem)
        out.append({
            "profile_id": data.get("profile_id", path.stem),
            "display_name": data.get("display_name", path.stem),
            "description": data.get("description", ""),
        })
    return out


def load_profile(profile_id: str) -> Dict[str, Any]:
    if profile_id in _PROFILE_CACHE:
        return _PROFILE_CACHE[profile_id]
    path = PROFILE_DIR / f"{profile_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cohort profile not found: {profile_id}")
    data = json.loads(path.read_text(encoding="utf-8"))
    _PROFILE_CACHE[profile_id] = data
    return data


def profile_for_index(index_name: str) -> str:
    """Map ES index name to profile_id via index_patterns in profile JSON."""
    index_lower = (index_name or "").lower()
    best = "generic"
    for path in sorted(PROFILE_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = data.get("profile_id", path.stem)
        if pid == "generic":
            continue
        for pat in data.get("index_patterns", []):
            if fnmatch.fnmatch(index_lower, pat.lower()):
                return pid
    return best


def find_criterion_entry(profile: Dict, query: str, criterion_id: Optional[str] = None) -> Optional[Dict]:
    criteria = profile.get("criteria") or []
    if criterion_id:
        for entry in criteria:
            if entry.get("id") == criterion_id:
                return entry
    lowered = (query or "").lower()
    for entry in criteria:
        if entry.get("canonical_text", "").lower().strip() == lowered.strip():
            return entry
    for entry in criteria:
        for sub in entry.get("match_substrings", []):
            if sub.lower() in lowered:
                return entry
    return None


def get_criterion_route(
    profile_id: str,
    criterion_id: Optional[str],
    conds: Dict,
    query: str = "",
) -> Dict[str, Any]:
    profile = load_profile(profile_id)
    entry = find_criterion_entry(profile, query, criterion_id)
    if entry and isinstance(entry.get("route"), dict):
        return dict(entry["route"])
    return {}


def _deep_merge(base: Dict, overrides: Dict) -> Dict:
    out = deepcopy(base)
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _sanitize_conflicting_llm_fields(out: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Drop LLM fields that fight profile parse_overrides (e.g. criterion echo in diagnoses)."""
    if overrides.get("inclusion_logic"):
        for key in (
            "diagnoses", "diagnoses_any", "inclusion_terms", "conditions_excluded",
            "medications", "time_conditions",
        ):
            out.pop(key, None)
        return

    if overrides.get("inclusion_terms"):
        out.pop("diagnoses", None)

    if overrides.get("diagnoses_any"):
        out.pop("diagnoses", None)

    if overrides.get("temporal_event"):
        out.pop("diagnoses", None)
        out.pop("time_conditions", None)

    if overrides.get("evidence_groups") or overrides.get("composite_min_count"):
        out.pop("diagnoses", None)
        out.pop("medications", None)

    if overrides.get("required_medications"):
        out.pop("medications", None)
        if not overrides.get("diagnoses_any"):
            out.pop("diagnoses", None)

    if overrides.get("lab_tests"):
        out.pop("diagnoses", None)

    if "temporal_event" not in overrides:
        out.pop("temporal_event", None)

    if "inclusion_logic" not in overrides:
        out.pop("inclusion_logic", None)


def _dedupe_lab_tests(out: Dict[str, Any]) -> None:
    labs = out.get("lab_tests")
    if not isinstance(labs, list):
        return
    seen = set()
    unique = []
    for lab in labs:
        key = json.dumps(lab, sort_keys=True) if isinstance(lab, dict) else str(lab)
        if key in seen:
            continue
        seen.add(key)
        unique.append(lab)
    if unique:
        out["lab_tests"] = unique
    else:
        out.pop("lab_tests", None)


def apply_profile_overrides(
    conds: Dict[str, Any],
    query: str,
    profile_id: str = "generic",
    criterion_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge profile parse_overrides into LLM parse result."""
    profile = load_profile(profile_id)
    entry = find_criterion_entry(profile, query, criterion_id)
    if not entry:
        return conds

    overrides = entry.get("parse_overrides") or {}
    if not overrides:
        return conds

    if entry.get("replace"):
        out = deepcopy(overrides)
    else:
        out = _deep_merge(conds, overrides)
        _sanitize_conflicting_llm_fields(out, overrides)

    # Drop empty lists from LLM that would block profile fields
    for key in list(out.keys()):
        if out[key] == [] or out[key] is None:
            if key not in overrides and key in conds:
                continue
            if out[key] == []:
                out.pop(key, None)

    _dedupe_lab_tests(out)
    from app.b1_parse_merge import _sanitize_temporal_event
    _sanitize_temporal_event(out)
    return out


def apply_generic_parse_pipeline(conds: Dict[str, Any], query: str) -> Dict[str, Any]:
    """Dataset-agnostic post-processing (always safe)."""
    from app.b1_parse_merge import enrich_track1_conditions

    # Reuse generic-only steps inside enrich_track1_conditions via enable_enrich=False
    return enrich_track1_conditions(deepcopy(conds), query, enable_enrich=False)
