"""Merge and enrich L1 parse results for B1 (without modifying B5 chia_parser)."""
import os
import re
from typing import Any, Dict, List

ENABLE_TRACK1_ENRICH = os.getenv("ENABLE_TRACK1_ENRICH", "1") not in ("0", "false", "False")


MAJOR_DIABETES_TERMS = [
    "amputation", "kidney damage", "skin conditions", "retinopathy",
    "nephropathy", "neuropathy", "renal failure", "kidney failure",
    "diabetic foot", "foot ulcer", "skin ulcer", "diabetic ulcer",
    "esrd", "dialysis", "neuropathy", "nephropathy",
]

ABDOMINAL_TERMS = [
    "abdominal surgery", "intra-abdominal surgery", "laparotomy",
    "appendectomy", "cholecystectomy", "colectomy",
    "colon resection", "bowel resection", "small bowel resection",
    "large bowel resection", "large intestine resection",
    "small bowel obstruction", "bowel obstruction",
    "gastric bypass", "hysterectomy", "exploratory laparotomy", "exploratory lap",
    "intestinal surgery", "partial colectomy", "sigmoid resection", "ileostomy", "colostomy",
    "history of appendectomy", "s/p appendectomy", "status post colectomy",
]

ALCOHOL_POSITIVE_TERMS = [
    "alcohol abuse", "etoh abuse", "alcohol dependence", "alcoholism",
    "heavy alcohol", "drinks heavily", "daily alcohol", "six pack",
    "beer", "beers per day", "drinks per day",
]

ALCOHOL_NEGATIVE_TERMS = [
    "social alcohol", "occasional alcohol", "rare alcohol",
    "denies alcohol", "no alcohol", "does not drink",
]

SUPPLEMENT_TERMS = [
    "dietary supplement", "herbal", "ginkgo", "ginseng", "st john",
    "fish oil", "glucosamine", "chondroitin", "coenzyme q",
    "coq10", "vitamin e", "vitamin c", "multivitamin",
]


def _cad_evidence_groups() -> List[Dict[str, Any]]:
    try:
        from app.b1_drug_terms import cad_medication_terms
        cad_meds = list(cad_medication_terms())
    except Exception:
        cad_meds = [
            "aspirin", "asa", "plavix", "clopidogrel", "nitroglycerin",
            "atenolol", "metoprolol", "lipitor", "atorvastatin", "pravastatin",
            "simvastatin", "zocor", "beta blocker", "statin",
        ]
    return [
        {"id": "cad_medications", "terms": cad_meds, "min_hits": 2, "mode": "med_count"},
        {"id": "myocardial_infarction", "terms": [
            "myocardial infarction", "history of mi", "s/p mi", " mi ", "acute mi",
        ], "mode": "positive"},
        {"id": "angina", "terms": ["angina", "chest pain"], "mode": "positive"},
        {"id": "ischemia", "terms": [
            "ischemia", "ischemic", "positive stress test", "cabg", "ptca", "pci", "stent",
        ], "mode": "positive"},
    ]


def _append_unique_list(target: List[Any], values: List[Any]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def merge_llm_with_chia(
    llm_result: Dict[str, Any],
    query: str,
    profile_id: str = None,
    criterion_id: str = None,
) -> Dict[str, Any]:
    """Fill LLM gaps from parse_criteria and apply cohort profile overrides."""
    try:
        from tests.chia_parser import parse_criteria
        from tests.drug_lexicon import load_drug_lexicon
        chia = parse_criteria(query, drug_lexicon=load_drug_lexicon())
    except Exception:
        chia = {}

    merged = dict(llm_result)
    profile = load_profile_id(profile_id)
    prof = _load_profile_safe(profile)
    if prof.get("parse_pipeline", {}).get("merge_chia_fallback", True):
        if chia.get("lab_tests") and (not merged.get("lab_tests")):
            merged["lab_tests"] = chia["lab_tests"]
        if chia.get("time_conditions") and not merged.get("time_conditions"):
            merged["time_conditions"] = chia["time_conditions"]

    from app.cohort_profile import apply_profile_overrides
    merged = apply_profile_overrides(merged, query, profile, criterion_id)
    return enrich_track1_conditions(merged, query, enable_enrich=_legacy_enrich_enabled(profile))


def load_profile_id(profile_id: str = None) -> str:
    if profile_id:
        return profile_id
    return "n2c2_track1" if ENABLE_TRACK1_ENRICH else "generic"


def _load_profile_safe(profile_id: str) -> Dict[str, Any]:
    try:
        from app.cohort_profile import load_profile
        return load_profile(profile_id)
    except Exception:
        return {}


def _legacy_enrich_enabled(profile_id: str) -> bool:
    """Legacy _fix_* path only when profile is n2c2 and env flag on (ablation compat)."""
    if profile_id != "n2c2_track1":
        return False
    try:
        from app.cohort_profile import load_profile
        prof = load_profile(profile_id)
        if prof.get("criteria"):
            return False  # JSON profile replaces legacy _fix_*
    except Exception:
        pass
    return ENABLE_TRACK1_ENRICH


def _sanitize_temporal_event(out: Dict[str, Any]) -> None:
    te = out.get("temporal_event")
    if not isinstance(te, dict):
        return
    for key in ("window_years", "window_months"):
        if te.get(key) is None:
            te.pop(key, None)
    if te.get("window_months") is None and te.get("window_years") is None:
        out.pop("temporal_event", None)


def enrich_track1_conditions(
    conds: Dict[str, Any],
    query: str,
    enable_enrich: bool = None,
) -> Dict[str, Any]:
    lowered = query.lower()
    out = dict(conds)

    # Generic normalizations (not criterion-specific) always run.
    _promote_diagnoses_to_any(out)
    _strip_criterion_echo_diagnoses(out, query)
    _extract_english_temporal(out, query)
    _normalize_lab_tests(out)
    _fix_lab_ranges_from_query(out, lowered)
    _extract_labs_from_query_if_missing(out, lowered)

    use_enrich = ENABLE_TRACK1_ENRICH if enable_enrich is None else enable_enrich
    # Criterion-specific (n2c2 Track1) enrichment. Gated for ablation / legacy path.
    if use_enrich:
        _fix_alcohol_abuse(out, lowered)
        _fix_major_diabetes(out, lowered)
        _fix_advanced_cad(out, lowered)
        _fix_asp_for_mi(out, lowered)
        _fix_abdominal(out, lowered)
        _fix_english(out, lowered)
        _fix_makes_decisions(out, lowered)
        _fix_drug_abuse(out, lowered)
        _fix_mi_6mos(out, lowered)
        _fix_keto_1yr(out, lowered)
        _fix_dietsupp_2mos(out, lowered)

    _strip_bad_chia_medications(out, lowered)
    _strip_lab_names_from_medications(out)
    _sanitize_temporal_event(out)

    return {k: v for k, v in out.items() if v is not None and v != []}


def _strip_lab_names_from_medications(out: Dict[str, Any]) -> None:
    lab_noise = {
        "creatinine", "hemoglobin", "hba1c", "glucose", "albumin", "bilirubin",
        "alt", "ast", "sodium", "potassium", "bun", "egfr",
    }
    meds = out.get("medications") or []
    cleaned = [m for m in meds if str(m).lower().strip() not in lab_noise]
    if cleaned:
        out["medications"] = cleaned
    else:
        out.pop("medications", None)


def _promote_diagnoses_to_any(out: Dict[str, Any]) -> None:
    if out.get("diagnoses") and not out.get("diagnoses_any"):
        out["diagnoses_any"] = out.pop("diagnoses")


def _strip_criterion_echo_diagnoses(out: Dict[str, Any], query: str) -> None:
    q_lower = (query or "").lower().strip(" .")
    if not q_lower:
        return
    for field in ("diagnoses", "diagnoses_any"):
        terms = out.get(field)
        if not terms:
            continue
        cleaned = []
        for term in terms:
            tl = str(term).lower().strip(" .")
            if tl == q_lower:
                continue
            if len(tl) > 50 and q_lower.startswith(tl[:40]):
                continue
            if q_lower in tl and abs(len(tl) - len(q_lower)) < 8:
                continue
            cleaned.append(term)
        if cleaned:
            out[field] = cleaned
        else:
            out.pop(field, None)


def _extract_english_temporal(out: Dict[str, Any], query: str) -> None:
    if out.get("temporal_event"):
        out.pop("time_conditions", None)
        return
    lowered = (query or "").lower()
    m = re.search(
        r"(?:in the (?:past|last)|within the (?:past|last))\s+"
        r"(?:(\d+)\s+(day|days|week|weeks|month|months|year|years)"
        r"|(?:a|an)\s+(day|days|week|weeks|month|months|year|years)"
        r"|(year|years|month|months|week|weeks|day|days))\b",
        lowered,
    )
    if not m:
        return

    if m.group(1):
        n = int(m.group(1))
        unit = m.group(2).rstrip("s")
    elif m.group(3):
        n = 1
        unit = m.group(3).rstrip("s")
    else:
        n = 1
        unit = m.group(4).rstrip("s")
    event_terms = []
    for term in out.pop("diagnoses", []) or []:
        if isinstance(term, str) and term.strip():
            event_terms.append(term.strip())
    for term in out.get("diagnoses_any") or []:
        if isinstance(term, str) and term.strip():
            event_terms.append(term.strip())

    if not event_terms:
        before = query[: m.start()].strip(" .,")
        for prefix in ("diagnosis of ", "history of ", "taken a ", "taken ", "any "):
            if before.lower().startswith(prefix):
                before = before[len(prefix) :]
        clause = before.split(",")[0].strip()
        if clause:
            event_terms = [clause]

    temporal: Dict[str, Any] = {"event_terms": event_terms}
    if unit == "month":
        temporal["window_months"] = n
    elif unit == "year":
        temporal["window_years"] = n
    elif unit == "week":
        temporal["window_months"] = max(1, round(n * 7 / 30))
    elif unit == "day":
        temporal["window_months"] = max(1, round(n / 30))

    if "excluding vitamin d" in lowered or "exclude vitamin d" in lowered:
        temporal["exclude_vitamin_d_only"] = True

    out["temporal_event"] = temporal
    out.pop("time_conditions", None)


def _extract_labs_from_query_if_missing(out: Dict[str, Any], lowered: str) -> None:
    if out.get("lab_tests"):
        return
    between = re.search(r"between\s+([\d.]+)\s*%?\s*and\s+([\d.]+)\s*%?", lowered)
    if between and any(k in lowered for k in ("hba1c", "a1c", "hemoglobin")):
        out["lab_tests"] = [{
            "test": "HbA1c",
            "op": "between",
            "value_min": float(between.group(1)),
            "value_max": float(between.group(2)),
        }]
        return
    if "creatinine" in lowered and ("upper limit" in lowered or "greater than" in lowered):
        out["lab_tests"] = [{"test": "Creatinine", "op": ">", "value": 1.3}]


def _normalize_lab_tests(out: Dict[str, Any]) -> None:
    labs = out.get("lab_tests") or []
    normalized = []
    for lab in labs:
        if not isinstance(lab, dict):
            continue
        item = dict(lab)
        value = item.get("value")
        if isinstance(value, str) and "upper limit" in value.lower():
            item["value"] = 1.3
            item["op"] = ">"
            item["test"] = item.get("test") or "creatinine"
        normalized.append(item)
    if normalized:
        out["lab_tests"] = normalized


def _fix_lab_ranges_from_query(out: Dict[str, Any], lowered: str) -> None:
    """Patch lab_tests entries that have a name but no op/value by extracting
    ranges directly from the criterion text. Handles LLMs that return only
    test names without operators (e.g. qwen returns ["HbA1c"] instead of
    [{"test":"HbA1c","op":"between","value_min":6.5,"value_max":9.5}]).
    """
    import re as _re

    labs = out.get("lab_tests") or []
    patched = []
    for lab in labs:
        if not isinstance(lab, dict):
            patched.append(lab)
            continue
        test = str(lab.get("test", "")).lower()
        has_value = lab.get("value") is not None or lab.get("value_min") is not None

        if has_value:
            patched.append(lab)
            continue

        # HbA1c / hemoglobin A1c — try to extract "between X and Y" from query
        if any(k in test for k in ("hba1c", "a1c", "hemoglobin")):
            m = _re.search(r"between\s+([\d.]+)\s*%?\s*and\s+([\d.]+)\s*%?", lowered)
            if m:
                lab = dict(lab)
                lab.update({"test": "HbA1c", "op": "between",
                            "value_min": float(m.group(1)), "value_max": float(m.group(2))})
            elif "upper limit" in lowered or "greater than" in lowered:
                pass  # leave as-is, handled elsewhere
        # Creatinine — "greater than the upper limit of normal"
        elif "creatinine" in test:
            if "upper limit" in lowered or "greater than" in lowered:
                lab = dict(lab)
                lab.update({"test": "Creatinine", "op": ">", "value": 1.3})
        patched.append(lab)

    if patched:
        out["lab_tests"] = patched
    else:
        out.pop("lab_tests", None)


def _strip_bad_chia_medications(out: Dict[str, Any], lowered: str) -> None:
    """Remove phrase fragments mis-parsed as medications (e.g. 'in the past 6 months')."""
    meds = out.get("medications") or []
    if not meds:
        return
    cleaned = []
    for med in meds:
        med_lower = str(med).lower()
        if any(x in med_lower for x in ("past", "month", "year", "week", "day", "within")):
            continue
        if len(med_lower) > 40:
            continue
        cleaned.append(med)
    if cleaned:
        out["medications"] = cleaned
    else:
        out.pop("medications", None)


def _fix_alcohol_abuse(out: Dict[str, Any], lowered: str) -> None:
    if "alcohol" not in lowered or "exclude" in lowered:
        return
    excluded = out.pop("conditions_excluded", None) or []
    kept = [x for x in excluded if "alcohol" not in str(x).lower()]
    if kept:
        out["conditions_excluded"] = kept
    out["inclusion_terms"] = list(ALCOHOL_POSITIVE_TERMS)
    out["inclusion_negative_terms"] = list(ALCOHOL_NEGATIVE_TERMS)
    meds = out.get("medications") or []
    if meds and all(str(m).lower() in ("alcohol", "ethanol") for m in meds):
        out.pop("medications", None)


def _fix_major_diabetes(out: Dict[str, Any], lowered: str) -> None:
    if "diabetes-related complication" not in lowered and "major diabetes" not in lowered:
        return
    if out.get("diagnoses"):
        out["diagnoses_any"] = out.pop("diagnoses")
    _append_unique_list(out.setdefault("diagnoses_any", []), MAJOR_DIABETES_TERMS)


def _fix_advanced_cad(out: Dict[str, Any], lowered: str) -> None:
    if "two or more" not in lowered or "cardiovascular" not in lowered:
        return
    out.pop("diagnoses", None)
    out["composite_min_count"] = 2
    out["evidence_groups"] = _cad_evidence_groups()


def _fix_asp_for_mi(out: Dict[str, Any], lowered: str) -> None:
    if "aspirin" not in lowered:
        return
    try:
        from app.b1_drug_terms import PREVENTIVE_ASPIRIN_TERMS
        out["required_medications"] = list(PREVENTIVE_ASPIRIN_TERMS)
    except Exception:
        out["required_medications"] = ["aspirin", "asa"]
    out["diagnoses_any"] = [
        "myocardial infarction", "coronary artery disease", "cad",
        "ischemia", "history of mi", "s/p mi", "stent", "cabg", "pci",
    ]
    out.pop("diagnoses", None)
    out.pop("medications", None)


def _fix_abdominal(out: Dict[str, Any], lowered: str) -> None:
    if "intra-abdominal surgery" not in lowered and "small bowel obstruction" not in lowered:
        return
    out.pop("conditions_excluded", None)
    out.pop("diagnoses_excluded", None)
    out.pop("diagnoses", None)
    terms = list(ABDOMINAL_TERMS)
    out["diagnoses_any"] = terms
    out["inclusion_terms"] = terms


def _fix_english(out: Dict[str, Any], lowered: str) -> None:
    if "speak english" not in lowered and "must speak english" not in lowered:
        return
    out.clear()
    out["inclusion_logic"] = "english_speaker"


def _fix_makes_decisions(out: Dict[str, Any], lowered: str) -> None:
    if "make their own medical decisions" not in lowered:
        return
    out.clear()
    out["inclusion_logic"] = "decision_capacity"


def _fix_drug_abuse(out: Dict[str, Any], lowered: str) -> None:
    if "drug abuse" not in lowered:
        return
    out["diagnoses_any"] = [
        "drug abuse", "substance abuse", "cocaine", "heroin", "ivdu",
        "marijuana abuse", "narcotic abuse", "opiate abuse", "opioid abuse",
        "polysubstance", "intravenous drug",
    ]
    out.pop("diagnoses", None)


def _fix_mi_6mos(out: Dict[str, Any], lowered: str) -> None:
    if "myocardial infarction" not in lowered or "6 month" not in lowered:
        return
    out.pop("diagnoses", None)
    out.pop("medications", None)
    out.pop("time_conditions", None)
    out["temporal_event"] = {
        "event_terms": ["myocardial infarction", "acute mi", "s/p mi", "stemi", "nstemi"],
        "window_months": 6,
    }
    out["diagnoses_any"] = ["myocardial infarction", "acute mi", "s/p mi"]


def _fix_keto_1yr(out: Dict[str, Any], lowered: str) -> None:
    if "ketoacidosis" not in lowered:
        return
    out.pop("diagnoses", None)
    out.pop("time_conditions", None)
    out["temporal_event"] = {
        "event_terms": ["ketoacidosis", "dka", "diabetic ketoacidosis"],
        "window_years": 1,
    }
    out["diagnoses_any"] = ["ketoacidosis", "dka", "diabetic ketoacidosis"]


def _fix_dietsupp_2mos(out: Dict[str, Any], lowered: str) -> None:
    if "dietary supplement" not in lowered:
        return
    out.pop("diagnoses", None)
    out.pop("time_conditions", None)
    out["temporal_event"] = {
        "event_terms": list(SUPPLEMENT_TERMS),
        "window_months": 2,
        "exclude_vitamin_d_only": True,
    }
    out["diagnoses_any"] = ["dietary supplement", "herbal", "multivitamin", "supplement"]
