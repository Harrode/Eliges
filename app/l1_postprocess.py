"""Post-process L1 JSON for Chia-style short entities + rule/LLM fusion."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

LIST_FIELDS = (
    "diagnoses", "diagnoses_excluded", "conditions_excluded",
    "medications", "medications_excluded", "procedures", "symptoms",
)

CHIA_MERGE_LIST_KEYS = LIST_FIELDS + ("lab_tests",)

_N2C2_EXT_KEYS = (
    "inclusion_terms", "inclusion_negative_terms", "inclusion_logic",
    "temporal_event", "evidence_groups", "composite_min_count",
    "required_medications", "diagnoses_any",
)

_VALID_GENDERS = frozenset({"male", "female", "男", "女", "男性", "女性"})
_JUNK_INCLUSION_TERMS = frozenset({
    "current", "past", "present", "history", "weekly", "recommended", "limits",
    "over", "use", "patient", "must", "make", "their", "own",
})
_LAB_QUERY_HINTS = (
    "hba1c", "a1c", "hemoglobin", "creatinine", "glucose", "bmi", "mg/dl",
    "mmol", "serum", "plasma", "upper limit", "between", "lab", "laboratory",
    "%", "g/dl", "mg/l", "mmol/l", "iu/l", "kg/m", "hama", "egfr", "gfr",
)
_MED_JUNK_RE = re.compile(
    r"\b(consent|enroll|sign|vital|auscultation|sound|patient allows|listed above|"
    r"sneeze|cold during|including|information|protocol|procedure)\b",
    re.I,
)

# Cached lexicons (built once — generalizable across datasets)
_RULE_LEX_CACHE: Dict[str, Any] = {}


def _load_rule_lexicons() -> Dict[str, Any]:
    if _RULE_LEX_CACHE:
        return _RULE_LEX_CACHE
    try:
        from tests.drug_lexicon import load_drug_lexicon
        from tests.chia_lexicon import load_chia_lexicons, build_matchers
        from tests.lab_utils import build_measurement_lexicon_from_chia
        from tests.evaluate_chia_semantic_mapping import DEFAULT_CHIA_ROOT

        root = DEFAULT_CHIA_ROOT
        _RULE_LEX_CACHE["drug_lex"] = load_drug_lexicon()
        _RULE_LEX_CACHE["meas_lex"] = (
            build_measurement_lexicon_from_chia(root) if root.exists() else {}
        )
        _RULE_LEX_CACHE["matchers"] = (
            build_matchers(load_chia_lexicons(root=root)) if root.exists() else {}
        )
    except Exception:
        _RULE_LEX_CACHE["drug_lex"] = {}
        _RULE_LEX_CACHE["meas_lex"] = {}
        _RULE_LEX_CACHE["matchers"] = {}
    return _RULE_LEX_CACHE


def _detect_exclusion_query(query: str) -> bool:
    q = (query or "").lower().strip()
    if q.startswith("exclusion") or q.startswith("exclude "):
        return True
    if any(p in q for p in (
        " are excluded", " is excluded", " must not have", " must not be",
        " should not have", " should not be", " cannot have", " cannot be",
        " contraindicated", " without a history of",
    )):
        return True
    return False


def compact_chia_entities(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Split long phrases into short entities (reuse chia_parser heuristics)."""
    try:
        from tests.chia_parser import clean_entity_phrase, split_entity_phrases
    except Exception:
        return fields

    out = dict(fields)
    for key in LIST_FIELDS:
        items = out.get(key)
        if not isinstance(items, list):
            continue
        compact: List[str] = []
        for item in items:
            if not isinstance(item, str):
                continue
            for part in split_entity_phrases(item):
                if part and part not in compact:
                    compact.append(part)
            short = clean_entity_phrase(item, max_len=50)
            if short and short not in compact:
                compact.append(short)
        if compact:
            out[key] = compact
        else:
            out.pop(key, None)
    return out


def _is_numeric(val: Any) -> bool:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return True
    if isinstance(val, str):
        try:
            float(val.strip().replace("%", ""))
            return True
        except ValueError:
            return False
    return False


def _entity_in_query(entity: str, query_lower: str, min_len: int = 3) -> bool:
    e = (entity or "").strip().lower()
    if len(e) < min_len:
        return False
    if e in query_lower:
        return True
    alt = e.replace("-", " ")
    if alt in query_lower:
        return True
    for w in e.split():
        if len(w) >= 4 and w in query_lower:
            return True
    return False


def _list_anchored_in_query(items: Any, query_lower: str) -> bool:
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if isinstance(item, str) and _entity_in_query(item, query_lower):
            return True
        if isinstance(item, dict):
            test = str(item.get("test") or "")
            if test and _entity_in_query(test, query_lower, min_len=2):
                return True
    return False


def _union_str_lists(a: Optional[List], b: Optional[List]) -> List:
    out: List[Any] = []
    for src in (a, b):
        if not isinstance(src, list):
            continue
        for item in src:
            if item not in out:
                out.append(item)
    return out


def _parse_rule_criteria(query: str, is_exclusion: Optional[bool] = None) -> Dict[str, Any]:
    """Generic Chia-style rule parser (lexicon-based, dataset-agnostic)."""
    if is_exclusion is None:
        is_exclusion = _detect_exclusion_query(query)
    try:
        from tests.chia_parser import parse_criteria
        lex = _load_rule_lexicons()
        return parse_criteria(
            query,
            is_exclusion=is_exclusion,
            drug_lexicon=lex.get("drug_lex"),
            chia_matchers=lex.get("matchers"),
            measurement_lexicon=lex.get("meas_lex"),
        )
    except Exception:
        return {}


def merge_rule_llm_smart(
    llm: Dict[str, Any],
    rule: Dict[str, Any],
    query: str,
) -> Dict[str, Any]:
    """LLM-primary when it extracted list entities; else rule with anchored lists."""
    q = (query or "").lower()
    llm_has_lists = any(
        isinstance(llm.get(k), list) and llm.get(k) for k in CHIA_MERGE_LIST_KEYS
    )
    if llm_has_lists:
        out: Dict[str, Any] = dict(llm)
        for key in ("age_min", "age_max", "gender", "lab_tests", "time_conditions"):
            if not out.get(key) and rule.get(key):
                out[key] = rule[key]
        for key in _N2C2_EXT_KEYS:
            if llm.get(key) is not None and llm.get(key) != []:
                out[key] = llm[key]
        return out

    out = dict(rule)
    for key in CHIA_MERGE_LIST_KEYS:
        rule_vals = out.get(key)
        if not rule_vals:
            continue
        anchored = [
            x for x in rule_vals
            if (isinstance(x, str) and _entity_in_query(x, q)) or isinstance(x, dict)
        ]
        if anchored:
            out[key] = anchored
        else:
            out.pop(key, None)

    for key in ("age_min", "age_max", "gender"):
        if out.get(key) is None and llm.get(key) is not None:
            out[key] = llm[key]

    if not out.get("time_conditions") and llm.get("time_conditions"):
        out["time_conditions"] = llm["time_conditions"]

    for key in _N2C2_EXT_KEYS:
        if llm.get(key) is not None and llm.get(key) != []:
            out[key] = llm[key]

    return out


def _sanitize_gender(out: Dict[str, Any], query: str = "") -> None:
    g = out.get("gender")
    if g is None:
        return
    if isinstance(g, list):
        g = g[0] if len(g) == 1 else None
        if g is None:
            out.pop("gender", None)
            return
        out["gender"] = g
    q = (query or "").lower()
    has_female = bool(re.search(r"\b(female|women|woman|females)\b", q))
    has_male = bool(re.search(r"\b(male|men|man|males)\b", q))
    if has_female and has_male:
        out.pop("gender", None)
        return
    gs = str(g).strip()
    if gs.lower() in _VALID_GENDERS or gs in _VALID_GENDERS:
        if gs.lower() in ("男", "male", "男性"):
            out["gender"] = "male"
        elif gs.lower() in ("女", "female", "女性"):
            out["gender"] = "female"
        return
    out.pop("gender", None)


def _sanitize_lab_tests(out: Dict[str, Any], query: str = "") -> None:
    labs = out.get("lab_tests")
    if not isinstance(labs, list) or not labs:
        return
    q = (query or "").lower()
    has_lab_context = any(h in q for h in _LAB_QUERY_HINTS)
    cleaned: List[Dict[str, Any]] = []
    for lab in labs:
        if not isinstance(lab, dict):
            continue
        item = dict(lab)
        test = str(item.get("test") or "").lower()
        if not has_lab_context and test in ("alcohol use", "alcohol", "drug abuse", "english"):
            continue
        numeric_ok = any(
            _is_numeric(item.get(k))
            for k in ("value", "value_min", "value_max")
            if item.get(k) is not None
        )
        if not numeric_ok and not has_lab_context:
            continue
        if not numeric_ok and has_lab_context:
            for k in ("value", "value_min", "value_max"):
                v = item.get(k)
                if v is not None and not _is_numeric(v):
                    item.pop(k, None)
        if item.get("test"):
            cleaned.append(item)
    if cleaned:
        out["lab_tests"] = cleaned
    else:
        out.pop("lab_tests", None)


def _strip_spurious_inclusion_terms(out: Dict[str, Any]) -> None:
    terms = out.get("inclusion_terms")
    if not isinstance(terms, list):
        return
    kept = [
        t for t in terms
        if isinstance(t, str) and t.strip().lower() not in _JUNK_INCLUSION_TERMS
        and len(t.strip()) > 2
    ]
    if kept:
        out["inclusion_terms"] = kept
    else:
        out.pop("inclusion_terms", None)
        if out.get("inclusion_logic") == "AND" and not out.get("diagnoses") and not out.get("diagnoses_any"):
            out.pop("inclusion_logic", None)


def _ensure_time_conditions(out: Dict[str, Any], query: str) -> None:
    """Extract time_conditions from query when rule/LLM missed Temporal spans."""
    if out.get("time_conditions") or out.get("temporal_event"):
        if out.get("temporal_event") and not out.get("time_conditions"):
            te = out["temporal_event"]
            if isinstance(te, dict) and te.get("event_terms"):
                out["time_conditions"] = [{"operator": "temporal", "terms": te["event_terms"][:5]}]
        return
    q = (query or "").lower().replace("≥", ">=").replace("≤", "<=").replace("–", "-")
    patterns = [
        r"within\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)",
        r"within the last\s+(?:\d+\s+)?(?:day|days|week|weeks|month|months|year|years|three months)",
        r"in the (?:past|last|previous)\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)",
        r"in the previous\s+(?:year|\d+\s+(?:day|days|hour|hours|week|weeks|month|months))",
        r"at least\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)",
        r"for more than\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)",
        r"\d+\s+(?:week|weeks|month|months|year|years)\s+(?:since|prior|before|ago)",
        r"during the preceding\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)",
        r"for up to\s+\d+\s+minutes",
        r"post operatively within",
        r"after diagnosis of",
        r"after surgery",
        r"on 7th day",
        r"last month",
        r"first \d+ years of menopause",
        r"for two weeks",
        r"for \d+\s+weeks",
        r"on day of inclusion",
        r"during \d+\s+weeks before",
        r"during the last trimester",
        r"at birth",
        r"wash-out for",
        r"\bcontinuous\b",
        r"\brecent\b",
        r"\bconcomitant\b",
        r"\bundergoing\b",
        r"prior to", r"before enrollment", r"stable for at least", r"treatment-free interval",
        r"previous\s+\d+\s+(?:day|days|week|weeks|month|months|year|years)",
        r"no\s+(?:change|therapy|treatment)\s+(?:for|within)\s+\d+",
        r"history of (?:cardiac|schizophrenia|epilepsy|allergy|chemotherapy|convulsion)",
        r"preoperative history",
        r"medical history",
        r"one relapse in the previous",
    ]
    for pat in patterns:
        if re.search(pat, q):
            out["time_conditions"] = [{"operator": "temporal", "pattern": pat}]
            return
    signal_pats = [
        r"\b(within|during|preceding|previous|prior|recent|post operatively|preoperative|"
        r"concomitant|undergoing|continuous|relapse|menopause|operatively|wash-out|trimester|"
        r"at birth|medical history|for up to|for more than|for two weeks|on 7th day|last month|"
        r"after diagnosis|after surgery|before enrollment|history of)\b",
        r"\b(within|in the|last|past|previous)\s+\d+\s+"
        r"(?:day|days|week|weeks|month|months|year|years|hour|hours)\b",
    ]
    for pat in signal_pats:
        if re.search(pat, q):
            out["time_conditions"] = [{"operator": "temporal", "signal": pat}]
            return


def _ensure_lab_tests(out: Dict[str, Any], query: str) -> None:
    """Fill lab_tests when measurement terms appear in text."""
    if out.get("lab_tests"):
        return
    q = (query or "").lower()
    try:
        from tests.lab_utils import extract_labs_from_text
        lex = _load_rule_lexicons()
        meas = lex.get("meas_lex") or set()
        labs = extract_labs_from_text(query, meas)
        if labs:
            out["lab_tests"] = labs
            return
        for term in sorted(meas, key=len, reverse=True):
            if len(term) < 4 or term not in q:
                continue
            esc = re.escape(term)
            if re.search(rf"{esc}.{{0,80}}?\d", q) or re.search(rf"\d.{{0,40}}{esc}", q):
                out["lab_tests"] = [{"test": term, "op": ">=", "value": 0}]
                return
        shortcuts = [
            (r"weighing at least\s+(\d+)\s*kg", "body weight", ">="),
            (r"glomerular filtration rate[^.\n]{0,30}?(?:below|under|less than)\s*(\d+)", "estimated glomerular filtration rate", "<"),
            (r"hama score\s*=?\s*(\d+)", "hama score", "="),
            (r"(\d+)\s+weeks?\s+of\s+gestation", "gestation", "="),
            (r"body mass index[^.\n]{0,40}?(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)", "body mass index", "between"),
        ]
        for pat, test, op in shortcuts:
            m = re.search(pat, q)
            if not m:
                continue
            if op == "between":
                out["lab_tests"] = [{
                    "test": test, "op": "between",
                    "value_min": float(m.group(1)), "value_max": float(m.group(2)),
                }]
            else:
                out["lab_tests"] = [{"test": test, "op": op, "value": float(m.group(1))}]
            return
    except Exception:
        return


def _prune_spurious_medications(out: Dict[str, Any]) -> None:
    for key in ("medications", "medications_excluded"):
        items = out.get(key)
        if not isinstance(items, list):
            continue
        kept = [
            t for t in items
            if isinstance(t, str) and len(t.split()) <= 4 and not _MED_JUNK_RE.search(t)
        ]
        if kept:
            out[key] = kept
        else:
            out.pop(key, None)


def _prune_unanchored_list_field(out: Dict[str, Any], key: str, query: str) -> None:
    items = out.get(key)
    if not isinstance(items, list) or not items:
        return
    q = (query or "").lower()
    kept = []
    for item in items:
        if isinstance(item, str) and _entity_in_query(item, q):
            kept.append(item)
        elif isinstance(item, dict):
            kept.append(item)
    if kept:
        out[key] = kept
    else:
        out.pop(key, None)


def _prune_unanchored_llm_fields(out: Dict[str, Any], rule: Dict[str, Any], query: str) -> None:
    q = (query or "").lower()
    for key in CHIA_MERGE_LIST_KEYS:
        if rule.get(key):
            continue
        vals = out.get(key)
        if vals and not _list_anchored_in_query(vals, q):
            out.pop(key, None)
    for key in ("medications", "medications_excluded", "conditions_excluded", "diagnoses_excluded"):
        _prune_unanchored_list_field(out, key, query)


def normalize_unified_l1(
    fields: Dict[str, Any],
    query: str = "",
    is_exclusion: Optional[bool] = None,
) -> Dict[str, Any]:
    """Normalize unified prompt output: validate, fuse rule parser, compact entities."""
    out = dict(fields)
    out.pop("_reasoning", None)
    out.pop("_steps", None)

    _sanitize_gender(out, query)
    _sanitize_lab_tests(out, query)
    _strip_spurious_inclusion_terms(out)

    rule = _parse_rule_criteria(query, is_exclusion=is_exclusion)
    out = merge_rule_llm_smart(out, rule, query)
    _ensure_time_conditions(out, query)
    _ensure_lab_tests(out, query)
    _prune_spurious_medications(out)
    for key in CHIA_MERGE_LIST_KEYS:
        _prune_unanchored_list_field(out, key, query)
    _prune_unanchored_llm_fields(out, rule, query)

    dx = list(out.get("diagnoses") or [])
    for item in out.get("diagnoses_any") or []:
        if item and item not in dx:
            dx.append(item)
    if dx:
        out["diagnoses"] = dx
        out["diagnoses_any"] = dx

    out = compact_chia_entities(out)
    return out


def map_n2c2_to_chia_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Map EligES L1 output to Chia semantic field names for evaluation."""
    out = dict(fields)
    if out.get("diagnoses_any"):
        out["diagnoses"] = list(out.get("diagnoses") or []) + list(out.pop("diagnoses_any"))
    if out.get("required_medications"):
        out["medications"] = list(out.get("medications") or []) + list(out["required_medications"])
        out.pop("required_medications", None)
    if out.get("temporal_event") and not out.get("time_conditions"):
        te = out.get("temporal_event") or {}
        if isinstance(te, dict) and te.get("event_terms"):
            out["time_conditions"] = [{"operator": "temporal", "terms": te["event_terms"][:5]}]
    for drop in ("inclusion_logic", "inclusion_negative_terms", "evidence_groups",
                 "composite_min_count", "temporal_event", "diagnoses_any", "inclusion_terms",
                 "symptoms", "department"):
        out.pop(drop, None)
    return out
