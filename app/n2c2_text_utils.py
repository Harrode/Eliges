"""Shared n2c2 note text helpers for B1 indexing and rule filtering."""
import re
from typing import Dict, List


def extract_numeric_labs(text: str) -> Dict[str, List[float]]:
    lowered = text.lower()
    labs: Dict[str, List[float]] = {"hba1c": [], "creatinine": []}
    hba1c_patterns = [
        r"(?:hba1c|a1c|hemoglobin a1c|glycohemoglobin|glycosylated hemoglobin)[^\d]{0,25}(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*%?\s*(?:hba1c|a1c|hemoglobin a1c|glycohemoglobin)",
    ]
    creatinine_patterns = [
        r"(?:creatinine|creat\.?)[^\d]{0,25}(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*(?:mg/dl)?\s*(?:creatinine|creat\.)",
    ]
    for pattern in hba1c_patterns:
        labs["hba1c"].extend(float(m.group(1)) for m in re.finditer(pattern, lowered))
    for pattern in creatinine_patterns:
        for match in re.finditer(pattern, lowered):
            value = float(match.group(1))
            if 0.2 <= value <= 12:
                labs["creatinine"].append(value)
    return labs


def build_es_lab_results(text: str) -> List[Dict]:
    """Build nested lab_results entries for Elasticsearch patient documents."""
    extracted = extract_numeric_labs(text)
    results: List[Dict] = []
    for value in extracted["hba1c"]:
        results.append({"name": "HbA1c", "value": value})
    for value in extracted["creatinine"]:
        results.append({"name": "Creatinine", "value": value})
    return results


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def is_negated_english(sentence: str, term: str) -> bool:
    lowered = sentence.lower()
    idx = lowered.find(term.lower())
    if idx < 0:
        return False
    window = lowered[max(0, idx - 55): idx + len(term) + 20]
    negators = [
        "no ", "not ", "without ", "denies ", "denied ", "denying ",
        "negative for ", "free of ", "no history of ", "never ",
        "non-", "none",
    ]
    return any(negator in window for negator in negators)


def has_positive_english_term(text: str, terms: List[str]) -> bool:
    for sentence in split_sentences(text):
        lowered = sentence.lower()
        for term in terms:
            if term.lower() in lowered and not is_negated_english(lowered, term):
                return True
    return False


def has_any_substring(text: str, terms: List[str]) -> bool:
    lowered = normalize_text(text)
    return any(term.lower() in lowered for term in terms)


def count_cad_evidence(text: str, cad_meds: List[str] = None) -> int:
    evidence = 0
    if cad_meds is None:
        try:
            from app.b1_drug_terms import cad_medication_terms
            cad_meds = list(cad_medication_terms())
        except Exception:
            cad_meds = [
                "aspirin", "asa", "plavix", "clopidogrel", "nitroglycerin",
                "atenolol", "metoprolol", "lipitor", "atorvastatin", "pravastatin",
                "simvastatin", "zocor", "beta blocker", "statin",
            ]
    lowered = normalize_text(text)
    med_hits = sum(1 for term in cad_meds if term.lower() in lowered)
    if med_hits >= 2:
        evidence += 1
    if has_positive_english_term(text, ["myocardial infarction", " mi ", "history of mi", "s/p mi"]):
        evidence += 1
    if has_positive_english_term(text, ["angina", "chest pain"]):
        evidence += 1
    if has_positive_english_term(text, ["ischemia", "ischemic", "positive stress test", "cabg", "ptca", "pci", "stent"]):
        evidence += 1
    return evidence


def extract_record_years(text: str) -> List[int]:
    years = []
    for match in re.finditer(r"record date:\s*(\d{4})-\d{2}-\d{2}", text, re.IGNORECASE):
        years.append(int(match.group(1)))
    return years


def has_recent_event(
    text: str,
    event_terms: List[str],
    years: float = 1.0,
    window_months: int = None,
) -> bool:
    """Temporal check using n2c2 Record date sections (relative order preserved)."""
    if window_months is not None:
        years = max(window_months / 12.0, years)

    sections = re.split(r"(?=Record date:\s*\d{4}-\d{2}-\d{2})", text, flags=re.IGNORECASE)
    latest_years = extract_record_years(text)
    if not latest_years:
        return has_positive_english_term(text, event_terms)

    cutoff = max(latest_years) - years
    for section in sections:
        match = re.search(r"record date:\s*(\d{4})-\d{2}-\d{2}", section, re.IGNORECASE)
        if not match or int(match.group(1)) < cutoff:
            continue
        if has_positive_english_term(section, event_terms):
            return True
    return False


def dietsupp_met(text: str, supplement_terms: List[str], window_months: int = 2) -> bool:
    """Match B2 DIETSUPP-2MOS logic: recent supplement use, excluding vitamin-D-only."""
    vitamin_d_only = has_any_substring(text, ["vitamin d"]) and not has_any_substring(text, supplement_terms)
    if vitamin_d_only:
        return False
    return has_recent_event(text, supplement_terms, window_months=window_months)
