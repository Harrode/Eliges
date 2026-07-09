"""Read-only B3 drug lexicon helpers for B1 rule enrichment."""
from functools import lru_cache
from typing import List, Set


CAD_SEED_TERMS = [
    "aspirin", "asa", "plavix", "clopidogrel", "nitroglycerin",
    "atenolol", "metoprolol", "lipitor", "atorvastatin", "pravastatin",
    "simvastatin", "zocor", "carvedilol", "lisinopril", "losartan",
    "beta blocker", "statin",
]

PREVENTIVE_ASPIRIN_TERMS = ["aspirin", "asa"]


@lru_cache(maxsize=1)
def load_b3_drug_lexicon() -> Set[str]:
    from tests.drug_lexicon import load_drug_lexicon
    return load_drug_lexicon()


@lru_cache(maxsize=1)
def cad_medication_terms() -> tuple:
    """Expand CAD seed terms using B3 drug lexicon (read-only)."""
    lexicon = load_b3_drug_lexicon()
    terms = set(CAD_SEED_TERMS)
    for seed in CAD_SEED_TERMS:
        for term in lexicon:
            if seed in term:
                terms.add(term)
    return tuple(sorted(terms))


def match_medications_in_text(text: str, seeds: List[str]) -> List[str]:
    """Return seed/CAD drug hits found in note text via B3 lexicon matcher."""
    from tests.drug_lexicon import match_drugs_in_text

    lexicon = load_b3_drug_lexicon()
    hits = match_drugs_in_text(text, lexicon)
    matched = []
    lowered = text.lower()
    for seed in seeds:
        if seed.lower() in lowered:
            matched.append(seed)
    for hit in hits:
        for seed in seeds:
            if seed in hit or hit in seed:
                matched.append(hit)
                break
    return sorted(set(matched))
