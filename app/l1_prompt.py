"""L1 NLU prompt variants — zero-shot unified + legacy ablations."""
from __future__ import annotations

from typing import Dict

# Unified superset schema: Chia entity fields + n2c2 retrieval/composite fields
UNIFIED_JSON_TEMPLATE = (
    '{"diagnoses":[],"diagnoses_excluded":[],"conditions_excluded":[],'
    '"inclusion_terms":[],"inclusion_negative_terms":[],"inclusion_logic":null,'
    '"medications":[],"medications_excluded":[],"required_medications":[],'
    '"procedures":[],"symptoms":[],"lab_tests":[],'
    '"temporal_event":null,"time_conditions":[],'
    '"evidence_groups":[],"composite_min_count":null,'
    '"age_min":null,"age_max":null,"gender":null}'
)

# ── English zero-shot unified prompt (no ICL examples, no _reasoning field) ──

UNIFIED_ZERO_SHOT_GUIDE = (
    "Extract fields directly from the criterion. Do not output analysis, "
    "explanation, or reasoning text.\n"
    "Identify inclusion, exclusion, demographic, laboratory, temporal, and "
    "composite constraints.\n"
    "Split alternatives and comma-separated lists into short entities of "
    "1-6 words; each entity must be grounded in the source text.\n"
    "Populate only supported fields; output one JSON object only."
)

UNIFIED_FIELD_RULES = (
    "Field rules (apply to any trial dataset; aligned with Chia/OMOP semantic views):\n"
    "- diagnoses / diagnoses_excluded / conditions_excluded / medications / "
    "procedures / symptoms: lists of short entities.\n"
    "- medications: drug names only; do not put consent, procedures, or other "
    "non-drug terms here.\n"
    "- lab_tests: [{test, op, value|value_min|value_max}]; standardize test "
    "names; numeric values must be numbers.\n"
    "- age_min / age_max: parse forms such as >=18, <=60, 18-45 years, "
    "age 18 and over.\n"
    "- gender: fill only when the criterion requires a single sex (male or "
    "female); omit when the text says male or female.\n"
    "- temporal_event: {event_terms, window_months|window_years} for events "
    "within N months/years.\n"
    "- time_conditions: fill when the text contains history / recent / "
    "previous / within / continuous / concomitant / undergoing.\n"
    "- inclusion_terms + inclusion_logic + composite_min_count: composite "
    "OR/AND criteria.\n"
    "- inclusion_logic: language, decision, or behavioral-capacity criteria; "
    "do not encode language requirements as gender.\n"
    "General constraints:\n"
    "- Every list entity must be grounded in the source text; do not output "
    "full sentences.\n"
    "- Behavioral or language criteria use inclusion_logic or inclusion_terms, "
    "not lab_tests.\n"
    "- Simple inclusion diagnoses go in diagnoses; exclusions go in *_excluded "
    "fields."
)

# Legacy English blocks (ablation variants only)
CHIA_JSON_TEMPLATE = (
    '{"diagnoses":[],"diagnoses_excluded":[],"conditions_excluded":[],'
    '"medications":[],"medications_excluded":[],"procedures":[],'
    '"symptoms":[],"lab_tests":[],"time_conditions":[],'
    '"age_min":null,"age_max":null,"gender":null}'
)

CHIA_ENTITY_RULES = (
    "Rules (Chia/OMOP entity style):\n"
    "- Output short clinical entities (1-6 words), NOT full sentences.\n"
    "- Split 'A or B' / comma lists into separate strings.\n"
    "- diagnoses: conditions/diseases; diagnoses_excluded: excluded diagnoses.\n"
    "- conditions_excluded: states like pregnancy, allergy (when exclusion).\n"
    "- medications: drug names only; procedures: surgery/procedure names.\n"
    "- lab_tests: [{test, op, value|value_min|value_max}] with standard test names.\n"
    "- age_min/age_max/gender: extract when explicit.\n"
    "- Do NOT use inclusion_logic, temporal_event, or diagnoses_any.\n"
    "- Omit empty fields; output only JSON."
)

CHIA_FEW_SHOTS = """
Example 1 (exclusion drugs):
Input: Known hypersensitivity to tetracycline or doxycycline.
Output: {"medications_excluded":["tetracycline","doxycycline"]}

Example 2 (age + inclusion):
Input: Age >= 65 years and < 90 years.
Output: {"age_min":65,"age_max":90}

Example 3 (lab):
Input: Serum creatinine >= 1.5 mg/dL.
Output: {"lab_tests":[{"test":"creatinine","op":">=","value":1.5}]}

Example 4 (condition OR list):
Input: History of myocardial infarction, angina, or ischemia.
Output: {"diagnoses":["myocardial infarction","angina","ischemia"]}
"""


def build_l1_prompt(
    query: str,
    variant: str = "default",
    feedback: str = "",
) -> tuple[str, str]:
    """Return (system_message, user_message)."""
    variant = variant or "default"

    if variant in ("default", "unified"):
        system = (
            "You are a structured-information extractor for clinical-trial "
            "eligibility criteria. Extract structured JSON fields from one "
            "criterion; the schema applies across datasets. Do not output "
            "analysis, explanation, or Markdown. Output the JSON object only."
        )
        user = (
            f"{UNIFIED_ZERO_SHOT_GUIDE}\n\n"
            f"{UNIFIED_FIELD_RULES}\n\n"
            f"Criterion to parse:\n{query}\n"
            f"-> {UNIFIED_JSON_TEMPLATE}"
        )
    elif variant == "chia_v1":
        system = "You extract structured eligibility fields from clinical trial criteria."
        user = (
            f"{CHIA_ENTITY_RULES}\n"
            f"Parse: {query}\n"
            f"-> {CHIA_JSON_TEMPLATE}"
        )
    elif variant == "chia_v2":
        system = "You extract structured eligibility fields from clinical trial criteria."
        user = (
            f"{CHIA_FEW_SHOTS}\n"
            f"{CHIA_ENTITY_RULES}\n"
            f"Parse: {query}\n"
            f"-> {CHIA_JSON_TEMPLATE}"
        )
    elif variant == "chia_v3":
        system = "You extract structured eligibility fields from clinical trial criteria."
        user = (
            f"{CHIA_FEW_SHOTS}\n"
            f"{CHIA_ENTITY_RULES}\n"
            f"Parse: {query}\n"
            f"-> {CHIA_JSON_TEMPLATE}"
        )
    elif variant == "chia_hybrid":
        raise ValueError("chia_hybrid is eval-only; use chia_v2 + rule merge in pipeline")
    else:
        raise ValueError(f"Unknown L1 prompt variant: {variant}")

    if feedback:
        user = f"{user}\n{feedback}"
    return system, user


def list_prompt_variants() -> Dict[str, str]:
    return {
        "unified": "English universal zero-shot: Chia entities + n2c2 temporal/composite + rule fusion",
        "default": "Alias of unified",
        "chia_v1": "Legacy: Chia/OMOP schema + short entity rules",
        "chia_v2": "Legacy: chia_v1 + 4 few-shot examples",
        "chia_v3": "Legacy: chia_v2 + entity compaction post-process",
        "chia_hybrid": "Eval-only: chia_v2 LLM + rule parser union",
    }
