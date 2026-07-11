from app.l1_prompt import UNIFIED_JSON_TEMPLATE, build_l1_prompt


def test_unified_prompt_is_zero_shot_and_reasoning_free():
    criterion = "HbA1c between 6.5% and 9.5%."
    system, user = build_l1_prompt(criterion, variant="unified")

    combined = f"{system}\n{user}"
    assert criterion in user
    assert "_reasoning" not in combined
    assert "示例" not in combined
    assert "Example" not in combined
    assert "思维链" not in combined
    assert "Do not output analysis" in system
    assert "Criterion to parse:" in user


def test_unified_template_has_only_structured_fields():
    assert "_reasoning" not in UNIFIED_JSON_TEMPLATE
    assert '"diagnoses":[]' in UNIFIED_JSON_TEMPLATE
    assert '"lab_tests":[]' in UNIFIED_JSON_TEMPLATE
