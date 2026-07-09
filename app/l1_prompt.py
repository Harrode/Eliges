"""L1 NLU prompt variants — unified (中文) + legacy ablation variants."""
from __future__ import annotations

from typing import Dict

# Unified superset schema: Chia entity fields + n2c2 retrieval/composite fields
UNIFIED_JSON_TEMPLATE = (
    '{"_reasoning":"","diagnoses":[],"diagnoses_excluded":[],"conditions_excluded":[],'
    '"inclusion_terms":[],"inclusion_negative_terms":[],"inclusion_logic":null,'
    '"medications":[],"medications_excluded":[],"required_medications":[],'
    '"procedures":[],"symptoms":[],"lab_tests":[],'
    '"temporal_event":null,"time_conditions":[],'
    '"evidence_groups":[],"composite_min_count":null,'
    '"age_min":null,"age_max":null,"gender":null}'
)

# ── 中文 unified prompt（泛化：Chia OMOP 实体 + n2c2 时间/复合/检验）──

UNIFIED_COT_GUIDE_ZH = (
    "思维链（必填）：在 _reasoning 中用 2-4 步中文简述，再填字段。\n"
    "  1) 纳入 / 排除 / 人口学 / 检验 / 时间窗？\n"
    "  2) 有哪些实体类型（诊断、药、手术、症状、检验、年龄、性别、时间、复合条件）？\n"
    "  3) 将「A 或 B」、逗号列表拆成 1-6 词的短实体；每个实体必须是原文短语。\n"
    "  4) 只输出有内容的字段；不要臆造。"
)

UNIFIED_FIELD_RULES_ZH = (
    "字段规则（适用任意试验数据集，对齐 Chia/OMOP 语义视图）：\n"
    "- diagnoses / diagnoses_excluded / conditions_excluded / medications / procedures / symptoms：短实体列表。\n"
    "- medications：仅药名；不要把 consent、procedure、sign 等非药物写入。\n"
    "- lab_tests：[{test, op, value|value_min|value_max}]；检验名标准化，数值必须为数字。\n"
    "- age_min / age_max：识别 ≥18、<=60、18-45 years、age 18 and over 等写法。\n"
    "- gender：仅当原文只要求单一性别（male 或 female）；「male or female」时不填 gender。\n"
    "- temporal_event：{event_terms, window_months|window_years}，用于「N 月/年内发生过 X」。\n"
    "- time_conditions：原文含 history/recent/previous/within/continuous/concomitant/undergoing 等时间语义时填写。\n"
    "- inclusion_terms + inclusion_logic + composite_min_count：复合 OR/AND 标准。\n"
    "- inclusion_logic：语言/决策/行为能力类标准；不要用 gender 表示语言要求。\n"
    "泛化约束：\n"
    "- 每个列表实体应能在原文中找到依据；不要输出整句。\n"
    "- 行为/语言类标准用 inclusion_logic 或 inclusion_terms，不要写入 lab_tests。\n"
    "- 简单纳入条件用 diagnoses；排除条件写入 *_excluded 字段。"
)

UNIFIED_FEW_SHOTS_ZH = """
示例1（排除药物）：
原文: Known hypersensitivity to tetracycline or doxycycline.
输出: {"_reasoning":"1)排除 2)药物过敏 3)拆分药名","medications_excluded":["tetracycline","doxycycline"]}

示例2（年龄区间，含 Unicode 符号）：
原文: Age >= 65 years and < 90 years.
输出: {"_reasoning":"1)人口学 2)年龄上下界","age_min":65,"age_max":90}

示例3（检验区间）：
原文: Any HbA1c value between 6.5% and 9.5%.
输出: {"_reasoning":"1)检验纳入 2)区间","lab_tests":[{"test":"HbA1c","op":"between","value_min":6.5,"value_max":9.5}]}

示例4（时间窗诊断）：
原文: Myocardial infarction in the past 6 months.
输出: {"_reasoning":"1)纳入+时间 2)事件MI 6月","diagnoses":["myocardial infarction"],"temporal_event":{"event_terms":["myocardial infarction"],"window_months":6},"time_conditions":[{"operator":"temporal"}]}

示例5（复合 OR，需 2+ 项）：
原文: Advanced CAD: use of 2+ of nitrates, beta-blockers, statins, aspirin, or history of MI, angina, ischemia.
输出: {"_reasoning":"1)复合 2)OR列表 3)至少2项","inclusion_terms":["nitrates","beta-blockers","statins","aspirin","myocardial infarction","angina","ischemia"],"inclusion_logic":"OR","composite_min_count":2,"time_conditions":[{"operator":"temporal"}]}

示例6（诊断 OR 列表）：
原文: History of myocardial infarction, angina, or ischemia.
输出: {"_reasoning":"1)诊断纳入 2)同义词拆分","diagnoses":["myocardial infarction","angina","ischemia"]}

示例7（必须用药）：
原文: Use of aspirin to prevent myocardial infarction.
输出: {"_reasoning":"1)必须用药 2)适应证","required_medications":["aspirin"],"diagnoses":["myocardial infarction"]}

示例8（排除状态 + 双性别不写 gender）：
原文: Male or female patients aged 18-65. Pregnant women are excluded.
输出: {"_reasoning":"1)年龄 2)排除妊娠 3)双性别不写gender","age_min":18,"age_max":65,"conditions_excluded":["pregnancy"]}

示例9（时间语义词）：
原文: Hospitalization was recent. No therapy change within 30 days.
输出: {"_reasoning":"1)时间recent 2)30天内","time_conditions":[{"operator":"temporal"}]}

示例10（检验阈值）：
原文: Estimated glomerular filtration rate below 40 ml/min.
输出: {"_reasoning":"1)检验排除 2)eGFR<40","lab_tests":[{"test":"estimated glomerular filtration rate","op":"<","value":40}]}
"""

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
            "你是临床试验纳排标准结构化抽取器。"
            "任务：从单条纳排标准中提取结构化 JSON 字段，适用于任意数据集。"
            "先在 _reasoning 中做简短中文思维链，再输出 JSON；仅输出 JSON 对象。"
        )
        user = (
            f"{UNIFIED_COT_GUIDE_ZH}\n\n"
            f"{UNIFIED_FIELD_RULES_ZH}\n\n"
            f"{UNIFIED_FEW_SHOTS_ZH}\n"
            f"待解析标准:\n{query}\n"
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
        "unified": "中文 universal: Chia 实体 + n2c2 时间/复合 + CoT + ICL + 规则融合",
        "default": "Alias of unified",
        "chia_v1": "Legacy: Chia/OMOP schema + short entity rules",
        "chia_v2": "Legacy: chia_v1 + 4 few-shot examples",
        "chia_v3": "Legacy: chia_v2 + entity compaction post-process",
        "chia_hybrid": "Eval-only: chia_v2 LLM + rule parser union",
    }
