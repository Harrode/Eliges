"""n2c2 Track1 few-shot examples for B1 LLM prompt (English criteria)."""

TRACK1_FEW_SHOT_BLOCK = """
## n2c2 Track1 英文标准示例（13 条）

Track1-1 纳入-药物滥用
输入：Drug abuse, current or past.
输出：{"diagnoses_any":["drug abuse","substance abuse","cocaine","heroin","ivdu"]}

Track1-2 纳入-饮酒（不是排除！）
输入：Current alcohol use over weekly recommended limits.
输出：{"inclusion_terms":["alcohol abuse","heavy alcohol","daily alcohol"]}

Track1-3 纳入-英语
输入：Patient must speak English.
输出：{"inclusion_logic":"english_speaker"}

Track1-4 纳入-决策能力
输入：Patient must make their own medical decisions.
输出：{"inclusion_logic":"decision_capacity"}

Track1-5 纳入-腹部手术史（met=有该病史）
输入：History of intra-abdominal surgery, small or large intestine resection, or small bowel obstruction.
输出：{"inclusion_terms":["abdominal surgery","bowel resection","bowel obstruction","appendectomy"]}

Track1-6 纳入-糖尿病并发症（OR 关系）
输入：Major diabetes-related complication: amputation, kidney damage, skin conditions, retinopathy, nephropathy, or neuropathy.
输出：{"diagnoses_any":["amputation","nephropathy","neuropathy","retinopathy","kidney damage"]}

Track1-7 纳入-复合 CAD（至少 2 项）
输入：Advanced cardiovascular disease: two or more of CAD medications, myocardial infarction, angina, or ischemia.
输出：{"composite_min_count":2,"evidence_groups":[{"id":"cad_meds","mode":"med_count","min_hits":2},{"id":"mi","mode":"positive","terms":["myocardial infarction"]}]}

Track1-8 纳入-近期 MI（6 个月内）
输入：Myocardial infarction in the past 6 months.
输出：{"temporal_event":{"event_terms":["myocardial infarction","acute mi"],"window_months":6}}

Track1-9 纳入-近期酮症酸中毒（1 年内）
输入：Diagnosis of ketoacidosis in the past year.
输出：{"temporal_event":{"event_terms":["ketoacidosis","dka","diabetic ketoacidosis"],"window_years":1}}

Track1-10 纳入-膳食补充剂（2 个月内，排除仅维 D）
输入：Taken a dietary supplement, excluding vitamin D, in the past 2 months.
输出：{"temporal_event":{"event_terms":["dietary supplement","herbal","multivitamin"],"window_months":2,"exclude_vitamin_d_only":true}}

Track1-11 纳入-阿司匹林+MI/CAD
输入：Use of aspirin to prevent myocardial infarction.
输出：{"required_medications":["aspirin","asa"],"diagnoses_any":["myocardial infarction","coronary artery disease","cad"]}

Track1-12 纳入-HbA1c 区间
输入：Any hemoglobin A1c or HbA1c value between 6.5% and 9.5%.
输出：{"lab_tests":[{"test":"hemoglobin A1c","op":"between","value_min":6.5,"value_max":9.5}]}

Track1-13 纳入-肌酐高于 ULN
输入：Serum creatinine greater than the upper limit of normal.
输出：{"lab_tests":[{"test":"creatinine","op":">","value":1.3}]}
"""
