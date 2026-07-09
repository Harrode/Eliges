"""
Initialize Elasticsearch index with mock patient EMR data.
Run: python scripts/init_es.py
"""

import json
from elasticsearch import Elasticsearch

ES_HOST = "http://localhost:9200"
ES_INDEX = "emr_records"

es = Elasticsearch(ES_HOST)

# Delete existing index
if es.indices.exists(index=ES_INDEX):
    es.indices.delete(index=ES_INDEX)
    print(f"Deleted existing index: {ES_INDEX}")

# Create index with mapping
mapping = {
    "mappings": {
        "properties": {
            "patient_id":      {"type": "keyword"},
            "name":            {"type": "keyword"},
            "age":             {"type": "integer"},
            "gender":          {"type": "keyword"},
            "diagnosis":       {"type": "text", "analyzer": "ik_max_word"},
            "chief_complaint": {"type": "text", "analyzer": "ik_max_word"},
            "history_present": {"type": "text", "analyzer": "ik_max_word"},
            "past_history":    {"type": "text", "analyzer": "ik_max_word"},
            "admission_record":{"type": "text", "analyzer": "ik_max_word"},
            "discharge_summary":{"type": "text", "analyzer": "ik_max_word"},
            "procedure_notes": {"type": "text", "analyzer": "ik_max_word"},
            "medications":     {"type": "text", "analyzer": "ik_max_word"},
            "department":      {"type": "keyword"},
            "hospital":        {"type": "keyword"},
            "admission_date":  {"type": "date", "format": "yyyy-MM-dd"},
            "discharge_date":  {"type": "date", "format": "yyyy-MM-dd"},
            "encounter_type":  {"type": "keyword"},
            "lab_results": {
                "type": "nested",
                "properties": {
                    "name":  {"type": "keyword"},
                    "value": {"type": "float"},
                    "unit":  {"type": "keyword"},
                }
            },
            "vital_signs": {
                "properties": {
                    "bp_systolic":  {"type": "integer"},
                    "bp_diastolic": {"type": "integer"},
                    "heart_rate":   {"type": "integer"},
                    "temperature":  {"type": "float"},
                }
            },
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
    }
}

# Try with ik_analyzer, fallback to standard if not installed
try:
    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"Created index with ik_max_word analyzer")
except Exception as e:
    print(f"ik_max_word not available, using standard analyzer: {e}")
    mapping["mappings"]["properties"]["diagnosis"]["analyzer"] = "standard"
    mapping["mappings"]["properties"]["chief_complaint"]["analyzer"] = "standard"
    mapping["mappings"]["properties"]["history_present"]["analyzer"] = "standard"
    mapping["mappings"]["properties"]["past_history"]["analyzer"] = "standard"
    mapping["mappings"]["properties"]["admission_record"]["analyzer"] = "standard"
    mapping["mappings"]["properties"]["discharge_summary"]["analyzer"] = "standard"
    mapping["mappings"]["properties"]["procedure_notes"]["analyzer"] = "standard"
    mapping["mappings"]["properties"]["medications"]["analyzer"] = "standard"
    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"Created index with standard analyzer")

# ── Mock Patient Data ───────────────────────────────────────────────────
patients = [
    {
        "patient_id": "P001",
        "name": "张三",
        "age": 45,
        "gender": "男",
        "diagnosis": "乙型病毒性肝炎, 肝硬化代偿期",
        "chief_complaint": "反复乏力、纳差3年，加重伴尿黄1周",
        "history_present": "患者3年前无明显诱因出现乏力、纳差，外院诊断为慢性乙型肝炎，长期服用恩替卡韦抗病毒治疗。1周前上述症状加重，伴尿色深黄，无腹痛腹泻。",
        "past_history": "高血压病史5年，口服氨氯地平控制。否认糖尿病、冠心病史。",
        "department": "消化内科",
        "hospital": "某某大学附属第一医院",
        "admission_date": "2024-06-15",
        "discharge_date": "2024-07-02",
        "encounter_type": "inpatient",
        "medications": "恩替卡韦 0.5mg qd, 氨氯地平 5mg qd",
        "procedure_notes": "腹部超声：肝脏弥漫性病变，脾脏轻度肿大",
        "lab_results": [
            {"name": "ALT", "value": 128.5, "unit": "U/L"},
            {"name": "AST", "value": 95.3, "unit": "U/L"},
            {"name": "总胆红素", "value": 42.1, "unit": "umol/L"},
            {"name": "白蛋白", "value": 32.8, "unit": "g/L"},
            {"name": "HBV-DNA", "value": 3500, "unit": "IU/mL"},
            {"name": "eGFR", "value": 88.2, "unit": "mL/min"},
        ],
        "vital_signs": {"bp_systolic": 135, "bp_diastolic": 85, "heart_rate": 78, "temperature": 36.5},
        "discharge_summary": "入院后完善检查，诊断为慢性乙型肝炎活动期、肝硬化代偿期。继续恩替卡韦抗病毒，辅以保肝降酶治疗。肝功能好转后出院。",
    },
    {
        "patient_id": "P002",
        "name": "李四",
        "age": 62,
        "gender": "男",
        "diagnosis": "2型糖尿病, 糖尿病肾病III期, 高血压3级",
        "chief_complaint": "口干多饮10年，双下肢水肿2月",
        "history_present": "患者10年前确诊2型糖尿病，口服二甲双胍+格列美脲控制血糖，近2年血糖控制不佳，糖化血红蛋白8.5%。2月前出现双下肢凹陷性水肿，尿泡沫增多。",
        "past_history": "高血压病史15年，血压控制不佳。否认肝炎、结核病史。",
        "department": "内分泌科",
        "hospital": "某某大学附属第一医院",
        "admission_date": "2024-05-20",
        "discharge_date": "2024-06-08",
        "encounter_type": "inpatient",
        "medications": "二甲双胍 500mg bid, 格列美脲 2mg qd, 缬沙坦 80mg qd",
        "procedure_notes": "肾脏超声：双肾弥漫性病变",
        "lab_results": [
            {"name": "空腹血糖", "value": 9.8, "unit": "mmol/L"},
            {"name": "糖化血红蛋白", "value": 8.5, "unit": "%"},
            {"name": "肌酐", "value": 156.0, "unit": "umol/L"},
            {"name": "尿素氮", "value": 12.3, "unit": "mmol/L"},
            {"name": "eGFR", "value": 42.5, "unit": "mL/min"},
            {"name": "尿蛋白定量", "value": 2.8, "unit": "g/24h"},
        ],
        "vital_signs": {"bp_systolic": 165, "bp_diastolic": 95, "heart_rate": 82, "temperature": 36.4},
        "discharge_summary": "入院后调整降糖方案为胰岛素联合口服药，加用SGLT2抑制剂保护肾功能。血压控制达标后出院，嘱定期复查肾功能。",
    },
    {
        "patient_id": "P003",
        "name": "王五",
        "age": 55,
        "gender": "男",
        "diagnosis": "冠状动脉粥样硬化性心脏病, 不稳定性心绞痛, 陈旧性心肌梗死",
        "chief_complaint": "反复胸闷胸痛2年，再发加重3天",
        "history_present": "患者2年前因急性前壁心肌梗死行PCI术（左前降支植入支架1枚），术后规律服用双抗及他汀。3天前活动后再次出现胸闷，伴左肩放射痛，含服硝酸甘油可缓解。",
        "past_history": "高血压病史10年，2型糖尿病史5年。吸烟30年，20支/日。",
        "department": "心内科",
        "hospital": "某某大学附属第二医院",
        "admission_date": "2024-07-01",
        "discharge_date": "2024-07-15",
        "encounter_type": "inpatient",
        "medications": "阿司匹林 100mg qd, 氯吡格雷 75mg qd, 阿托伐他汀 20mg qn, 美托洛尔 47.5mg qd",
        "procedure_notes": "冠脉造影：左前降支支架内再狭窄60%，回旋支狭窄50%",
        "lab_results": [
            {"name": "肌钙蛋白I", "value": 0.08, "unit": "ng/mL"},
            {"name": "BNP", "value": 450, "unit": "pg/mL"},
            {"name": "LDL-C", "value": 2.8, "unit": "mmol/L"},
            {"name": "肌酐", "value": 98.0, "unit": "umol/L"},
            {"name": "eGFR", "value": 72.1, "unit": "mL/min"},
        ],
        "vital_signs": {"bp_systolic": 142, "bp_diastolic": 88, "heart_rate": 76, "temperature": 36.6},
        "discharge_summary": "入院后行冠脉造影，支架内轻度狭窄，药物保守治疗。加强抗血小板、调脂治疗，症状缓解后出院。",
    },
    {
        "patient_id": "P004",
        "name": "赵六",
        "age": 38,
        "gender": "女",
        "diagnosis": "系统性红斑狼疮, 狼疮性肾炎IV型",
        "chief_complaint": "面部红斑反复发作3年，蛋白尿1年",
        "history_present": "患者3年前出现面部蝶形红斑，伴关节疼痛，外院诊断系统性红斑狼疮。1年前发现蛋白尿，肾活检提示狼疮性肾炎IV型。目前口服泼尼松+霉酚酸酯。",
        "past_history": "否认高血压、糖尿病史。对磺胺类药物过敏。",
        "department": "肾内科",
        "hospital": "某某大学附属第一医院",
        "admission_date": "2024-04-10",
        "discharge_date": "2024-04-28",
        "encounter_type": "inpatient",
        "medications": "泼尼松 30mg qd, 霉酚酸酯 750mg bid, 羟氯喹 200mg bid",
        "procedure_notes": "肾活检：狼疮性肾炎IV型，活动性指数8/24，慢性指数3/12",
        "lab_results": [
            {"name": "肌酐", "value": 112.0, "unit": "umol/L"},
            {"name": "eGFR", "value": 58.3, "unit": "mL/min"},
            {"name": "尿蛋白定量", "value": 3.5, "unit": "g/24h"},
            {"name": "C3", "value": 0.45, "unit": "g/L"},
            {"name": "C4", "value": 0.08, "unit": "g/L"},
            {"name": "抗dsDNA抗体", "value": 185, "unit": "IU/mL"},
        ],
        "vital_signs": {"bp_systolic": 128, "bp_diastolic": 78, "heart_rate": 88, "temperature": 37.2},
        "discharge_summary": "入院后完善免疫指标评估，调整免疫抑制方案，加用贝利尤单抗。症状改善、尿蛋白下降后出院。",
    },
    {
        "patient_id": "P005",
        "name": "钱七",
        "age": 70,
        "gender": "男",
        "diagnosis": "慢性阻塞性肺疾病急性加重, II型呼吸衰竭",
        "chief_complaint": "反复咳嗽咳痰20年，气促加重1周",
        "history_present": "患者20年来反复咳嗽、咳白色黏痰，冬季加重。1周前受凉后咳黄痰，活动后气促明显，平地行走即感呼吸困难。",
        "past_history": "吸烟40年，30支/日，已戒5年。否认结核病史。",
        "department": "呼吸内科",
        "hospital": "某某大学附属第二医院",
        "admission_date": "2024-03-05",
        "discharge_date": "2024-03-25",
        "encounter_type": "inpatient",
        "medications": "沙美特罗替卡松吸入剂 bid, 噻托溴铵吸入剂 qd, 阿莫西林克拉维酸钾 0.375g tid",
        "procedure_notes": "胸部CT：双肺透亮度增高，双下肺感染灶",
        "lab_results": [
            {"name": "WBC", "value": 12.8, "unit": "x10^9/L"},
            {"name": "CRP", "value": 68.5, "unit": "mg/L"},
            {"name": "PaO2", "value": 55, "unit": "mmHg"},
            {"name": "PaCO2", "value": 62, "unit": "mmHg"},
            {"name": "FEV1", "value": 42.0, "unit": "%pred"},
        ],
        "vital_signs": {"bp_systolic": 138, "bp_diastolic": 82, "heart_rate": 96, "temperature": 37.8},
        "discharge_summary": "入院后抗感染、解痉平喘、无创通气支持。感染控制、血气改善后出院，嘱家庭氧疗。",
    },
    {
        "patient_id": "P006",
        "name": "孙八",
        "age": 52,
        "gender": "女",
        "diagnosis": "乳腺癌（左乳浸润性导管癌）cT2N1M0 IIB期",
        "chief_complaint": "发现左乳肿块3月",
        "history_present": "患者3月前自检发现左乳外上象限肿块，约3cm，质硬，活动度差，无乳头溢液。穿刺活检提示浸润性导管癌。",
        "past_history": "否认高血压、糖尿病史。母亲有乳腺癌病史。",
        "department": "肿瘤科",
        "hospital": "某某大学附属第一医院",
        "admission_date": "2024-08-01",
        "discharge_date": "2024-08-20",
        "encounter_type": "inpatient",
        "medications": "AC-T方案化疗（多柔比星+环磷酰胺序贯紫杉醇）",
        "procedure_notes": "左乳肿块穿刺活检：浸润性导管癌，ER(+), PR(+), HER2(-), Ki-67 30%",
        "lab_results": [
            {"name": "CEA", "value": 3.2, "unit": "ng/mL"},
            {"name": "CA153", "value": 28.5, "unit": "U/mL"},
            {"name": "ALT", "value": 22.0, "unit": "U/L"},
            {"name": "肌酐", "value": 65.0, "unit": "umol/L"},
            {"name": "eGFR", "value": 95.8, "unit": "mL/min"},
        ],
        "vital_signs": {"bp_systolic": 120, "bp_diastolic": 75, "heart_rate": 72, "temperature": 36.4},
        "discharge_summary": "完成第一次AC方案化疗，耐受性可，轻度恶心呕吐。嘱3周后返院行第二次化疗。",
    },
    {
        "patient_id": "P007",
        "name": "周九",
        "age": 68,
        "gender": "男",
        "diagnosis": "胰腺癌（胰头癌）IV期, 肝转移",
        "chief_complaint": "上腹痛2月，皮肤巩膜黄染1周",
        "history_present": "患者2月前出现上腹部隐痛，伴食欲下降、体重减轻约5kg。1周前出现皮肤巩膜黄染，尿色深黄，大便颜色变浅。",
        "past_history": "2型糖尿病史8年。吸烟20年，已戒10年。",
        "department": "消化内科",
        "hospital": "某某大学附属第二医院",
        "admission_date": "2024-09-10",
        "discharge_date": "2024-09-30",
        "encounter_type": "inpatient",
        "medications": "吉西他滨+白蛋白紫杉醇方案化疗, 奥沙利铂",
        "procedure_notes": "腹部CT增强：胰头部占位4.5x3.8cm，肝内多发转移灶，胆总管扩张",
        "lab_results": [
            {"name": "CA199", "value": 1250, "unit": "U/mL"},
            {"name": "总胆红素", "value": 185.0, "unit": "umol/L"},
            {"name": "直接胆红素", "value": 128.5, "unit": "umol/L"},
            {"name": "ALT", "value": 156.0, "unit": "U/L"},
            {"name": "白蛋白", "value": 28.5, "unit": "g/L"},
        ],
        "vital_signs": {"bp_systolic": 118, "bp_diastolic": 70, "heart_rate": 88, "temperature": 36.8},
        "discharge_summary": "行PTCD减黄引流，肝功能改善后开始全身化疗。目前一般情况可，嘱门诊随访。",
    },
    {
        "patient_id": "P008",
        "name": "吴十",
        "age": 28,
        "gender": "女",
        "diagnosis": "甲状腺功能亢进症, Graves病",
        "chief_complaint": "心悸、手抖、体重下降2月",
        "history_present": "患者2月前无明显诱因出现心悸、双手细颤，体重下降约6kg，伴怕热多汗、易怒。外院查甲功提示T3、T4升高，TSH降低。",
        "past_history": "既往体健，否认肝炎、结核病史。",
        "department": "内分泌科",
        "hospital": "某某大学附属第一医院",
        "admission_date": "2024-05-01",
        "discharge_date": "2024-05-08",
        "encounter_type": "inpatient",
        "medications": "甲巯咪唑 10mg tid, 普萘洛尔 10mg tid",
        "procedure_notes": "甲状腺超声：甲状腺弥漫性肿大，血流信号丰富",
        "lab_results": [
            {"name": "FT3", "value": 15.8, "unit": "pmol/L"},
            {"name": "FT4", "value": 42.5, "unit": "pmol/L"},
            {"name": "TSH", "value": 0.01, "unit": "mIU/L"},
            {"name": "TRAb", "value": 12.5, "unit": "IU/L"},
        ],
        "vital_signs": {"bp_systolic": 130, "bp_diastolic": 75, "heart_rate": 110, "temperature": 37.0},
        "discharge_summary": "确诊Graves病，启动抗甲亢药物治疗。心率控制后出院，嘱定期复查甲功。",
    },
    {
        "patient_id": "P009",
        "name": "郑十一",
        "age": 75,
        "gender": "男",
        "diagnosis": "脑梗死（左侧大脑中动脉供血区）, 心房颤动",
        "chief_complaint": "突发右侧肢体无力伴言语不清4小时",
        "history_present": "患者4小时前突发右侧上下肢无力，持物不稳，伴言语含糊不清，无头痛呕吐。急诊CT排除脑出血。",
        "past_history": "房颤病史3年，未规律抗凝。高血压病史20年。",
        "department": "神经内科",
        "hospital": "某某大学附属第一医院",
        "admission_date": "2024-02-15",
        "discharge_date": "2024-03-10",
        "encounter_type": "inpatient",
        "medications": "阿替普酶静脉溶栓, 华法林 2.5mg qd, 氨氯地平 5mg qd",
        "procedure_notes": "头颅MRI：左侧基底节区急性脑梗死。MRA：左侧大脑中动脉M1段狭窄。",
        "lab_results": [
            {"name": "INR", "value": 1.1, "unit": ""},
            {"name": "D-dimer", "value": 1.2, "unit": "mg/L"},
            {"name": "LDL-C", "value": 3.5, "unit": "mmol/L"},
            {"name": "HbA1c", "value": 6.2, "unit": "%"},
        ],
        "vital_signs": {"bp_systolic": 158, "bp_diastolic": 92, "heart_rate": 90, "temperature": 36.5},
        "discharge_summary": "入院后行静脉溶栓，肢体肌力部分恢复。启动华法林抗凝，血压控制达标后转康复科继续治疗。",
    },
    {
        "patient_id": "P010",
        "name": "冯十二",
        "age": 42,
        "gender": "女",
        "diagnosis": "胃癌（胃窦腺癌）cT3N2M0 IIIA期",
        "chief_complaint": "上腹痛伴黑便1月",
        "history_present": "患者1月前出现上腹部隐痛，餐后加重，伴黑便，体重下降约3kg。胃镜示胃窦溃疡型肿物，活检提示低分化腺癌。",
        "past_history": "幽门螺杆菌感染史，未规范根除。否认肝炎病史。",
        "department": "肿瘤科",
        "hospital": "某某大学附属第二医院",
        "admission_date": "2024-07-20",
        "discharge_date": "2024-08-10",
        "encounter_type": "inpatient",
        "medications": "SOX方案化疗（奥沙利铂+替吉奥）",
        "procedure_notes": "胃镜：胃窦小弯侧溃疡型肿物4x3cm。腹腔镜探查：腹膜无转移。",
        "lab_results": [
            {"name": "CEA", "value": 8.5, "unit": "ng/mL"},
            {"name": "CA724", "value": 45.2, "unit": "U/mL"},
            {"name": "血红蛋白", "value": 92, "unit": "g/L"},
            {"name": "白蛋白", "value": 35.2, "unit": "g/L"},
            {"name": "肌酐", "value": 58.0, "unit": "umol/L"},
        ],
        "vital_signs": {"bp_systolic": 110, "bp_diastolic": 68, "heart_rate": 82, "temperature": 36.6},
        "discharge_summary": "新辅助化疗2周期后评估，肿物缩小至2.5x2cm。计划行根治性远端胃大部切除术。",
    },
]

# ── Insert data ─────────────────────────────────────────────────────────
for i, patient in enumerate(patients):
    es.index(index=ES_INDEX, id=patient["patient_id"], body=patient)

es.indices.refresh(index=ES_INDEX)
count = es.count(index=ES_INDEX)["count"]
print(f"Inserted {count} patient records into {ES_INDEX}")
print("Done!")
