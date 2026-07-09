"""
Generate five large, comprehensive, PHI-free English patient cohorts for the
EligES demo.

Design goals
------------
* **Big**: ~500 patients per cohort (~2,500 total).
* **Comprehensive ("full")**: every record carries the same rich field set as
  the benchmark n2c2/MIMIC indices, plus a *shared comprehensive laboratory
  panel* (CBC + BMP + HbA1c + lipids + LFTs + thyroid + coagulation) merged
  with cohort-specific labs. This means any lab-, diagnosis-, demographic-,
  temporal-, or exclusion-based criterion can be exercised and compared across
  every cohort, exactly as on n2c2.
* **Comparable to existing datasets**: identical key schema to the n2c2 index
  (patient_id, name, age, gender, diagnosis, chief_complaint, history_present,
  past_history, department, hospital, admission_date, discharge_date,
  encounter_type, medications, procedure_notes, lab_results, vital_signs,
  discharge_summary) plus additional structured fields surfaced in the UI
  (allergies, family_history, social_history, surgical_history,
  dietary_supplements, immunizations, functional_status).

All records are synthetic and contain no real PHI. The generator is
deterministic (seeded) so the datasets are reproducible.

Run:  python data/generate_synthetic_datasets.py
Output: data/synthetic/<index>.json   (one JSON array per cohort)
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20260704
N_PER_COHORT = 500
BASE_DATE = date(2026, 6, 1)  # "today" for relative temporal events
OUT_DIR = Path(__file__).resolve().parent / "synthetic"

# ---------------------------------------------------------------------------
# Shared demographic pools (culturally varied, clearly synthetic)
# ---------------------------------------------------------------------------
FIRST_NAMES_M = [
    "James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Steven",
    "Andrew", "Kevin", "Brian", "George", "Edward", "Ronald", "Wei", "Hiroshi",
    "Omar", "Diego", "Ahmed", "Raj", "Ivan", "Kwame", "Luca", "Mateo",
]
FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan",
    "Jessica", "Sarah", "Karen", "Nancy", "Lisa", "Margaret", "Sandra",
    "Ashley", "Emily", "Donna", "Michelle", "Carol", "Amanda", "Mei", "Yuki",
    "Fatima", "Sofia", "Aisha", "Priya", "Olga", "Amara", "Giulia", "Valentina",
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Chen", "Wang", "Kim", "Patel", "Nguyen", "Kumar", "Okafor",
    "Rossi", "Ivanov",
]
ETHNICITIES = [
    "White", "Black or African American", "Hispanic or Latino",
    "Asian", "Native American", "Middle Eastern", "Mixed race",
]
HOSPITALS = [
    "Riverside General Hospital", "Lakeview Medical Center",
    "St. Aquinas University Hospital", "Northgate Regional Hospital",
    "Cedar Valley Health System", "Harborview Teaching Hospital",
]
NON_ENGLISH_LANGS = ["Spanish", "Mandarin", "Arabic", "Vietnamese", "Russian", "Haitian Creole"]

ALLERGIES = [
    "No known drug allergies.", "No known drug allergies.", "No known drug allergies.",
    "Penicillin (rash).", "Sulfa drugs (hives).", "NSAIDs (angioedema).",
    "Iodinated contrast (anaphylaxis).", "Latex.", "Aspirin (bronchospasm).",
    "Shellfish.", "Codeine (nausea).",
]
FAMILY_HX = [
    "Father with coronary artery disease.", "Mother with type 2 diabetes.",
    "Sibling with breast cancer.", "Father deceased from myocardial infarction.",
    "Mother with hypertension and stroke.", "Family history of colorectal cancer.",
    "No significant family history.", "Maternal history of chronic kidney disease.",
    "Father with COPD (long-term smoker).", "Family history of early-onset dementia.",
]
SURGERIES = [
    "appendectomy", "laparoscopic cholecystectomy", "inguinal hernia repair",
    "coronary artery bypass grafting", "total hip replacement",
    "cesarean section", "hysterectomy", "partial colectomy",
    "tonsillectomy", "cataract extraction", "lumbar laminectomy",
    "thyroidectomy",
]
ABDOMINAL_SURG = {"appendectomy", "laparoscopic cholecystectomy",
                  "inguinal hernia repair", "cesarean section",
                  "hysterectomy", "partial colectomy"}
SUPPLEMENTS = [
    "Vitamin D3 2000 IU daily", "Fish oil (omega-3) 1000 mg daily",
    "Multivitamin daily", "Calcium carbonate 600 mg daily",
    "Vitamin B12 1000 mcg daily", "Magnesium 400 mg daily",
    "Coenzyme Q10 100 mg daily", "Iron supplement 65 mg daily",
]
IMMUNIZATIONS = [
    "Influenza (current season), pneumococcal, and COVID-19 vaccines up to date.",
    "Influenza vaccine up to date; pneumococcal vaccine not yet administered.",
    "COVID-19 primary series complete; booster overdue.",
    "Immunizations not up to date per records.",
    "Influenza and COVID-19 vaccines up to date.",
]
FUNCTIONAL = [
    "Independent in all activities of daily living.",
    "Requires minimal assistance with instrumental activities of daily living.",
    "Ambulates with a cane; otherwise independent.",
    "Requires assistance with bathing and dressing.",
]
CODE_STATUS = ["Full code", "Full code", "Full code", "DNR/DNI"]
INSURANCE = ["Medicare", "Medicaid", "Private (commercial)", "Self-pay", "Veterans Affairs"]

SMOKING = [
    ("Never smoker.", 0),
    ("Former smoker, quit {yrs} years ago ({pk} pack-year history).", None),
    ("Current smoker, {pk} pack-year history.", None),
    ("Current smoker, half a pack per day for {pk} years.", None),
]
ALCOHOL = [
    "Denies alcohol use.",
    "Social alcohol use, 1-2 drinks per week.",
    "Reports heavy alcohol use, approximately {n} drinks per day; meets criteria for alcohol use disorder.",
    "History of alcohol use disorder, currently in sustained remission.",
    "Occasional alcohol use on weekends.",
]
DRUGS = [
    "No history of illicit drug use.",
    "No history of illicit drug use.",
    "No history of illicit drug use.",
    "Past history of intravenous drug use; reports abstinence for {yrs} years.",
    "Active cocaine use disorder reported on intake.",
    "History of opioid use disorder, maintained on buprenorphine.",
    "Reports occasional cannabis use.",
]

# ---------------------------------------------------------------------------
# Shared comprehensive laboratory panel (present in EVERY cohort so that any
# lab-based criterion is comparable across cohorts). Cohort-specific labs may
# override these ranges by name.
# ---------------------------------------------------------------------------
CORE_LABS = [
    ("WBC", (2.0, 16.0), "10^9/L", 1),
    ("Hemoglobin", (7.5, 17.5), "g/dL", 1),
    ("Hematocrit", (24, 52), "%", 0),
    ("Platelet count", (60, 450), "10^9/L", 0),
    ("Sodium", (128, 148), "mmol/L", 0),
    ("Potassium", (3.0, 6.2), "mmol/L", 1),
    ("Chloride", (95, 112), "mmol/L", 0),
    ("Bicarbonate", (16, 30), "mmol/L", 0),
    ("BUN", (7, 90), "mg/dL", 0),
    ("Creatinine", (0.5, 4.5), "mg/dL", 2),
    ("eGFR", (15, 120), "mL/min/1.73m2", 0),
    ("Glucose", (60, 320), "mg/dL", 0),
    ("HbA1c", (4.8, 12.5), "%", 1),
    ("Total cholesterol", (110, 320), "mg/dL", 0),
    ("LDL cholesterol", (40, 220), "mg/dL", 0),
    ("HDL cholesterol", (22, 90), "mg/dL", 0),
    ("Triglycerides", (50, 520), "mg/dL", 0),
    ("ALT", (8, 190), "U/L", 0),
    ("AST", (8, 210), "U/L", 0),
    ("Total bilirubin", (0.2, 4.5), "mg/dL", 1),
    ("Albumin", (2.0, 5.0), "g/dL", 1),
    ("TSH", (0.1, 9.0), "mIU/L", 2),
    ("INR", (0.9, 3.5), "", 1),
    ("BMI", (17, 44), "kg/m2", 1),
]

# ---------------------------------------------------------------------------
# Cohort definitions (disease-specific content layered on the shared panel)
# ---------------------------------------------------------------------------
COHORTS = {
    "emr_cardiometabolic": {
        "label": "Cardiometabolic clinic",
        "department": "Endocrinology / Cardiology",
        "diagnoses": [
            ("Type 2 diabetes mellitus", "E11.9"),
            ("Type 2 diabetes mellitus with diabetic nephropathy", "E11.21"),
            ("Coronary artery disease", "I25.10"),
            ("Old myocardial infarction", "I25.2"),
            ("Essential hypertension", "I10"),
            ("Hyperlipidemia", "E78.5"),
            ("Congestive heart failure with reduced ejection fraction", "I50.22"),
            ("Atrial fibrillation", "I48.91"),
            ("Peripheral arterial disease", "I73.9"),
            ("Obesity, class II", "E66.01"),
        ],
        "comorbid": [
            "obstructive sleep apnea", "chronic kidney disease stage 3",
            "non-alcoholic fatty liver disease", "gout", "hypothyroidism",
            "diabetic retinopathy", "diabetic peripheral neuropathy",
        ],
        "chief": [
            "polyuria, polydipsia and unintentional weight loss",
            "exertional chest pressure and dyspnea",
            "poorly controlled blood glucose despite oral agents",
            "palpitations and lightheadedness",
            "bilateral lower-extremity edema",
        ],
        "hpi": [
            "The patient reports a {dur}-month history of {chief}. Home glucose logs show fasting values in the 180-260 mg/dL range.",
            "The patient describes {chief} with symptoms worsening over the past {dur} weeks. Prior aspirin and statin therapy noted.",
            "Symptoms of {chief} began approximately {dur} months ago and are aggravated by exertion.",
        ],
        "meds": [
            "Metformin 1000 mg PO BID", "Empagliflozin 10 mg PO daily",
            "Insulin glargine 24 units SC nightly", "Atorvastatin 40 mg PO daily",
            "Lisinopril 20 mg PO daily", "Aspirin 81 mg PO daily",
            "Metoprolol succinate 50 mg PO daily", "Furosemide 40 mg PO daily",
            "Clopidogrel 75 mg PO daily", "Amlodipine 5 mg PO daily",
        ],
        "procedures": [
            "Coronary angiography with drug-eluting stent to the LAD.",
            "Transthoracic echocardiogram showing EF 40%.",
            "Ambulatory 24-hour Holter monitoring.",
            "Diabetic retinopathy screening.",
        ],
        "labs": [
            ("HbA1c", (5.4, 11.8), "%", 1),
            ("Fasting glucose", (85, 280), "mg/dL", 0),
            ("Troponin T", (0.001, 0.25), "ng/mL", 3),
            ("NT-proBNP", (30, 5200), "pg/mL", 0),
            ("Creatinine", (0.6, 2.6), "mg/dL", 2),
        ],
        "events": ["myocardial infarction", "diabetic ketoacidosis", "coronary stent placement"],
    },
    "emr_oncology": {
        "label": "Medical oncology service",
        "department": "Medical Oncology",
        "diagnoses": [
            ("Invasive ductal carcinoma of the breast, stage II", "C50.911"),
            ("Non-small cell lung carcinoma, stage IIIA", "C34.90"),
            ("Colon adenocarcinoma, stage III", "C18.9"),
            ("Diffuse large B-cell lymphoma", "C83.30"),
            ("Prostate adenocarcinoma, Gleason 7", "C61"),
            ("Pancreatic ductal adenocarcinoma", "C25.9"),
            ("Metastatic melanoma", "C43.9"),
            ("Ovarian high-grade serous carcinoma", "C56.9"),
            ("Chronic lymphocytic leukemia", "C91.10"),
            ("Hepatocellular carcinoma", "C22.0"),
        ],
        "comorbid": [
            "chemotherapy-induced neutropenia", "anemia of chronic disease",
            "cancer-associated venous thromboembolism", "type 2 diabetes",
            "chronic pain requiring opioids", "hypertension",
        ],
        "chief": [
            "a palpable mass and progressive fatigue",
            "unintentional weight loss and night sweats",
            "worsening cough with hemoptysis",
            "abdominal pain and early satiety",
            "new lymphadenopathy",
        ],
        "hpi": [
            "The patient presents for cycle {dur} of systemic chemotherapy. Restaging imaging shows a partial response.",
            "The patient reports {chief} over the past {dur} months, prompting oncologic workup and biopsy confirmation.",
            "Following diagnosis {dur} months ago, the patient has completed neoadjuvant therapy and is being evaluated for surgery.",
        ],
        "meds": [
            "Carboplatin AUC 5 IV every 3 weeks", "Paclitaxel 175 mg/m2 IV every 3 weeks",
            "Pembrolizumab 200 mg IV every 3 weeks", "Ondansetron 8 mg PO PRN nausea",
            "Filgrastim 300 mcg SC daily", "Dexamethasone 4 mg PO BID",
            "Oxycodone 5 mg PO q6h PRN pain", "Enoxaparin 40 mg SC daily",
            "Tamoxifen 20 mg PO daily", "Rituximab 375 mg/m2 IV",
        ],
        "procedures": [
            "CT-guided core needle biopsy of the primary lesion.",
            "Port-a-cath placement for chemotherapy access.",
            "PET-CT for restaging.",
            "Bone marrow biopsy and aspirate.",
        ],
        "labs": [
            ("Absolute neutrophil count", (0.2, 8.5), "10^9/L", 1),
            ("LDH", (120, 900), "U/L", 0),
            ("CA 19-9", (5, 800), "U/mL", 0),
            ("CEA", (0.5, 120), "ng/mL", 1),
            ("Hemoglobin", (7.0, 14.0), "g/dL", 1),
        ],
        "events": ["venous thromboembolism", "febrile neutropenia", "tumor resection"],
    },
    "emr_respiratory": {
        "label": "Pulmonary medicine service",
        "department": "Pulmonology",
        "diagnoses": [
            ("Chronic obstructive pulmonary disease, GOLD stage III", "J44.1"),
            ("Asthma, moderate persistent", "J45.40"),
            ("Community-acquired pneumonia", "J18.9"),
            ("Idiopathic pulmonary fibrosis", "J84.112"),
            ("Obstructive sleep apnea", "G47.33"),
            ("Bronchiectasis", "J47.9"),
            ("Pulmonary embolism", "I26.99"),
            ("Pulmonary arterial hypertension", "I27.20"),
            ("Sarcoidosis with pulmonary involvement", "D86.0"),
            ("Chronic respiratory failure with hypoxia", "J96.11"),
        ],
        "comorbid": [
            "cor pulmonale", "gastroesophageal reflux disease",
            "chronic sinusitis", "type 2 diabetes", "coronary artery disease",
            "hypertension",
        ],
        "chief": [
            "progressive dyspnea on exertion",
            "productive cough with purulent sputum",
            "wheezing and chest tightness",
            "acute onset pleuritic chest pain and dyspnea",
            "recurrent respiratory infections",
        ],
        "hpi": [
            "The patient reports {chief} for the past {dur} weeks, with a documented decline in exercise tolerance.",
            "The patient presents with {chief}. Home oxygen saturation has dropped to the low 90s.",
            "An exacerbation of the underlying condition developed {dur} days ago despite maintenance inhaler therapy.",
        ],
        "meds": [
            "Tiotropium 18 mcg inhaled daily", "Fluticasone/salmeterol 250/50 inhaled BID",
            "Albuterol MDI 2 puffs q4h PRN", "Prednisone 40 mg PO daily taper",
            "Azithromycin 500 mg PO daily x3", "Roflumilast 500 mcg PO daily",
            "Montelukast 10 mg PO nightly", "Pirfenidone 801 mg PO TID",
            "Apixaban 5 mg PO BID", "Supplemental oxygen 2 L/min via nasal cannula",
        ],
        "procedures": [
            "Flexible bronchoscopy with bronchoalveolar lavage.",
            "High-resolution CT of the chest.",
            "Pulmonary function testing with DLCO.",
            "CT pulmonary angiography.",
        ],
        "labs": [
            ("pO2 (arterial)", (48, 95), "mmHg", 0),
            ("pCO2 (arterial)", (30, 68), "mmHg", 0),
            ("pH (arterial)", (7.25, 7.48), "", 2),
            ("CRP", (2, 220), "mg/L", 0),
            ("D-dimer", (0.2, 6.5), "mg/L", 1),
            ("Eosinophils", (0.0, 1.2), "10^9/L", 2),
        ],
        "events": ["pulmonary embolism", "COPD exacerbation", "pneumonia hospitalization"],
    },
    "emr_nephrology": {
        "label": "Nephrology service",
        "department": "Nephrology",
        "diagnoses": [
            ("Chronic kidney disease, stage 4", "N18.4"),
            ("End-stage renal disease on hemodialysis", "N18.6"),
            ("Acute kidney injury", "N17.9"),
            ("Diabetic nephropathy", "E11.21"),
            ("IgA nephropathy", "N02.8"),
            ("Hypertensive nephrosclerosis", "I12.9"),
            ("Polycystic kidney disease", "Q61.2"),
            ("Lupus nephritis", "M32.14"),
            ("Nephrotic syndrome", "N04.9"),
            ("Renal transplant recipient", "Z94.0"),
        ],
        "comorbid": [
            "secondary hyperparathyroidism", "renal anemia",
            "type 2 diabetes", "resistant hypertension", "metabolic acidosis",
            "coronary artery disease",
        ],
        "chief": [
            "declining urine output and lower-extremity edema",
            "fatigue and pruritus",
            "elevated creatinine noted on routine labs",
            "hypertension refractory to three agents",
            "foamy urine and periorbital edema",
        ],
        "hpi": [
            "The patient's renal function has declined over the past {dur} months, with rising creatinine and worsening edema.",
            "The patient presents with {chief}. Nephrology is consulted for further management and possible renal replacement therapy.",
            "Following {dur} sessions of maintenance hemodialysis, the patient reports {chief}.",
        ],
        "meds": [
            "Sevelamer 800 mg PO TID with meals", "Calcitriol 0.25 mcg PO daily",
            "Epoetin alfa 4000 units SC weekly", "Furosemide 80 mg PO BID",
            "Lisinopril 40 mg PO daily", "Sodium bicarbonate 650 mg PO TID",
            "Tacrolimus 3 mg PO BID", "Mycophenolate mofetil 1000 mg PO BID",
            "Cinacalcet 30 mg PO daily", "Iron sucrose 100 mg IV weekly",
        ],
        "procedures": [
            "Ultrasound-guided native kidney biopsy.",
            "Arteriovenous fistula creation for dialysis access.",
            "Tunneled hemodialysis catheter placement.",
            "Renal transplant, deceased donor.",
        ],
        "labs": [
            ("Creatinine", (1.4, 9.5), "mg/dL", 2),
            ("eGFR", (5, 55), "mL/min/1.73m2", 0),
            ("Phosphate", (2.5, 8.0), "mg/dL", 1),
            ("Parathyroid hormone", (60, 900), "pg/mL", 0),
            ("Urine protein/creatinine ratio", (0.1, 9.0), "g/g", 1),
        ],
        "events": ["dialysis initiation", "acute kidney injury", "kidney transplant"],
    },
    "emr_neuropsychiatric": {
        "label": "Neurology & behavioral health service",
        "department": "Neurology / Psychiatry",
        "diagnoses": [
            ("Acute ischemic stroke", "I63.9"),
            ("Epilepsy, focal onset", "G40.209"),
            ("Major depressive disorder, recurrent", "F33.1"),
            ("Generalized anxiety disorder", "F41.1"),
            ("Parkinson disease", "G20"),
            ("Alzheimer disease with dementia", "G30.9"),
            ("Multiple sclerosis, relapsing-remitting", "G35"),
            ("Opioid use disorder", "F11.20"),
            ("Alcohol use disorder", "F10.20"),
            ("Bipolar I disorder", "F31.9"),
        ],
        "comorbid": [
            "hypertension", "type 2 diabetes", "chronic pain",
            "insomnia", "tobacco use disorder", "hyperlipidemia",
        ],
        "chief": [
            "acute-onset unilateral weakness and slurred speech",
            "recurrent seizures despite therapy",
            "low mood, anhedonia and poor sleep",
            "resting tremor and bradykinesia",
            "progressive memory impairment",
        ],
        "hpi": [
            "The patient presents with {chief} beginning {dur} hours ago; NIH Stroke Scale documented on arrival.",
            "The patient reports {chief} over the past {dur} weeks, with functional decline noted by family.",
            "Symptoms of {chief} have persisted for {dur} months despite outpatient management.",
        ],
        "meds": [
            "Aspirin 81 mg PO daily", "Atorvastatin 80 mg PO daily",
            "Levetiracetam 750 mg PO BID", "Sertraline 100 mg PO daily",
            "Carbidopa/levodopa 25/100 mg PO TID", "Donepezil 10 mg PO nightly",
            "Buprenorphine/naloxone 8/2 mg SL daily", "Naltrexone 50 mg PO daily",
            "Lamotrigine 200 mg PO BID", "Lorazepam 0.5 mg PO PRN anxiety",
        ],
        "procedures": [
            "CT head without contrast, followed by MRI brain with DWI.",
            "Continuous video EEG monitoring.",
            "Mechanical thrombectomy of the MCA.",
            "Lumbar puncture for CSF analysis.",
        ],
        "labs": [
            ("Blood alcohol level", (0, 320), "mg/dL", 0),
            ("Ammonia", (10, 140), "umol/L", 0),
            ("Valproate level", (20, 130), "ug/mL", 0),
            ("Vitamin B12", (120, 900), "pg/mL", 0),
            ("Urine toxicology", (0, 1), "screen", 0),
        ],
        "events": ["ischemic stroke", "status epilepticus", "psychiatric hospitalization"],
    },
}


def _rand_value(rng, lo, hi, decimals):
    v = rng.uniform(lo, hi)
    return round(v, decimals) if decimals else int(round(v))


def _build_lab_panel(rng, cohort):
    """Merge the shared core panel with cohort-specific labs (cohort overrides)."""
    by_name = {}
    for name, rng_tuple, unit, dec in CORE_LABS:
        by_name[name] = (name, rng_tuple, unit, dec)
    for name, rng_tuple, unit, dec in cohort["labs"]:
        by_name[name] = (name, rng_tuple, unit, dec)  # override / add
    labs = []
    for name, (lo, hi), unit, dec in by_name.values():
        labs.append({"name": name, "value": _rand_value(rng, lo, hi, dec), "unit": unit})
    return labs


def _make_social_history(rng):
    smoke_tpl, _ = rng.choice(SMOKING)
    smoke = smoke_tpl.format(yrs=rng.randint(1, 25), pk=rng.randint(5, 60))
    alc = rng.choice(ALCOHOL).format(n=rng.randint(4, 12))
    drug = rng.choice(DRUGS).format(yrs=rng.randint(1, 15))
    return smoke, alc, drug


def _recent_event_sentence(rng, cohort):
    """Dated clinical event, enabling temporal criteria (e.g. MI in past 6 months)."""
    event = rng.choice(cohort["events"])
    months_ago = rng.choice([1, 2, 3, 4, 5, 6, 8, 11, 14, 18, 24])
    when = BASE_DATE - timedelta(days=months_ago * 30)
    return (f"The patient has a documented history of {event} approximately "
            f"{months_ago} months ago (around {when.strftime('%B %Y')})."), event, months_ago


def _make_patient(rng, cohort_key, cohort, idx):
    gender = rng.choice(["Male", "Female"])
    first = rng.choice(FIRST_NAMES_M if gender == "Male" else FIRST_NAMES_F)
    last = rng.choice(LAST_NAMES)
    name = f"{first} {last}"
    age = rng.randint(28, 89)
    ethnicity = rng.choice(ETHNICITIES)

    primary, icd = rng.choice(cohort["diagnoses"])
    n_comorbid = rng.randint(1, 3)
    comorbid = rng.sample(cohort["comorbid"], min(n_comorbid, len(cohort["comorbid"])))
    diagnosis = primary + f" (ICD-10 {icd})"
    if comorbid:
        diagnosis += "; comorbidities: " + ", ".join(comorbid)

    chief = rng.choice(cohort["chief"])
    dur = rng.randint(1, 12)
    hpi = rng.choice(cohort["hpi"]).format(chief=chief, dur=dur)

    speaks_english = rng.random() < 0.82
    lang_note = (
        "The patient is a fluent English speaker and is able to understand and make their own medical decisions (decision-making capacity intact)."
        if speaks_english else
        "The patient primarily speaks {ln} and requires a professional interpreter; decision-making capacity is intact.".format(ln=rng.choice(NON_ENGLISH_LANGS))
    )

    smoke, alc, drug = _make_social_history(rng)
    event_sentence, event_name, event_months = _recent_event_sentence(rng, cohort)

    # Dietary supplement history with timing (supports DIETSUPP-2MOS-style criteria)
    if rng.random() < 0.55:
        supp = rng.choice(SUPPLEMENTS)
        supp_months = rng.choice([1, 2, 3, 6])
        supplements_note = f"Currently taking {supp} (started {supp_months} months ago)."
        dietary_supplements = supp
    else:
        supplements_note = "No dietary supplements reported."
        dietary_supplements = "None"

    # Surgical history (supports abdominal-surgery-style criteria)
    n_surg = rng.randint(0, 2)
    surgeries = rng.sample(SURGERIES, n_surg)
    surgical_history = "; ".join(s.capitalize() for s in surgeries) if surgeries else "No prior surgeries."

    allergies = rng.choice(ALLERGIES)
    family_history = rng.choice(FAMILY_HX)
    immunizations = rng.choice(IMMUNIZATIONS)
    functional_status = rng.choice(FUNCTIONAL)
    code_status = rng.choice(CODE_STATUS)
    insurance = rng.choice(INSURANCE)

    social_history = f"{smoke} {alc} {drug}"
    history_present = (
        f"{hpi} {event_sentence} Social history: {social_history} {supplements_note} {lang_note}"
    )

    n_meds = rng.randint(3, 6)
    meds = rng.sample(cohort["meds"], min(n_meds, len(cohort["meds"])))
    medications = "; ".join(meds)

    n_proc = rng.randint(1, 2)
    procs = rng.sample(cohort["procedures"], min(n_proc, len(cohort["procedures"])))
    procedure_notes = " ".join(procs)

    lab_results = _build_lab_panel(rng, cohort)

    bp_sys = rng.randint(95, 190)
    bp_dia = rng.randint(55, 110)
    hr = rng.randint(52, 122)
    temp = round(rng.uniform(36.0, 39.2), 1)
    spo2 = rng.randint(84, 100)
    rr = rng.randint(12, 30)
    vital_signs = {
        "bp_systolic": bp_sys, "bp_diastolic": bp_dia, "heart_rate": hr,
        "temperature": temp, "spo2": spo2, "respiratory_rate": rr,
    }

    admit = BASE_DATE - timedelta(days=rng.randint(20, 760))
    los = rng.randint(1, 21)
    discharge = admit + timedelta(days=los)
    encounter_type = rng.choice([
        "Inpatient admission", "Outpatient clinic visit",
        "Emergency department visit", "Day-case / observation",
    ])

    past_bits = sorted(set(comorbid + rng.sample(
        ["hypertension", "hyperlipidemia", "type 2 diabetes"], rng.randint(0, 2))))
    past_history = "Medical history: " + (", ".join(past_bits) if past_bits else "unremarkable") + \
                   f". Surgical history: {surgical_history} Family history: {family_history}"

    key_labs = ", ".join(
        f"{lab['name']} {lab['value']}{(' ' + lab['unit']) if lab['unit'] else ''}"
        for lab in lab_results if lab["name"] in
        ("HbA1c", "Creatinine", "Hemoglobin", "eGFR", "Glucose", "WBC")
    )
    disposition = rng.choice([
        "discharged home in stable condition",
        "discharged home with outpatient follow-up",
        "transferred to a skilled nursing facility for rehabilitation",
        "discharged home with home-health services",
    ])
    discharge_summary = (
        f"{age}-year-old {gender.lower()} patient ({ethnicity}) with a {encounter_type.lower()} to "
        f"{cohort['department']} for {chief}. Primary diagnosis: {primary} ({icd}). "
        f"{event_sentence} Relevant comorbidities: {', '.join(comorbid) if comorbid else 'none'}. "
        f"Selected labs: {key_labs}. Vitals: BP {bp_sys}/{bp_dia} mmHg, HR {hr} bpm, "
        f"Temp {temp} C, SpO2 {spo2}%. Managed on {meds[0].lower()} among other agents; "
        f"{disposition} after a {los}-day stay. {functional_status} Code status: {code_status}."
    )

    pid = f"{cohort_key.upper()}_{idx:04d}"
    return {
        "patient_id": pid,
        "name": name,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "language": "English" if speaks_english else "Non-English (interpreter required)",
        "insurance": insurance,
        "encounter_type": encounter_type,
        "diagnosis": diagnosis,
        "chief_complaint": chief.capitalize() + ".",
        "history_present": history_present,
        "past_history": past_history,
        "surgical_history": surgical_history,
        "family_history": family_history,
        "social_history": social_history,
        "allergies": allergies,
        "dietary_supplements": dietary_supplements,
        "immunizations": immunizations,
        "functional_status": functional_status,
        "code_status": code_status,
        "recent_event": event_name,
        "recent_event_months_ago": event_months,
        "medications": medications,
        "procedure_notes": procedure_notes,
        "discharge_summary": discharge_summary,
        "department": cohort["department"],
        "hospital": rng.choice(HOSPITALS),
        "admission_date": admit.isoformat(),
        "discharge_date": discharge.isoformat(),
        "length_of_stay_days": los,
        "vital_signs": vital_signs,
        "lab_results": lab_results,
        "cohort": cohort["label"],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cohort_key, cohort in COHORTS.items():
        rng = random.Random(f"{SEED}-{cohort_key}")
        patients = [_make_patient(rng, cohort_key, cohort, i + 1) for i in range(N_PER_COHORT)]
        out_path = OUT_DIR / f"{cohort_key}.json"
        out_path.write_text(json.dumps(patients, indent=2, ensure_ascii=False), encoding="utf-8")
        n_labs = len(patients[0]["lab_results"])
        summary[cohort_key] = f"{len(patients)} patients, {n_labs} labs/record"
        print(f"[gen] {cohort_key}: {summary[cohort_key]} -> {out_path.name}")
    total = sum(int(v.split()[0]) for v in summary.values())
    print(f"[gen] done: {total} patients across {len(summary)} cohorts")


if __name__ == "__main__":
    main()
