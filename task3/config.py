"""Shared constants for the task 3 diabetes-screening pipeline."""
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = TASK_DIR.parent / "smoking_driking_dataset_Ver01.csv"
DATA_DIR = TASK_DIR / "data"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Fasting blood glucose >= 126 mg/dL is the WHO/ADA diagnostic threshold for diabetes.
GLUCOSE_COLUMN = "BLDS"
DIABETES_THRESHOLD_MG_DL = 126
TARGET = "diabetes"

# Korean NHIS code for blindness in the visual acuity columns.
SIGHT_BLIND_CODE = 9.9

# Wide, physiologically plausible bounds (inclusive) for adults. They only catch data-entry
# errors and sentinel codes (e.g. waistline 999, AST 9999, HDL 8110); extreme but real
# values are kept and handled later by the train-fitted QuantileClipper.
PLAUSIBLE_RANGES = {
    "height": (130, 200),
    "weight": (25, 200),
    "waistline": (50, 150),
    "SBP": (60, 250),
    "DBP": (30, 150),
    "BLDS": (40, 600),
    "tot_chole": (50, 1000),
    "HDL_chole": (10, 200),
    "LDL_chole": (10, 600),
    "triglyceride": (10, 2000),
    "hemoglobin": (4, 22),
    "serum_creatinine": (0.2, 15),
    "SGOT_AST": (1, 2000),
    "SGOT_ALT": (1, 2000),
    "gamma_GTP": (1, 999),
}

# Computed from cleaned columns by feature_extraction.add_derived_features.
DERIVED_FEATURES = ["bmi", "waist_to_height"]

# Feature groups by preprocessing treatment; each model's feature set selects from them.
# Right-skewed lab values (skewness 4-22 on the raw scale, ~0.4-1.9 after log1p).
SKEWED_FEATURES = ["triglyceride", "serum_creatinine", "SGOT_AST", "SGOT_ALT", "gamma_GTP"]
CONTINUOUS_FEATURES = [
    "age", "height", "weight", "waistline",
    "sight_left", "sight_right",
    "SBP", "DBP",
    "tot_chole", "HDL_chole", "LDL_chole",
    "hemoglobin",
] + DERIVED_FEATURES
# Urine dipstick protein: 1 (-) .. 6 (4+), an ordered scale.
ORDINAL_FEATURES = ["urine_protein"]
BINARY_FEATURES = [
    "male", "drinker",
    "hear_abnormal_left", "hear_abnormal_right",
    "blind_left", "blind_right",
]
# 1 = never, 2 = former, 3 = current smoker.
SMOKING_FEATURE = "smoking_status"
SMOKING_CATEGORIES = [1, 2, 3]

# Columns produced by cleaning.clean, before feature extraction.
CLEANED_FEATURES = [
    feature
    for feature in CONTINUOUS_FEATURES + SKEWED_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES
    if feature not in DERIVED_FEATURES
] + [SMOKING_FEATURE]

# Values that need a blood or urine sample. The questionnaire model excludes them because a
# blood sample could be used to measure HbA1c directly, making the screening tool redundant.
LAB_FEATURES = [
    "tot_chole", "HDL_chole", "LDL_chole", "triglyceride", "hemoglobin",
    "serum_creatinine", "SGOT_AST", "SGOT_ALT", "gamma_GTP", "urine_protein",
]

# Measured with professional equipment (blood-pressure monitor, eye chart, hearing test). The
# questionnaire model only uses values a person can report or measure at home.
EQUIPMENT_FEATURES = [
    "SBP", "DBP",
    "sight_left", "sight_right", "blind_left", "blind_right",
    "hear_abnormal_left", "hear_abnormal_right",
]

ALL_FEATURES = CLEANED_FEATURES + DERIVED_FEATURES
FEATURE_SETS = {
    # Main model: self-reported, or measurable at home with a scale and a tape measure.
    "questionnaire": [
        feature for feature in ALL_FEATURES
        if feature not in LAB_FEATURES + EQUIPMENT_FEATURES
    ],
    # Comparison model: everything, to quantify what a blood test would add.
    "full": ALL_FEATURES,
}
