"""Row-level cleaning that learns no statistics from the data, so it is safe to run before the
train/test split. Anything fitted on data (imputation, clipping, scaling) lives in preprocessor.py."""
import logging

import numpy as np
import pandas as pd

from config import (
    CLEANED_FEATURES,
    DIABETES_THRESHOLD_MG_DL,
    GLUCOSE_COLUMN,
    PLAUSIBLE_RANGES,
    RAW_DATA_PATH,
    SIGHT_BLIND_CODE,
    SMOKING_FEATURE,
    TARGET,
)

logger = logging.getLogger(__name__)


def load_raw(path=RAW_DATA_PATH):
    return pd.read_csv(path)


def drop_duplicates(df):
    deduplicated = df.drop_duplicates()
    logger.info("Dropped %d exact duplicate rows", len(df) - len(deduplicated))
    return deduplicated


def add_target(df):
    """Derive the binary diabetes label from fasting glucose, then drop the glucose column.

    Rows with an implausible glucose value are deleted because their label is unknown.
    BLDS must not remain as a feature: the label is a direct threshold on it (target leakage).
    """
    low, high = PLAUSIBLE_RANGES[GLUCOSE_COLUMN]
    valid = df[GLUCOSE_COLUMN].between(low, high)
    logger.info("Dropped %d rows with implausible %s (label unknown)", (~valid).sum(), GLUCOSE_COLUMN)

    df = df[valid].copy()
    df[TARGET] = (df[GLUCOSE_COLUMN] >= DIABETES_THRESHOLD_MG_DL).astype(int)
    return df.drop(columns=[GLUCOSE_COLUMN])


def recode_blindness(df):
    """Visual acuity 9.9 is a blindness code, not a measurement: move it to a flag and blank the acuity."""
    df = df.copy()
    for side in ("left", "right"):
        sight = f"sight_{side}"
        is_blind = np.isclose(df[sight], SIGHT_BLIND_CODE)
        df[f"blind_{side}"] = is_blind.astype(int)
        df.loc[is_blind, sight] = np.nan
        logger.info("%s: %d blindness codes moved to blind_%s", sight, is_blind.sum(), side)
    return df


def mask_implausible_values(df):
    """Set out-of-range feature values to NaN instead of deleting the row.

    A screening model receives whole patients at deployment time and cannot discard them,
    so one bad lab value should cost that value only; the imputer fills it in later.
    """
    df = df.copy()
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        if column not in df.columns:
            continue
        implausible = df[column].notna() & ~df[column].between(low, high)
        df.loc[implausible, column] = np.nan
        if implausible.any():
            logger.info("%s: %d implausible values set to NaN", column, implausible.sum())
    return df


def encode_categoricals(df):
    """Turn the string and 1/2-coded columns into 0/1 indicators with descriptive names."""
    df = df.copy()
    df["male"] = (df["sex"] == "Male").astype(int)
    df["drinker"] = (df["DRK_YN"] == "Y").astype(int)
    # Hearing is coded 1 = normal, 2 = abnormal.
    df["hear_abnormal_left"] = (df["hear_left"] == 2).astype(int)
    df["hear_abnormal_right"] = (df["hear_right"] == 2).astype(int)
    df[SMOKING_FEATURE] = df["SMK_stat_type_cd"].astype(int)
    return df.drop(columns=["sex", "DRK_YN", "hear_left", "hear_right", "SMK_stat_type_cd"])


def clean(df):
    df = drop_duplicates(df)
    df = add_target(df)
    df = recode_blindness(df)
    df = mask_implausible_values(df)
    df = encode_categoricals(df)
    return df[CLEANED_FEATURES + [TARGET]].reset_index(drop=True)
