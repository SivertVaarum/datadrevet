"""Derived features computed row by row from cleaned columns. Learns nothing from the data, so it is
safe to apply before the train/test split and identical at training and deployment time."""


def add_derived_features(df):
    """Add clinically established obesity markers a person can compute from self-reported values.

    - bmi: weight / height^2 (kg/m^2), general adiposity.
    - waist_to_height: waist / height, abdominal adiposity relative to body size.

    Both are non-linear combinations of the inputs, so they give linear and distance-based models
    information they cannot construct themselves. A missing waistline gives a missing ratio,
    which the preprocessor's imputer fills in.
    """
    df = df.copy()
    height_m = df["height"] / 100
    df["bmi"] = df["weight"] / height_m**2
    df["waist_to_height"] = df["waistline"] / df["height"]
    return df
