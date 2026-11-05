import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TEST_CSV = "../data/test_processed_questionnaire.csv"   # <-- enter your filenames
TRAIN_CSV = "../data/train_processed_questionnaire.csv"
TARGET = "diabetes"
CONTINUOUS = ["age", "height", "weight", "waistline", "bmi", "waist_to_height"]
# male, drinker, smoking_status_2, smoking_status_3 are binary and left unscaled


def load_split():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    return (train.drop(columns=[TARGET]), test.drop(columns=[TARGET]),
            train[TARGET], test[TARGET])


def make_pipeline(**mlp_params):
    # Scaler is inside the pipeline, so in CV it is refit on each training fold only.
    prep = ColumnTransformer([("scale", StandardScaler(), CONTINUOUS)],
                             remainder="passthrough")
    defaults = dict(early_stopping=True, validation_fraction=0.1,
                    n_iter_no_change=5, tol=1e-3, max_iter=100)
    defaults.update(mlp_params)          # caller values override defaults
    return Pipeline([("prep", prep), ("mlp", MLPClassifier(**defaults))])