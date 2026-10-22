"""Clean the raw data, add derived features, split it, fit one preprocessor per feature set on train
and save every artifact to data/.

Outputs:
    train.csv / test.csv                                  cleaned, unscaled, all features
    train_processed_<set>.csv / test_processed_<set>.csv  transformed with that set's train-fitted preprocessor
    preprocessor_<set>.joblib                             the fitted preprocessor
for each <set> in config.FEATURE_SETS ("questionnaire" = main model, "full" = comparison model).
"""
import logging

import joblib

from cleaning import clean, load_raw
from config import DATA_DIR, FEATURE_SETS, TARGET
from feature_extraction import add_derived_features
from preprocessor import build_preprocessor
from split import split_train_test

logger = logging.getLogger(__name__)


def run():
    raw = load_raw()
    logger.info("Loaded %d raw rows", len(raw))

    cleaned = add_derived_features(clean(raw))
    logger.info("Cleaned: %d rows, diabetes prevalence %.2f%%", len(cleaned), 100 * cleaned[TARGET].mean())

    X_train, X_test, y_train, y_test = split_train_test(cleaned)
    logger.info("Train: %d rows, test: %d rows", len(X_train), len(X_test))

    DATA_DIR.mkdir(exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(DATA_DIR / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(DATA_DIR / "test.csv", index=False)

    for feature_set in FEATURE_SETS:
        preprocessor = build_preprocessor(feature_set)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        X_train_processed.assign(**{TARGET: y_train}).to_csv(
            DATA_DIR / f"train_processed_{feature_set}.csv", index=False
        )
        X_test_processed.assign(**{TARGET: y_test}).to_csv(
            DATA_DIR / f"test_processed_{feature_set}.csv", index=False
        )
        joblib.dump(preprocessor, DATA_DIR / f"preprocessor_{feature_set}.joblib")
        logger.info("%s: %d features -> %s", feature_set, X_train_processed.shape[1], list(X_train_processed.columns))

    logger.info("Saved splits and fitted preprocessors to %s", DATA_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
