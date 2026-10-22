"""Clean the raw data, split it, fit the preprocessor on train and save every artifact to data/.

Outputs:
    train.csv / test.csv                      cleaned, unscaled (use with build_preprocessor() in a Pipeline)
    train_processed.csv / test_processed.csv  transformed with the train-fitted preprocessor
    preprocessor.joblib                       the fitted preprocessor
"""
import logging

import joblib

from cleaning import clean, load_raw
from config import DATA_DIR, TARGET
from preprocessor import build_preprocessor
from split import split_train_test

logger = logging.getLogger(__name__)


def run():
    raw = load_raw()
    logger.info("Loaded %d raw rows", len(raw))

    cleaned = clean(raw)
    logger.info("Cleaned: %d rows, diabetes prevalence %.2f%%", len(cleaned), 100 * cleaned[TARGET].mean())

    X_train, X_test, y_train, y_test = split_train_test(cleaned)
    logger.info("Train: %d rows, test: %d rows", len(X_train), len(X_test))

    preprocessor = build_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    DATA_DIR.mkdir(exist_ok=True)
    X_train.assign(**{TARGET: y_train}).to_csv(DATA_DIR / "train.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(DATA_DIR / "test.csv", index=False)
    X_train_processed.assign(**{TARGET: y_train}).to_csv(DATA_DIR / "train_processed.csv", index=False)
    X_test_processed.assign(**{TARGET: y_test}).to_csv(DATA_DIR / "test_processed.csv", index=False)
    joblib.dump(preprocessor, DATA_DIR / "preprocessor.joblib")
    logger.info("Saved splits and fitted preprocessor to %s", DATA_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run()
