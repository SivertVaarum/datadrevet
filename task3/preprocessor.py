"""Fitted preprocessing steps. Everything here learns statistics, so it must be fitted on the
training split only - ideally as the first step of a model Pipeline so cross-validation refits it
per fold."""
import numpy as np
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from config import (
    BINARY_FEATURES,
    CONTINUOUS_FEATURES,
    ORDINAL_FEATURES,
    SKEWED_FEATURES,
    SMOKING_CATEGORIES,
    SMOKING_FEATURE,
)


class QuantileClipper(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    """Winsorize each column to quantile bounds learned on the training data.

    Clipping instead of deleting keeps every patient, while stopping extreme but plausible
    values from dominating distance- and gradient-based models (KNN, SVM, neural networks).
    NaNs are ignored when fitting and passed through unchanged.
    """

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        values = np.asarray(X, dtype=float)
        self.n_features_in_ = values.shape[1]
        self.lower_bounds_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_bounds_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, ["lower_bounds_", "upper_bounds_"])
        return np.clip(np.asarray(X, dtype=float), self.lower_bounds_, self.upper_bounds_)


def build_preprocessor():
    """Return an unfitted ColumnTransformer that outputs a pandas DataFrame.

    - skewed labs:  log1p -> clip -> median impute -> standardize
    - continuous:   clip -> median impute -> standardize
    - ordinal:      mode impute -> standardize
    - binary flags: mode impute, left as 0/1
    - smoking:      one-hot, 'never' dropped as the reference category
    """
    skewed = Pipeline([
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    continuous = Pipeline([
        ("clip", QuantileClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="most_frequent")
    smoking = OneHotEncoder(categories=[SMOKING_CATEGORIES], drop="first", sparse_output=False)

    preprocessor = ColumnTransformer(
        [
            ("skewed", skewed, SKEWED_FEATURES),
            ("continuous", continuous, CONTINUOUS_FEATURES),
            ("ordinal", ordinal, ORDINAL_FEATURES),
            ("binary", binary, BINARY_FEATURES),
            ("smoking", smoking, [SMOKING_FEATURE]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")
