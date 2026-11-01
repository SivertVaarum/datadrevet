"""sklearn-compatible wrapper around pygam's LogisticGAM.

pygam's estimator has ``get_params``/``clone`` support, but its ``predict_proba`` returns a
one-dimensional vector and it does not expose ``classes_``. This wrapper makes it behave like a
standard sklearn classifier so it can be a step in a Pipeline and be used by GridSearchCV and
cross-validation. It also resolves column names, because the preprocessor outputs a DataFrame.
"""
import numpy as np
from pygam import LogisticGAM, l, s
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class PyGAMClassifier(ClassifierMixin, BaseEstimator):
    """Additive logistic model: smooth terms for continuous features, linear terms for the rest.

    Parameters mirror the pygam knobs used in the report: the number of spline basis functions per
    smooth term and the smoothing penalty ``lam``. The feature lists are names in the
    preprocessor's output, resolved to column indices in ``fit``.
    """

    def __init__(self, n_splines=8, lam=0.6, spline_features=(), linear_features=(),
                 max_iter=100, tol=1e-4, verbose=False):
        self.n_splines = n_splines
        self.lam = lam
        self.spline_features = spline_features
        self.linear_features = linear_features
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

    def _resolve_indices(self, X):
        if hasattr(X, "columns"):
            names = list(X.columns)
        else:
            names = [f"x{index}" for index in range(np.asarray(X).shape[1])]
        spline_indices = [names.index(name) for name in self.spline_features if name in names]
        linear_indices = [names.index(name) for name in self.linear_features if name in names]
        if not spline_indices and not linear_indices:
            spline_indices = list(range(len(names)))
        return names, spline_indices, linear_indices

    def _build_terms(self):
        terms = None
        for index in self.spline_indices_:
            term = s(index, n_splines=self.n_splines)
            terms = term if terms is None else terms + term
        for index in self.linear_indices_:
            term = l(index)
            terms = term if terms is None else terms + term
        return terms

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=int)
        if np.unique(target).size < 2:
            raise ValueError("PyGAMClassifier needs both classes in the training fold")

        self.feature_names_in_, self.spline_indices_, self.linear_indices_ = self._resolve_indices(X)
        self.n_features_in_ = values.shape[1]
        self.classes_ = np.array([0, 1])
        self.gam_ = LogisticGAM(
            self._build_terms(), lam=self.lam, max_iter=self.max_iter, tol=self.tol,
            verbose=self.verbose,
        )
        self.gam_.fit(values, target)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "gam_")
        positive = np.asarray(self.gam_.predict_proba(np.asarray(X, dtype=float)))
        if positive.ndim == 2:
            positive = positive[:, 1]
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def term_names(self):
        """Names of the fitted design-matrix columns, in preprocessor order."""
        check_is_fitted(self, "feature_names_in_")
        return list(self.feature_names_in_)
