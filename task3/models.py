"""Shared model fitting, evaluation and plotting for the section 5+ algorithms.

Everything that learns from data is fitted inside a Pipeline whose first step is the
train-fitted preprocessor, so cross-validation refits clipping, imputation and scaling on every
training fold. The test set is used once, after the threshold has been fixed on out-of-fold
predictions.
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    CONTINUOUS_FEATURES,
    DATA_DIR,
    FEATURE_SETS,
    RANDOM_STATE,
    SKEWED_FEATURES,
    TARGET,
    TASK_DIR,
)
from data_exploration import GRID, SERIES_1, SERIES_2, TEXT_PRIMARY, TEXT_SECONDARY, style_axes
from preprocessor import build_preprocessor

FIGURES_DIR = TASK_DIR / "figures" / "section5"
N_SPLITS = 5
TARGET_RECALL = 0.80
PRECISION_FLOOR = 0.20

MODEL_COLORS = {
    "Majority class": "#9a9a9a",
    "Logistic (untuned)": "#8fb8e8",
    "Logistic regression": SERIES_1,
    "GAM": SERIES_2,
    "Decision tree": "#2e8b57",
}

DISPLAY_NAMES = {
    "smoking_status_2": "smoking (former)",
    "smoking_status_3": "smoking (current)",
    "waist_to_height": "waist-to-height ratio",
}


def display_name(feature):
    return DISPLAY_NAMES.get(feature, feature.replace("_", " "))


# --------------------------------------------------------------------------------------
# Data and pipelines
# --------------------------------------------------------------------------------------

def load_split(feature_set="questionnaire"):
    """Load the cleaned, unscaled train/test split. The preprocessor selects the feature set."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X_train = train.drop(columns=[TARGET])
    X_test = test.drop(columns=[TARGET])
    return X_train, train[TARGET].to_numpy(), X_test, test[TARGET].to_numpy()


def make_pipeline(model, feature_set="questionnaire"):
    """Preprocessor first, so every CV fold fits clipping/imputation/scaling on its own train part."""
    return Pipeline([("prep", build_preprocessor(feature_set)), ("model", model)])


def preprocessed_feature_names(feature_set, X_sample):
    """Column names the fitted preprocessor produces, in order. A small sample is enough."""
    return list(build_preprocessor(feature_set).fit(X_sample).get_feature_names_out())


def gam_feature_lists(feature_set, names):
    """Split preprocessor output into smooth (continuous) and linear (binary/ordinal) terms."""
    continuous = set(CONTINUOUS_FEATURES) | set(SKEWED_FEATURES)
    spline = [name for name in names if name in continuous]
    linear = [name for name in names if name not in continuous]
    return spline, linear


def subsample(X, y, size, random_state=RANDOM_STATE):
    """Stratified subsample used only for hyperparameter search, never for the final fit."""
    if size is None or size >= len(y):
        return X, y
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=size, stratify=y, random_state=random_state
    )
    return X_sample, y_sample


# --------------------------------------------------------------------------------------
# Folds, metrics and thresholds
# --------------------------------------------------------------------------------------

def make_folds(y, n_splits=N_SPLITS):
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(len(y)), y))


def fold_ids(y, folds):
    ids = np.full(len(y), -1, dtype=int)
    for fold, (_, valid) in enumerate(folds):
        ids[valid] = fold
    return ids


def score_probabilities(y, probabilities):
    return {
        "pr_auc": average_precision_score(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "brier": brier_score_loss(y, probabilities),
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
    }


def choose_threshold(y, probabilities, target_recall=TARGET_RECALL):
    """Largest threshold whose recall is still at least the target.

    A larger threshold refers fewer patients and gives higher precision; the constraint keeps
    recall at the level required by section 1.4. If the target is unreachable, fall back to the
    threshold with the best F2-score so the model still has a defined operating point.
    """
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    if len(thresholds) == 0:
        return 0.5, False

    valid = np.where(recall[:-1] >= target_recall)[0]
    if valid.size:
        index = int(valid[-1])
        return float(thresholds[index]), True

    f2 = 5 * precision[:-1] * recall[:-1] / (4 * precision[:-1] + recall[:-1] + 1e-12)
    index = int(np.nanargmax(f2))
    return float(thresholds[index]), False


def metrics_at_threshold(y, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "threshold": float(threshold),
        "recall": recall,
        "precision": precision,
        "f2": fbeta_score(y, predicted, beta=2, zero_division=0),
        "referral_rate": float(predicted.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# --------------------------------------------------------------------------------------
# Cross-validation and final fit
# --------------------------------------------------------------------------------------

def _fit_fold(pipeline, X, y, train_index, valid_index):
    model = clone(pipeline)
    started = time.perf_counter()
    model.fit(X.iloc[train_index], y[train_index])
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probabilities = model.predict_proba(X.iloc[valid_index])[:, 1]
    predict_seconds = time.perf_counter() - started
    return valid_index, probabilities, fit_seconds, predict_seconds


def cross_validate_model(pipeline, X, y, folds, n_jobs=1):
    """Manual loop so we keep out-of-fold probabilities, fold metrics and timings together."""
    arguments = [(pipeline, X, y, train_index, valid_index) for train_index, valid_index in folds]
    if n_jobs == 1:
        results = [_fit_fold(*argument) for argument in arguments]
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(_fit_fold)(*argument) for argument in arguments)

    oof = np.full(len(y), np.nan)
    rows = []
    for valid_index, probabilities, fit_seconds, predict_seconds in results:
        oof[valid_index] = probabilities
        rows.append({
            **score_probabilities(y[valid_index], probabilities),
            "fit_seconds": fit_seconds,
            "predict_seconds": predict_seconds,
            "n_valid": len(valid_index),
        })

    fold_table = pd.DataFrame(rows)
    summary = {
        "pr_auc_mean": fold_table["pr_auc"].mean(),
        "pr_auc_sd": fold_table["pr_auc"].std(ddof=0),
        "roc_auc_mean": fold_table["roc_auc"].mean(),
        "roc_auc_sd": fold_table["roc_auc"].std(ddof=0),
        "brier_mean": fold_table["brier"].mean(),
        "log_loss_mean": fold_table["log_loss"].mean(),
        "fit_seconds_mean": fold_table["fit_seconds"].mean(),
        "predict_ms_per_100k": 1000 * 100_000 * (
            fold_table["predict_seconds"] / fold_table["n_valid"]
        ).mean(),
    }
    return {"oof": oof, "folds": fold_table, "summary": summary}


def fit_and_predict(pipeline, X_train, y_train, X_test):
    model = clone(pipeline)
    started = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    probabilities = model.predict_proba(X_test)[:, 1]
    predict_seconds = time.perf_counter() - started
    return model, probabilities, fit_seconds, predict_seconds


def search_model(pipeline, param_grid, X, y, folds, n_jobs=1):
    """Grid search on the given folds. The caller decides whether to pass a subsample."""
    search = GridSearchCV(
        pipeline, param_grid, scoring="average_precision", cv=folds,
        n_jobs=n_jobs, refit=True, return_train_score=False,
    )
    search.fit(X, y)
    table = pd.DataFrame(search.cv_results_)
    return search.best_params_, table, float(search.best_score_)


def save_predictions(name, feature_set, oof, test, threshold, ids, path):
    np.savez_compressed(
        path, oof=oof, test=test, threshold=np.array(threshold),
        fold_id=ids, feature_set=np.array([feature_set]),
    )


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------

def _color(name, index):
    return MODEL_COLORS.get(name, plt.cm.tab10(index % 10))


def plot_pr_curves(y, predictions, thresholds, path):
    prevalence = float(np.mean(y))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for index, (name, probabilities) in enumerate(predictions.items()):
        score = average_precision_score(y, probabilities)
        color = _color(name, index)
        if np.unique(probabilities).size == 1:
            # A constant classifier has one operating point, not a curve.
            point = metrics_at_threshold(y, probabilities, thresholds.get(name, probabilities[0]))
            ax.scatter([point["recall"]], [point["precision"]], s=50, color=color,
                       edgecolor="white", linewidth=0.8, zorder=4,
                       label=f"{name} (AP {score:.3f})")
            continue
        precision, recall, _ = precision_recall_curve(y, probabilities)
        ax.plot(recall, precision, linewidth=1.5, color=color,
                label=f"{name} (AP {score:.3f})")
        if name in thresholds:
            point = metrics_at_threshold(y, probabilities, thresholds[name])
            ax.scatter([point["recall"]], [point["precision"]], s=42, color=color,
                       edgecolor="white", linewidth=0.8, zorder=4)

    ax.axhline(prevalence, color=TEXT_SECONDARY, linewidth=0.9, linestyle="--",
               label=f"No skill ({prevalence:.3f})")
    ax.set_xlabel("Recall", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Precision", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Precision-recall curves, test set", loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_roc_curves(y, predictions, thresholds, path):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for index, (name, probabilities) in enumerate(predictions.items()):
        score = roc_auc_score(y, probabilities)
        color = _color(name, index)
        if np.unique(probabilities).size == 1:
            point = metrics_at_threshold(y, probabilities, thresholds.get(name, probabilities[0]))
            false_positive_rate = 1 - point["tn"] / (point["tn"] + point["fp"])
            ax.scatter([false_positive_rate], [point["recall"]], s=50, color=color,
                       edgecolor="white", linewidth=0.8, zorder=4,
                       label=f"{name} (AUC {score:.3f})")
            continue
        false_positive, true_positive, _ = roc_curve(y, probabilities)
        ax.plot(false_positive, true_positive, linewidth=1.5, color=color,
                label=f"{name} (AUC {score:.3f})")
        if name in thresholds:
            point = metrics_at_threshold(y, probabilities, thresholds[name])
            ax.scatter([1 - point["tn"] / (point["tn"] + point["fp"])], [point["recall"]],
                       s=42, color=color, edgecolor="white", linewidth=0.8, zorder=4)

    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=0.9, linestyle="--")
    ax.set_xlabel("False positive rate", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("True positive rate", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("ROC curves, test set", loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_calibration(y, predictions, path):
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    for index, (name, probabilities) in enumerate(predictions.items()):
        observed, predicted = calibration_curve(y, probabilities, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", markersize=3.5, linewidth=1.3,
                color=_color(name, index), label=name)
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=0.9, linestyle="--")
    ax.set_xlabel("Predicted probability", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Observed diabetes rate", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(0, 0.8)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Calibration, test set", loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def coefficient_table(pipeline, X_train, path, figure_path=None):
    figure_path = figure_path or FIGURES_DIR / "lr_coefficients.png"
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    coefficients = pipeline.named_steps["model"].coef_[0]
    continuous = set(CONTINUOUS_FEATURES) | set(SKEWED_FEATURES)

    rows = []
    for name, coefficient in zip(names, coefficients):
        standard_deviation = float(X_train[name].std(ddof=0)) if name in X_train else np.nan
        per_unit = coefficient / standard_deviation if name in continuous else coefficient
        rows.append({
            "feature": name,
            "display_name": display_name(name),
            "coef_per_sd": coefficient,
            "odds_ratio_per_sd": float(np.exp(coefficient)),
            "train_sd": standard_deviation,
            "coef_per_unit": per_unit,
            "odds_ratio_per_unit": float(np.exp(per_unit)),
        })
    table = pd.DataFrame(rows).sort_values("coef_per_sd")
    table.to_csv(path, index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = [SERIES_2 if value > 0 else SERIES_1 for value in table["coef_per_sd"]]
    positions = np.arange(len(table))
    ax.barh(positions, table["coef_per_sd"], color=colors, height=0.6)
    for position, (value, ratio) in enumerate(zip(table["coef_per_sd"], table["odds_ratio_per_sd"])):
        ax.annotate(
            f"{ratio:.2f}×", (value, position),
            xytext=(-5 if value < 0 else 5, 0), textcoords="offset points",
            ha="right" if value < 0 else "left", va="center",
            fontsize=8, color=TEXT_SECONDARY,
        )
    ax.set_yticks(positions, table["display_name"], fontsize=9)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.9)
    ax.margins(x=0.18)
    ax.set_xlabel("Coefficient (log-odds per 1 SD)", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_title("Logistic regression coefficients", loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return table


def plot_gam_partial_effects(pipeline, feature_names, X_train, path):
    gam = pipeline.named_steps["model"].gam_
    continuous = set(CONTINUOUS_FEATURES) | set(SKEWED_FEATURES)
    spline_indices = [index for index, name in enumerate(feature_names) if name in continuous]

    columns = 3
    rows = int(np.ceil(len(spline_indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.2 * rows), squeeze=False)
    for panel, term_index in enumerate(spline_indices):
        axis = axes[panel // columns][panel % columns]
        name = feature_names[term_index]
        grid = gam.generate_X_grid(term=term_index)
        effect, intervals = gam.partial_dependence(term=term_index, X=grid, width=0.95)

        standard_deviation = float(X_train[name].std(ddof=0))
        mean = float(X_train[name].mean())
        x_values = grid[:, term_index] * standard_deviation + mean

        axis.plot(x_values, effect, color=SERIES_1, linewidth=1.6)
        axis.fill_between(x_values, intervals[:, 0], intervals[:, 1], color=SERIES_1, alpha=0.18,
                          linewidth=0)
        axis.axhline(0, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")
        axis.set_title(display_name(name), loc="left", color=TEXT_PRIMARY, fontsize=10)
        style_axes(axis)
        axis.yaxis.grid(True, color=GRID, linewidth=0.8)
        axis.xaxis.grid(False)

    for empty in range(len(spline_indices), rows * columns):
        axes[empty // columns][empty % columns].set_visible(False)

    for axis in axes[-1]:
        axis.set_xlabel("Feature value", color=TEXT_SECONDARY, fontsize=9)
    for axis in axes[:, 0]:
        axis.set_ylabel("Partial effect (log-odds)", color=TEXT_SECONDARY, fontsize=9)

    fig.suptitle("GAM partial effects on the log-odds scale (95 % bands)", x=0.01, ha="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tree_diagram(pipeline, feature_names, path, max_depth=3):
    from sklearn.tree import plot_tree

    model = pipeline.named_steps["model"]
    labels = [display_name(name) for name in feature_names]
    fig, ax = plt.subplots(figsize=(16, 9.5))
    plot_tree(
        model, max_depth=max_depth, feature_names=labels,
        class_names=["No diabetes", "Diabetes"], filled=True, rounded=True,
        proportion=True, impurity=True, fontsize=7, ax=ax,
    )
    ax.set_title(f"Decision tree, first {max_depth} levels "
                 f"(depth {model.get_depth()}, {model.get_n_leaves()} leaves)",
                 loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tree_depth_curve(search_table, path):
    table = search_table.copy()
    table = table[table["param_model__max_depth"].notna()]
    table["depth"] = table["param_model__max_depth"].astype(int)
    # One line per leaf size; average the two class-weight settings at each depth so the plot
    # does not draw a vertical segment between them.
    table = (table.groupby(["depth", "param_model__min_samples_leaf"], as_index=False)
             ["mean_test_score"].max())
    leaves = sorted(table["param_model__min_samples_leaf"].unique())

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    for index, leaf in enumerate(leaves):
        subset = table[table["param_model__min_samples_leaf"] == leaf].sort_values("depth")
        ax.plot(subset["depth"], subset["mean_test_score"], marker="o", linewidth=1.3,
                color=plt.cm.viridis(index / max(len(leaves) - 1, 1)),
                label=f"min samples/leaf {leaf}")
    ax.set_xlabel("Maximum depth", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("CV PR-AUC", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Decision tree: depth against PR-AUC (best class weighting)",
                 loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def tree_importance_table(pipeline, X_sample, y_sample, path, figure_path):
    """Impurity and permutation importance for the fitted tree.

    Impurity importance is computed from the tree itself. Permutation importance is measured on a
    held-out sample with the PR-AUC drop, so it answers how much each column adds to the model.
    """
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    model = pipeline.named_steps["model"]
    # Permute the columns the tree actually sees, so names and importances line up. A DataFrame
    # keeps the preprocessor's column names and avoids sklearn's feature-name warning.
    transformed = pipeline.named_steps["prep"].transform(X_sample)
    permutation = permutation_importance(
        model, transformed, y_sample, scoring="average_precision",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=5,
    )
    table = pd.DataFrame({
        "feature": names,
        "display_name": [display_name(name) for name in names],
        "impurity": model.feature_importances_,
        "permutation_mean": permutation.importances_mean,
        "permutation_std": permutation.importances_std,
    }).sort_values("permutation_mean", ascending=False)
    table.to_csv(path, index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    positions = np.arange(len(table))
    ax.barh(positions, table["permutation_mean"], height=0.6, color=SERIES_1,
            xerr=table["permutation_std"], error_kw={"elinewidth": 0.9, "ecolor": TEXT_SECONDARY})
    ax.set_yticks(positions, table["display_name"], fontsize=9)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.9)
    ax.set_xlabel("Drop in PR-AUC when shuffled", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_title("Decision tree permutation importance", loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return table


def gam_edof_table(pipeline, path):
    """Effective degrees of freedom per additive term, a compact summary of GAM flexibility."""
    gam = pipeline.named_steps["model"].gam_
    names = list(pipeline.named_steps["prep"].get_feature_names_out())
    edof_per_coefficient = np.asarray(gam.statistics_["edof_per_coef"])

    rows = []
    offset = 0
    for term, name in zip(gam.terms, names):
        count = term.n_coefs
        rows.append({
            "feature": name,
            "display_name": display_name(name),
            "n_coefficients": count,
            "edof": float(edof_per_coefficient[offset:offset + count].sum()),
        })
        offset += count
    table = pd.DataFrame(rows)
    table.to_csv(path, index=False)
    return table
