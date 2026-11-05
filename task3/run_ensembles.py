"""Boosting and bagging for section 7.

Fits gradient-boosted trees (boosting) and a random forest (bagging) on the questionnaire
features, using the same folds, preprocessing, threshold rule and test set as section 5.
"""
import argparse
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from config import DATA_DIR, RANDOM_STATE, TASK_DIR
from data_exploration import GRID, SERIES_1, SERIES_2, TEXT_PRIMARY, TEXT_SECONDARY, style_axes
from preprocessor import build_preprocessor
from models import (
    choose_threshold,
    cross_validate_model,
    fit_and_predict,
    fold_ids,
    load_split,
    make_folds,
    make_pipeline,
    metrics_at_threshold,
    preprocessed_feature_names,
    save_predictions,
    score_probabilities,
    search_model,
    subsample,
)

logger = logging.getLogger(__name__)
FIGURES_DIR = TASK_DIR / "figures" / "section7"
SEARCH_SAMPLE = 300_000
SEARCH_FOLDS = 3
N_SPLITS = 5


def build_specs(feature_set):
    return [
        {
            "name": "Boosting (HGB)",
            "slug": "hgb",
            "pipeline": make_pipeline(
                HistGradientBoostingClassifier(
                    random_state=RANDOM_STATE,
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.1,
                ),
                feature_set,
            ),
            "param_grid": {
                "model__learning_rate": [0.05, 0.1],
                "model__max_iter": [150, 300],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [20, 50],
                "model__l2_regularization": [0.0, 1.0],
            },
            "search_jobs": 5,
            "cv_jobs": 1,
        },
        {
            "name": "Bagging (RF)",
            "slug": "rf",
            "pipeline": make_pipeline(
                RandomForestClassifier(
                    random_state=RANDOM_STATE, n_jobs=1, bootstrap=True
                ),
                feature_set,
            ),
            "param_grid": {
                "model__n_estimators": [200, 400],
                "model__max_features": ["sqrt", 0.3],
                "model__min_samples_leaf": [1, 5, 20],
                "model__max_depth": [None, 20],
                "model__class_weight": [None, "balanced_subsample"],
            },
            "search_jobs": 5,
            "cv_jobs": 1,
        },
    ]


def plot_boosting_curve(params, X, y, path):
    """Validation PR-AUC against the number of boosting iterations."""
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    iterations = [50, 100, 150, 200, 300, 400]
    scores = []
    for max_iter in iterations:
        model = HistGradientBoostingClassifier(
            random_state=RANDOM_STATE,
            early_stopping=False,
            max_iter=max_iter,
            learning_rate=params["model__learning_rate"],
            max_leaf_nodes=params["model__max_leaf_nodes"],
            min_samples_leaf=params["model__min_samples_leaf"],
            l2_regularization=params["model__l2_regularization"],
        )
        model.fit(X_fit, y_fit)
        scores.append(average_precision_score(y_valid, model.predict_proba(X_valid)[:, 1]))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(iterations, scores, marker="o", linewidth=1.5, color=SERIES_1)
    ax.set_xlabel("Boosting iterations", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Validation PR-AUC", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.set_title("Boosting: validation PR-AUC against iterations", loc="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_oob_curve(params, X, y, path):
    """Out-of-bag PR-AUC as trees are added to the forest."""
    increments = [50, 100, 150, 200, 300, 400]
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        warm_start=True,
        oob_score=True,
        bootstrap=True,
        n_jobs=-1,
        n_estimators=increments[0],
        max_features=params["model__max_features"],
        min_samples_leaf=params["model__min_samples_leaf"],
        max_depth=params["model__max_depth"],
        class_weight=params["model__class_weight"],
    )
    scores = []
    for trees in increments:
        model.set_params(n_estimators=trees)
        model.fit(X, y)
        oob = model.oob_decision_function_[:, 1]
        mask = np.isfinite(oob)
        scores.append(average_precision_score(y[mask], oob[mask]))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(increments, scores, marker="o", linewidth=1.5, color=SERIES_2)
    ax.set_xlabel("Number of trees", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("OOB PR-AUC", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.set_title("Bagging: out-of-bag PR-AUC against forest size", loc="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_search_heatmap(table, row_param, column_param, title, path):
    grouped = table.groupby([row_param, column_param])["mean_test_score"].max().reset_index()
    pivot = grouped.pivot(index=row_param, columns=column_param, values="mean_test_score")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    values = pivot.to_numpy()
    image = ax.imshow(values, cmap="viridis", aspect="auto")
    lower, upper = values.min(), values.max()
    span = upper - lower if upper > lower else 1.0
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.iloc[row, column]
            normalized = (value - lower) / span
            ax.text(column, row, f"{value:.4f}", ha="center", va="center",
                    fontsize=8, color="white" if normalized < 0.35 else "black")
    ax.set_xticks(range(pivot.shape[1]), [str(value) for value in pivot.columns], fontsize=8)
    ax.set_yticks(range(pivot.shape[0]), [str(value) for value in pivot.index], fontsize=8)
    ax.set_xlabel(column_param.replace("param_model__", ""), color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(row_param.replace("param_model__", ""), color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_importances(pipelines, X, y, path):
    names = list(pipelines["hgb"].named_steps["prep"].get_feature_names_out())
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for axis, slug, label in zip(
        axes, ["hgb", "rf"], ["Boosting (permutation)", "Bagging (impurity)"]
    ):
        pipeline = pipelines[slug]
        transformed = pipeline.named_steps["prep"].transform(X)
        model = pipeline.named_steps["model"]
        if slug == "hgb":
            values = permutation_importance(
                model, transformed, y, scoring="average_precision",
                n_repeats=5, random_state=RANDOM_STATE, n_jobs=5,
            ).importances_mean
        else:
            values = model.feature_importances_
        order = np.argsort(values)
        axis.barh(np.arange(len(order)), np.asarray(values)[order], color=SERIES_1 if slug == "hgb" else SERIES_2,
                  height=0.6)
        axis.set_yticks(np.arange(len(order)), [names[index] for index in order], fontsize=8)
        axis.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
        axis.set_title(label, loc="left", color=TEXT_PRIMARY, fontsize=10)
        style_axes(axis)
        axis.yaxis.grid(False)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_partial_dependence(pipeline, X, path):
    X = X.astype(float)
    features = ["age", "waistline", "bmi", "waist_to_height"]
    fig, axes = plt.subplots(1, len(features), figsize=(13, 3.2))
    display = PartialDependenceDisplay.from_estimator(
        pipeline, X, features=features, kind="average", ax=axes, n_jobs=1,
    )
    for axis, feature in zip(axes, features):
        axis.set_title(feature.replace("_", " "), loc="left", color=TEXT_PRIMARY, fontsize=10)
        style_axes(axis)
    fig.suptitle("Boosting partial dependence", x=0.01, ha="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default="questionnaire", choices=["questionnaire"])
    return parser.parse_args()


def main():
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, X_test, y_test = load_split(args.feature_set)
    logger.info("Train %d rows, test %d rows", len(y_train), len(y_test))
    feature_names = preprocessed_feature_names(args.feature_set, X_train.head(200))
    folds = make_folds(y_train)
    ids = fold_ids(y_train, folds)
    np.savez(DATA_DIR / "folds.npz", fold_id=ids)

    cv_rows, test_rows = [], []
    predictions, fitted, search_tables, best_params_by_slug = {}, {}, {}, {}
    for spec in build_specs(args.feature_set):
        name = spec["name"]
        logger.info("--- %s", name)
        X_search, y_search = subsample(X_train, y_train, SEARCH_SAMPLE)
        search_folds = make_folds(y_search, n_splits=SEARCH_FOLDS)
        best_params, table, best_score = search_model(
            spec["pipeline"], spec["param_grid"], X_search, y_search, search_folds,
            n_jobs=spec["search_jobs"],
        )
        table.to_csv(DATA_DIR / f"search_{spec['slug']}.csv", index=False)
        search_tables[spec["slug"]] = table
        best_params_by_slug[spec["slug"]] = best_params
        pipeline = spec["pipeline"].set_params(**best_params)
        if spec["slug"] == "rf":
            pipeline = pipeline.set_params(model__n_jobs=-1)
        logger.info("Best %s (CV PR-AUC %.4f)", best_params, best_score)

        cv_result = cross_validate_model(
            pipeline, X_train, y_train, folds, n_jobs=spec["cv_jobs"]
        )
        cv_result["folds"].to_csv(DATA_DIR / f"folds_{spec['slug']}.csv", index=False)
        threshold, target_met = choose_threshold(y_train, cv_result["oof"])
        cv_point = metrics_at_threshold(y_train, cv_result["oof"], threshold)
        model, test_probabilities, fit_seconds, predict_seconds = fit_and_predict(
            pipeline, X_train, y_train, X_test
        )
        test_metrics = {
            **score_probabilities(y_test, test_probabilities),
            **metrics_at_threshold(y_test, test_probabilities, threshold),
        }
        save_predictions(
            spec["slug"], args.feature_set, cv_result["oof"], test_probabilities,
            threshold, ids, DATA_DIR / f"preds_{spec['slug']}.npz",
        )
        predictions[name] = test_probabilities
        fitted[spec["slug"]] = model
        cv_rows.append({
            "model": name,
            **cv_result["summary"],
            "threshold": threshold,
            "cv_recall": cv_point["recall"],
            "cv_precision": cv_point["precision"],
            "cv_f2": cv_point["f2"],
            "cv_referral_rate": cv_point["referral_rate"],
            "cv_target_recall_met": target_met,
        })
        test_rows.append({
            "model": name,
            "pr_auc": test_metrics["pr_auc"],
            "roc_auc": test_metrics["roc_auc"],
            "brier": test_metrics["brier"],
            "log_loss": test_metrics["log_loss"],
            "threshold": threshold,
            "recall": test_metrics["recall"],
            "precision": test_metrics["precision"],
            "f2": test_metrics["f2"],
            "referral_rate": test_metrics["referral_rate"],
            "target_recall_met": test_metrics["recall"] >= 0.80,
            "fit_seconds": fit_seconds,
            "predict_ms_per_100k": 1000 * 100_000 * predict_seconds / len(y_test),
        })
        logger.info("CV PR-AUC %.4f ± %.4f | test PR-AUC %.4f | recall %.3f precision %.3f",
                    cv_result["summary"]["pr_auc_mean"], cv_result["summary"]["pr_auc_sd"],
                    test_metrics["pr_auc"], test_metrics["recall"], test_metrics["precision"])

    cv_table = pd.DataFrame(cv_rows)
    test_table = pd.DataFrame(test_rows)
    cv_table.to_csv(DATA_DIR / "ensemble_cv.csv", index=False)
    test_table.to_csv(DATA_DIR / "ensemble_test.csv", index=False)

    X_curve, y_curve = subsample(X_train, y_train, SEARCH_SAMPLE)
    curve_preprocessor = build_preprocessor(args.feature_set).fit(X_curve)
    X_curve_transformed = curve_preprocessor.transform(X_curve)
    plot_boosting_curve(
        best_params_by_slug["hgb"], X_curve_transformed, y_curve,
        FIGURES_DIR / "boosting_curve.png",
    )
    plot_oob_curve(
        best_params_by_slug["rf"], X_curve_transformed, y_curve,
        FIGURES_DIR / "bagging_oob.png",
    )
    plot_search_heatmap(
        search_tables["hgb"], "param_model__learning_rate", "param_model__max_leaf_nodes",
        "Boosting: PR-AUC by learning rate and leaves", FIGURES_DIR / "hgb_heatmap.png",
    )
    plot_search_heatmap(
        search_tables["rf"], "param_model__max_features", "param_model__min_samples_leaf",
        "Bagging: PR-AUC by features and leaf size", FIGURES_DIR / "rf_heatmap.png",
    )

    X_importance, y_importance = subsample(X_train, y_train, 50_000)
    plot_importances(fitted, X_importance, y_importance, FIGURES_DIR / "ensemble_importance.png")
    X_partial, _ = subsample(X_train, y_train, 20_000)
    plot_partial_dependence(fitted["hgb"], X_partial, FIGURES_DIR / "hgb_partial_dependence.png")

    print("\nCross-validated results:")
    print(cv_table.round(4).to_string(index=False))
    print("\nTest results:")
    print(test_table.round(4).to_string(index=False))
    logger.info("Saved section 7 tables and figures")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
