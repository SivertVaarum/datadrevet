"""Fit the three section-5 algorithms and the section-1.4 baselines on one feature set.

Usage:
    python run_algorithms.py                       # questionnaire set, full data
    python run_algorithms.py --quick               # 50k train / 10k test, for a fast check
    python run_algorithms.py --feature-set full    # comparison model (section 6)

Outputs (with a tag suffix for non-final runs):
    data/basic_model_cv.csv          out-of-fold metrics per model
    data/basic_model_test.csv        held-out test metrics at the frozen threshold
    data/thresholds.csv              operating point and confusion matrix per model
    data/folds.npz                   fold assignment, reused by section 6
    data/preds_<model>.npz           OOF and test probabilities, for section 6
    data/search_<model>.csv          grid-search results
    data/lr_coefficients.csv         coefficients and odds ratios
    figures/section5/pr_curves_basic.png, roc_curves_basic.png, calibration_basic.png,
    figures/section5/lr_coefficients.png, gam_partial_effects.png, tree_depth_curve.png,
    figures/section5/decision_tree.png, tree_importances.png
"""
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from config import DATA_DIR, FEATURE_SETS, RANDOM_STATE
from models import (
    FIGURES_DIR,
    TARGET_RECALL,
    choose_threshold,
    coefficient_table,
    cross_validate_model,
    fit_and_predict,
    fold_ids,
    gam_edof_table,
    gam_feature_lists,
    load_split,
    make_folds,
    make_pipeline,
    metrics_at_threshold,
    plot_calibration,
    plot_gam_partial_effects,
    plot_pr_curves,
    plot_roc_curves,
    plot_tree_depth_curve,
    plot_tree_diagram,
    preprocessed_feature_names,
    save_predictions,
    score_probabilities,
    search_model,
    subsample,
    tree_importance_table,
)
from pygam_model import PyGAMClassifier

logger = logging.getLogger(__name__)

QUICK_TRAIN = 50_000
QUICK_TEST = 10_000


def build_specs(feature_set, spline_features, linear_features):
    """Model definitions, search spaces and the compute settings for each fit."""
    return [
        {
            "name": "Majority class", "slug": "majority",
            "pipeline": make_pipeline(DummyClassifier(strategy="prior"), feature_set),
            "param_grid": None, "search_sample": None, "search_folds": None,
            "search_jobs": 1, "cv_jobs": 1,
        },
        {
            "name": "Logistic (untuned)", "slug": "logistic_untuned",
            "pipeline": make_pipeline(LogisticRegression(max_iter=2000), feature_set),
            "param_grid": None, "search_sample": None, "search_folds": None,
            "search_jobs": 1, "cv_jobs": 5,
        },
        {
            "name": "Logistic regression", "slug": "logistic",
            "pipeline": make_pipeline(LogisticRegression(max_iter=2000), feature_set),
            "param_grid": {
                "model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                "model__class_weight": [None, "balanced"],
            },
            "search_sample": None, "search_folds": 5, "search_jobs": 5, "cv_jobs": 5,
        },
        {
            "name": "GAM", "slug": "gam",
            "pipeline": make_pipeline(
                PyGAMClassifier(
                    spline_features=tuple(spline_features),
                    linear_features=tuple(linear_features),
                ),
                feature_set,
            ),
            "param_grid": {
                "model__n_splines": [5, 8, 12],
                "model__lam": [0.1, 0.6, 1.0, 10.0],
            },
            # pygam fits are single-threaded and hold ~3.7 GB on the full data, so tune on a
            # sample and run the five CV folds sequentially.
            "search_sample": 150_000, "search_folds": 3, "search_jobs": 1, "cv_jobs": 1,
        },
        {
            "name": "Decision tree", "slug": "tree",
            "pipeline": make_pipeline(
                DecisionTreeClassifier(random_state=RANDOM_STATE), feature_set
            ),
            "param_grid": {
                "model__max_depth": [4, 6, 8, 10, 12, None],
                "model__min_samples_leaf": [200, 1000, 5000, 10_000],
                "model__class_weight": [None, "balanced"],
            },
            "search_sample": 300_000, "search_folds": 3, "search_jobs": 5, "cv_jobs": 5,
        },
    ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default="questionnaire", choices=list(FEATURE_SETS))
    parser.add_argument("--quick", action="store_true", help="small stratified sample for a fast check")
    parser.add_argument("--tag", default="", help="suffix for output files, e.g. quick")
    return parser.parse_args()


def main():
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ("_quick" if args.quick else "")
    figure_dir = FIGURES_DIR / "quick" if args.quick else FIGURES_DIR
    figure_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Feature set: %s%s", args.feature_set, " (quick)" if args.quick else "")

    X_train, y_train, X_test, y_test = load_split(args.feature_set)
    if args.quick:
        X_train, y_train = subsample(X_train, y_train, QUICK_TRAIN)
        X_test, y_test = subsample(X_test, y_test, QUICK_TEST)
    logger.info("Train %d rows (%.2f%% positive), test %d rows",
                len(y_train), 100 * y_train.mean(), len(y_test))

    feature_names = preprocessed_feature_names(args.feature_set, X_train.head(200))
    spline_features, linear_features = gam_feature_lists(args.feature_set, feature_names)
    logger.info("Preprocessed columns: %s", feature_names)

    folds = make_folds(y_train)
    ids = fold_ids(y_train, folds)
    np.savez(DATA_DIR / f"folds{suffix}.npz", fold_id=ids)

    cv_rows, test_rows, threshold_rows = [], [], []
    predictions, fitted, search_tables = {}, {}, {}

    for spec in build_specs(args.feature_set, spline_features, linear_features):
        name = spec["name"]
        logger.info("--- %s", name)

        if spec["param_grid"]:
            X_search, y_search = subsample(X_train, y_train, spec["search_sample"])
            search_folds = make_folds(y_search, n_splits=min(spec["search_folds"], len(y_search)))
            best_params, table, best_score = search_model(
                spec["pipeline"], spec["param_grid"], X_search, y_search, search_folds,
                n_jobs=spec["search_jobs"],
            )
            table.to_csv(DATA_DIR / f"search_{spec['slug']}{suffix}.csv", index=False)
            search_tables[spec["slug"]] = table
            pipeline = spec["pipeline"].set_params(**best_params)
            logger.info("Best %s (CV PR-AUC %.4f on %d rows)", best_params, best_score, len(y_search))
        else:
            pipeline = spec["pipeline"]

        cv_result = cross_validate_model(pipeline, X_train, y_train, folds, n_jobs=spec["cv_jobs"])
        cv_result["folds"].to_csv(DATA_DIR / f"folds_{spec['slug']}{suffix}.csv", index=False)
        threshold, target_met = choose_threshold(y_train, cv_result["oof"], TARGET_RECALL)
        cv_point = metrics_at_threshold(y_train, cv_result["oof"], threshold)

        model, test_probabilities, fit_seconds, predict_seconds = fit_and_predict(
            pipeline, X_train, y_train, X_test
        )
        test_metrics = {
            **score_probabilities(y_test, test_probabilities),
            **metrics_at_threshold(y_test, test_probabilities, threshold),
        }
        test_target_met = test_metrics["recall"] >= TARGET_RECALL

        save_predictions(
            spec["slug"], args.feature_set, cv_result["oof"], test_probabilities, threshold,
            ids, DATA_DIR / f"preds_{spec['slug']}{suffix}.npz",
        )
        predictions[name] = test_probabilities
        fitted[spec["slug"]] = model

        cv_rows.append({
            "model": name,
            "feature_set": args.feature_set,
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
            "feature_set": args.feature_set,
            "pr_auc": test_metrics["pr_auc"],
            "roc_auc": test_metrics["roc_auc"],
            "brier": test_metrics["brier"],
            "log_loss": test_metrics["log_loss"],
            "threshold": threshold,
            "recall": test_metrics["recall"],
            "precision": test_metrics["precision"],
            "f2": test_metrics["f2"],
            "referral_rate": test_metrics["referral_rate"],
            "target_recall_met": test_target_met,
            "fit_seconds": fit_seconds,
            "predict_ms_per_100k": 1000 * 100_000 * predict_seconds / len(y_test),
        })
        threshold_rows.append({
            "model": name,
            "split": "test",
            **test_metrics,
            "target_recall_met": test_target_met,
        })

        logger.info("CV PR-AUC %.4f ± %.4f | threshold %.4f | test recall %.3f precision %.3f",
                    cv_result["summary"]["pr_auc_mean"], cv_result["summary"]["pr_auc_sd"],
                    threshold, test_metrics["recall"], test_metrics["precision"])

    cv_table = pd.DataFrame(cv_rows)
    test_table = pd.DataFrame(test_rows)
    threshold_table = pd.DataFrame(threshold_rows)
    cv_table.to_csv(DATA_DIR / f"basic_model_cv{suffix}.csv", index=False)
    test_table.to_csv(DATA_DIR / f"basic_model_test{suffix}.csv", index=False)
    threshold_table.to_csv(DATA_DIR / f"thresholds{suffix}.csv", index=False)
    cv_table[cv_table["model"].isin(["Majority class", "Logistic (untuned)"])].to_csv(
        DATA_DIR / f"baselines{suffix}.csv", index=False
    )

    thresholds = dict(zip(test_table["model"], test_table["threshold"]))
    plot_pr_curves(y_test, predictions, thresholds, figure_dir / f"pr_curves_basic{suffix}.png")
    plot_roc_curves(y_test, predictions, thresholds, figure_dir / f"roc_curves_basic{suffix}.png")
    plot_calibration(y_test, predictions, figure_dir / f"calibration_basic{suffix}.png")

    coefficient_table(
        fitted["logistic"], X_train, DATA_DIR / f"lr_coefficients{suffix}.csv",
        figure_path=figure_dir / f"lr_coefficients{suffix}.png",
    )
    gam_feature_names = list(fitted["gam"].named_steps["prep"].get_feature_names_out())
    plot_gam_partial_effects(
        fitted["gam"], gam_feature_names, X_train,
        figure_dir / f"gam_partial_effects{suffix}.png",
    )
    tree_feature_names = list(fitted["tree"].named_steps["prep"].get_feature_names_out())
    plot_tree_diagram(
        fitted["tree"], tree_feature_names, figure_dir / f"decision_tree{suffix}.png"
    )
    if "tree" in search_tables:
        plot_tree_depth_curve(search_tables["tree"], figure_dir / f"tree_depth_curve{suffix}.png")

    gam_edof_table(fitted["gam"], DATA_DIR / f"gam_edof{suffix}.csv")
    X_importance, y_importance = subsample(X_train, y_train, 50_000)
    tree_importance_table(
        fitted["tree"], X_importance, y_importance,
        DATA_DIR / f"tree_importances{suffix}.csv",
        figure_dir / f"tree_importances{suffix}.png",
    )

    print("\nCross-validated (out-of-fold) results:")
    print(cv_table.round(4).to_string(index=False))
    print("\nTest results at the frozen recall-%.0f%% threshold:" % (100 * TARGET_RECALL))
    print(test_table.round(4).to_string(index=False))
    print("\nOperating points:")
    print(threshold_table.round(4).to_string(index=False))
    logger.info("Saved tables to %s and figures to %s", DATA_DIR, figure_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
