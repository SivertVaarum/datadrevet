"""Teacher-student transfer for section 8.

A GAM trained on the full 28-feature set is the teacher. Its out-of-fold probabilities are soft
targets for a questionnaire-only HGB student. The student never sees the laboratory features at
prediction time.
"""
import argparse
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from config import DATA_DIR, RANDOM_STATE, TASK_DIR
from data_exploration import GRID, SERIES_1, SERIES_2, TEXT_PRIMARY, TEXT_SECONDARY, style_axes
from models import (
    choose_threshold,
    fold_ids,
    load_split,
    make_folds,
    metrics_at_threshold,
    save_predictions,
    score_probabilities,
)
from preprocessor import build_preprocessor

logger = logging.getLogger(__name__)
FIGURES_DIR = TASK_DIR / "figures" / "section8"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
TUNE_SAMPLE = 300_000
TUNE_FOLDS = 3
FINAL_FOLDS = 5


def best_hgb_params():
    path = DATA_DIR / "search_hgb.csv"
    if not path.exists():
        raise FileNotFoundError("Run run_ensembles.py first: search_hgb.csv is missing")
    row = pd.read_csv(path).sort_values("rank_test_score").iloc[0]
    return {
        "learning_rate": float(row["param_model__learning_rate"]),
        "max_iter": int(row["param_model__max_iter"]),
        "max_leaf_nodes": int(row["param_model__max_leaf_nodes"]),
        "min_samples_leaf": int(row["param_model__min_samples_leaf"]),
        "l2_regularization": float(row["param_model__l2_regularization"]),
    }


def make_student(params):
    return HistGradientBoostingClassifier(
        random_state=RANDOM_STATE,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        **params,
    )


def expand_soft_labels(X, y, teacher, alpha):
    """Represent soft targets as weighted positive/negative rows.

    For logistic loss, a soft target t is equivalent to a positive row with weight t and a
    negative row with weight 1-t. The true label is kept with weight 1-alpha.
    """
    weights_true = np.full(len(y), 1.0 - alpha)
    weights_positive = alpha * teacher
    weights_negative = alpha * (1.0 - teacher)
    X_all = np.vstack([X, X, X])
    y_all = np.concatenate([y, np.ones(len(y), dtype=int), np.zeros(len(y), dtype=int)])
    weights = np.concatenate([weights_true, weights_positive, weights_negative])
    keep = weights > 0
    return X_all[keep], y_all[keep], weights[keep]


def fit_predict_fold(X_train, y_train, teacher_train, X_valid, params, alpha):
    preprocessor = build_preprocessor("questionnaire").fit(X_train)
    transformed_train = np.asarray(preprocessor.transform(X_train))
    transformed_valid = np.asarray(preprocessor.transform(X_valid))
    X_expanded, y_expanded, weights = expand_soft_labels(
        transformed_train, y_train, teacher_train, alpha
    )
    model = make_student(params)
    model.fit(X_expanded, y_expanded, sample_weight=weights)
    return model.predict_proba(transformed_valid)[:, 1]


def cross_validate_soft(X, y, teacher, folds, params, alpha):
    oof = np.full(len(y), np.nan)
    scores = []
    for train_index, valid_index in folds:
        probabilities = fit_predict_fold(
            X.iloc[train_index], y[train_index], teacher[train_index],
            X.iloc[valid_index], params, alpha,
        )
        oof[valid_index] = probabilities
        scores.append(metrics_at_threshold(y[valid_index], probabilities, 0.5))
    from sklearn.metrics import average_precision_score
    return oof, average_precision_score(y, oof), np.std([
        average_precision_score(y[valid], oof[valid]) for _, valid in folds
    ])


def plot_alpha(table, path):
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.errorbar(table["alpha"], table["pr_auc"], yerr=table["pr_auc_sd"],
                marker="o", linewidth=1.5, capsize=3, color=SERIES_1)
    best = table.loc[table["pr_auc"].idxmax()]
    ax.scatter([best["alpha"]], [best["pr_auc"]], s=70, color=SERIES_2, zorder=3)
    ax.set_xlabel("Distillation weight α", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("CV PR-AUC", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.set_title("Transfer: soft-label weight against PR-AUC", loc="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-tuning", action="store_true",
                        help="reuse data/transfer_alpha.csv instead of rerunning the alpha sweep")
    args = parser.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    params = best_hgb_params()
    logger.info("Student HGB parameters: %s", params)

    X_train, y_train, X_test, y_test = load_split("questionnaire")
    teacher_oof = np.load(DATA_DIR / "preds_gam_full.npz", allow_pickle=True)["oof"]
    if len(teacher_oof) != len(y_train):
        raise ValueError("Teacher OOF predictions are not aligned with the training rows")

    sample_index, _ = train_test_split(
        np.arange(len(y_train)), train_size=TUNE_SAMPLE, stratify=y_train,
        random_state=RANDOM_STATE,
    )
    X_tune = X_train.iloc[sample_index].reset_index(drop=True)
    y_tune = y_train[sample_index]
    teacher_tune = teacher_oof[sample_index]

    tuning_path = DATA_DIR / "transfer_alpha.csv"
    if args.reuse_tuning and tuning_path.exists():
        tuning = pd.read_csv(tuning_path)
        logger.info("Reusing existing alpha sweep from %s", tuning_path)
    else:
        tuning_rows = []
        for alpha in ALPHAS:
            folds = make_folds(y_tune, n_splits=TUNE_FOLDS)
            _, score, spread = cross_validate_soft(
                X_tune, y_tune, teacher_tune, folds, params, alpha
            )
            tuning_rows.append({"alpha": alpha, "pr_auc": score, "pr_auc_sd": spread})
            logger.info("alpha %.2f: CV PR-AUC %.4f ± %.4f", alpha, score, spread)
        tuning = pd.DataFrame(tuning_rows)
        tuning.to_csv(tuning_path, index=False)
    plot_alpha(tuning, FIGURES_DIR / "transfer_alpha.png")
    best_alpha = float(tuning.loc[tuning["pr_auc"].idxmax(), "alpha"])
    logger.info("Best alpha: %.2f", best_alpha)

    folds = make_folds(y_train, n_splits=FINAL_FOLDS)
    oof = np.full(len(y_train), np.nan)
    fold_rows = []
    for fold, (train_index, valid_index) in enumerate(folds):
        probabilities = fit_predict_fold(
            X_train.iloc[train_index], y_train[train_index], teacher_oof[train_index],
            X_train.iloc[valid_index], params, best_alpha,
        )
        oof[valid_index] = probabilities
        fold_rows.append({
            "fold": fold,
            **score_probabilities(y_train[valid_index], probabilities),
            **metrics_at_threshold(y_train[valid_index], probabilities, 0.5),
        })
        logger.info("fold %d done", fold)
    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(DATA_DIR / "transfer_cv.csv", index=False)

    threshold, target_met = choose_threshold(y_train, oof, 0.80)
    cv_point = metrics_at_threshold(y_train, oof, threshold)

    preprocessor = build_preprocessor("questionnaire").fit(X_train)
    X_expanded, y_expanded, weights = expand_soft_labels(
        np.asarray(preprocessor.transform(X_train)), y_train, teacher_oof, best_alpha
    )
    model = make_student(params)
    started = time.perf_counter()
    model.fit(X_expanded, y_expanded, sample_weight=weights)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probabilities = model.predict_proba(np.asarray(preprocessor.transform(X_test)))[:, 1]
    predict_seconds = time.perf_counter() - started
    test_metrics = {
        **score_probabilities(y_test, probabilities),
        **metrics_at_threshold(y_test, probabilities, threshold),
    }

    save_predictions(
        "transfer", "questionnaire", oof, probabilities, threshold,
        fold_ids(y_train, folds), DATA_DIR / "preds_transfer.npz",
    )
    test_table = pd.DataFrame([{
        "model": "Transfer (distilled HGB)",
        "alpha": best_alpha,
        "threshold": threshold,
        **test_metrics,
        "cv_pr_auc": float(np.mean([row["pr_auc"] for row in fold_rows])),
        "cv_pr_auc_sd": float(np.std([row["pr_auc"] for row in fold_rows], ddof=0)),
        "cv_recall": cv_point["recall"],
        "cv_precision": cv_point["precision"],
        "cv_f2": cv_point["f2"],
        "cv_referral_rate": cv_point["referral_rate"],
        "cv_target_recall_met": target_met,
        "fit_seconds": fit_seconds,
        "predict_ms_per_100k": 1000 * 100_000 * predict_seconds / len(y_test),
    }])
    test_table.to_csv(DATA_DIR / "transfer_test.csv", index=False)
    print(test_table.round(4).to_string(index=False))
    logger.info("Saved transfer predictions and table")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
