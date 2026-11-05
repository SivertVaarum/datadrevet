"""Section 6 comparisons from the saved out-of-fold and test predictions.

Reads data/preds_<model>[_<feature_set>].npz and data/test.csv, then writes the comparison
tables and figures in data/ and figures/section6/. No model is refitted here.
"""
import argparse
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    log_loss,
    roc_auc_score,
)

from config import DATA_DIR, FEATURE_SETS, RANDOM_STATE, TARGET, TASK_DIR
from data_exploration import GRID, SERIES_1, SERIES_2, TEXT_PRIMARY, TEXT_SECONDARY, style_axes

logger = logging.getLogger(__name__)

FIGURES_DIR = TASK_DIR / "figures" / "section6"
N_BOOT = 1_000
SUBGROUP_BOOT = 400
CALIBRATION_BOOT = 400
N_JOBS = 5

BASE_MODELS = {
    "Logistic regression": "logistic",
    "GAM": "gam",
    "Decision tree": "tree",
}
MODEL_COLORS = {
    "Logistic regression": SERIES_1,
    "GAM": SERIES_2,
    "Decision tree": "#2e8b57",
}
FEATURE_SETS_DISPLAY = {
    "questionnaire": "Questionnaire",
    "equipment": "Questionnaire + equipment",
    "full": "Full",
}
FEATURE_SUFFIX = {"questionnaire": "", "equipment": "_equipment", "full": "_full"}


def load_test():
    test = pd.read_csv(DATA_DIR / "test.csv")
    return test[TARGET].to_numpy(), test


def load_predictions(slug, feature_set="questionnaire"):
    path = DATA_DIR / f"preds_{slug}{FEATURE_SUFFIX[feature_set]}.npz"
    data = np.load(path, allow_pickle=True)
    return {
        "test": data["test"],
        "oof": data["oof"],
        "threshold": float(data["threshold"]),
        "feature_set": str(data["feature_set"][0]),
    }


def metrics_at(y, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f2 = fbeta_score(y, predicted, beta=2, zero_division=0)
    return {
        "recall": recall,
        "precision": precision,
        "f2": f2,
        "referral_rate": float(predicted.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def ranking_metrics(y, probabilities):
    return {
        "pr_auc": average_precision_score(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "brier": brier_score_loss(y, probabilities),
        "log_loss": log_loss(y, probabilities, labels=[0, 1]),
    }


def model_table(y, models):
    rows = []
    for name, values in models.items():
        row = {
            "model": name,
            **ranking_metrics(y, values["test"]),
            **metrics_at(y, values["test"], values["threshold"]),
            "threshold": values["threshold"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def stratified_indices(y, rng):
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    return np.concatenate([
        rng.choice(positive, size=len(positive), replace=True),
        rng.choice(negative, size=len(negative), replace=True),
    ])


def bootstrap_models(y, models, n_boot=N_BOOT, n_jobs=N_JOBS):
    """Paired stratified bootstrap over the test set for the three main models."""
    names = list(models)

    def one(replicate):
        rng = np.random.default_rng(RANDOM_STATE + replicate)
        index = stratified_indices(y, rng)
        y_boot = y[index]
        row = {"replicate": replicate}
        for name in names:
            probabilities = models[name]["test"][index]
            row[f"{name}|pr_auc"] = average_precision_score(y_boot, probabilities)
            row[f"{name}|roc_auc"] = roc_auc_score(y_boot, probabilities)
            point = metrics_at(y_boot, probabilities, models[name]["threshold"])
            row[f"{name}|recall"] = point["recall"]
            row[f"{name}|precision"] = point["precision"]
            row[f"{name}|f2"] = point["f2"]
        return row

    started = time.perf_counter()
    rows = Parallel(n_jobs=n_jobs)(delayed(one)(replicate) for replicate in range(n_boot))
    logger.info("Bootstrap: %d replicates in %.1f s", n_boot, time.perf_counter() - started)
    return pd.DataFrame(rows)


def pair_summaries(bootstrap):
    pairs = [
        ("GAM", "Logistic regression"),
        ("Decision tree", "Logistic regression"),
        ("GAM", "Decision tree"),
    ]
    metrics = ["pr_auc", "roc_auc", "recall", "precision", "f2"]
    rows = []
    for first, second in pairs:
        for metric in metrics:
            difference = bootstrap[f"{first}|{metric}"].to_numpy() - bootstrap[f"{second}|{metric}"].to_numpy()
            rows.append({
                "comparison": f"{first} − {second}",
                "metric": metric,
                "mean": float(np.mean(difference)),
                "ci_lower": float(np.percentile(difference, 2.5)),
                "ci_upper": float(np.percentile(difference, 97.5)),
                "probability_positive": float(np.mean(difference > 0)),
            })
    return pd.DataFrame(rows)


def threshold_curves(y, models, n_thresholds=200):
    thresholds = np.linspace(0.005, 0.5, n_thresholds)
    rows = []
    for name, values in models.items():
        for threshold in thresholds:
            point = metrics_at(y, values["test"], threshold)
            rows.append({
                "model": name,
                "threshold": threshold,
                "precision": point["precision"],
                "recall": point["recall"],
                "referral_rate": point["referral_rate"],
                "f2": point["f2"],
            })
    return pd.DataFrame(rows)


def gains_curve(y, models, n_points=100):
    rows = []
    total_positive = int(y.sum())
    for name, values in models.items():
        order = np.argsort(-values["test"])
        cumulative = np.cumsum(y[order])
        points = np.linspace(1, len(y), n_points, dtype=int) - 1
        for index in points:
            rows.append({
                "model": name,
                "fraction_referred": (index + 1) / len(y),
                "fraction_captured": cumulative[index] / total_positive,
            })
    return pd.DataFrame(rows)


def subgroup_masks(test):
    age = test["age"]
    bmi = test["bmi"]
    male = test["male"] == 1
    return {
        "Age <40": age < 40,
        "Age 40–59": (age >= 40) & (age < 60),
        "Age 60+": age >= 60,
        "Female": ~male,
        "Male": male,
        "BMI <25": bmi < 25,
        "BMI 25–30": (bmi >= 25) & (bmi < 30),
        "BMI 30+": bmi >= 30,
    }


def subgroup_table(y, test, models):
    rows = []
    for label, mask in subgroup_masks(test).items():
        y_group = y[mask]
        if y_group.size == 0 or np.unique(y_group).size < 2:
            continue
        for name, values in models.items():
            probabilities = values["test"][mask]
            point = metrics_at(y_group, probabilities, values["threshold"])
            rng = np.random.default_rng(RANDOM_STATE)
            recalls, precisions = [], []
            for _ in range(SUBGROUP_BOOT):
                index = stratified_indices(y_group, rng)
                boot = metrics_at(y_group[index], probabilities[index], values["threshold"])
                recalls.append(boot["recall"])
                precisions.append(boot["precision"])
            rows.append({
                "subgroup": label,
                "n": int(mask.sum()),
                "prevalence": float(y_group.mean()),
                "model": name,
                "recall": point["recall"],
                "recall_lower": float(np.percentile(recalls, 2.5)),
                "recall_upper": float(np.percentile(recalls, 97.5)),
                "precision": point["precision"],
                "precision_lower": float(np.percentile(precisions, 2.5)),
                "precision_upper": float(np.percentile(precisions, 97.5)),
            })
    return pd.DataFrame(rows)


def calibration_table(y, models):
    rows = []
    eps = 1e-6
    for name, values in models.items():
        probabilities = values["test"]
        clipped = np.clip(probabilities, eps, 1 - eps)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        regressor = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
        regressor.fit(logit, y)
        brier = brier_score_loss(y, probabilities)
        loss = log_loss(y, probabilities, labels=[0, 1])

        rng = np.random.default_rng(RANDOM_STATE)
        brier_boot, loss_boot = [], []
        for _ in range(CALIBRATION_BOOT):
            index = stratified_indices(y, rng)
            brier_boot.append(brier_score_loss(y[index], probabilities[index]))
            loss_boot.append(log_loss(y[index], probabilities[index], labels=[0, 1]))

        rows.append({
            "model": name,
            "brier": brier,
            "brier_lower": float(np.percentile(brier_boot, 2.5)),
            "brier_upper": float(np.percentile(brier_boot, 97.5)),
            "log_loss": loss,
            "log_loss_lower": float(np.percentile(loss_boot, 2.5)),
            "log_loss_upper": float(np.percentile(loss_boot, 97.5)),
            "calibration_slope": float(regressor.coef_[0, 0]),
            "calibration_intercept": float(regressor.intercept_[0]),
        })
    return pd.DataFrame(rows)


def calibration_deciles(y, models):
    rows = []
    for name, values in models.items():
        frame = pd.DataFrame({"p": values["test"], "y": y})
        frame["decile"] = pd.qcut(frame["p"], 10, labels=False, duplicates="drop")
        grouped = frame.groupby("decile").agg(
            count=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
        ).reset_index()
        grouped["model"] = name
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def feature_set_table(y):
    rows = []
    for feature_set in ["questionnaire", "equipment", "full"]:
        for name, slug in BASE_MODELS.items():
            try:
                values = load_predictions(slug, feature_set)
            except FileNotFoundError:
                logger.warning("Missing predictions for %s/%s", name, feature_set)
                continue
            point = {
                "model": name,
                "feature_set": feature_set,
                "n_features": len(FEATURE_SETS[feature_set]) + 1,
                **ranking_metrics(y, values["test"]),
                **metrics_at(y, values["test"], values["threshold"]),
                "threshold": values["threshold"],
            }
            rows.append(point)
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    baseline = table[table["feature_set"] == "questionnaire"].set_index("model")
    table["delta_pr_auc"] = table.apply(
        lambda row: row["pr_auc"] - baseline.loc[row["model"], "pr_auc"], axis=1
    )
    table["delta_precision"] = table.apply(
        lambda row: row["precision"] - baseline.loc[row["model"], "precision"], axis=1
    )
    return table


def fold_differences():
    rows = []
    frames = {name: pd.read_csv(DATA_DIR / f"folds_{slug}.csv") for name, slug in BASE_MODELS.items()}
    for first, second in [
        ("GAM", "Logistic regression"),
        ("Decision tree", "Logistic regression"),
        ("GAM", "Decision tree"),
    ]:
        for metric in ["pr_auc", "roc_auc"]:
            difference = frames[first][metric].to_numpy() - frames[second][metric].to_numpy()
            rows.append({
                "comparison": f"{first} − {second}",
                "metric": metric,
                "mean_difference": float(np.mean(difference)),
                "sd_difference": float(np.std(difference, ddof=0)),
                "fold_differences": ";".join(f"{value:.6f}" for value in difference),
            })
    return pd.DataFrame(rows)


def plot_paired_differences(summary, path):
    metrics = {"pr_auc": "PR-AUC", "precision": "Precision"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    comparisons = ["GAM − Logistic regression", "Decision tree − Logistic regression", "GAM − Decision tree"]
    for axis, (metric, label) in zip(axes, metrics.items()):
        subset = summary[summary["metric"] == metric].set_index("comparison").loc[comparisons]
        positions = np.arange(len(comparisons))
        axis.errorbar(
            subset["mean"], positions,
            xerr=[subset["mean"] - subset["ci_lower"], subset["ci_upper"] - subset["mean"]],
            fmt="o", color=SERIES_1, ecolor=TEXT_SECONDARY, capsize=4, markersize=6,
        )
        axis.axvline(0, color=TEXT_SECONDARY, linewidth=0.9, linestyle="--")
        axis.set_yticks(positions, comparisons, fontsize=8)
        axis.set_xlabel(f"Δ{label} (95 % bootstrap CI)", color=TEXT_SECONDARY, fontsize=9)
        style_axes(axis)
        axis.yaxis.grid(False)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    axes[0].set_title("Paired test-set differences", loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_threshold_tradeoff(curves, models, path):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharex=True)
    for name in models:
        subset = curves[curves["model"] == name]
        for axis, metric in zip(axes, ["precision", "recall", "referral_rate"]):
            axis.plot(subset["threshold"], subset[metric], linewidth=1.4,
                      color=MODEL_COLORS[name], label=name)
            axis.axvline(models[name]["threshold"], color=MODEL_COLORS[name],
                         linewidth=0.8, linestyle=":")
    for axis, title in zip(axes, ["Precision", "Recall", "Referral rate"]):
        axis.set_xlabel("Decision threshold", color=TEXT_SECONDARY, fontsize=9)
        axis.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=10)
        axis.set_ylim(0, 1.02)
        style_axes(axis)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_gains(gains, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linewidth=0.9, linestyle="--",
            label="Random referral")
    for name, color in MODEL_COLORS.items():
        subset = gains[gains["model"] == name]
        ax.plot(subset["fraction_referred"], subset["fraction_captured"],
                linewidth=1.6, color=color, label=name)
    ax.axhline(0.8, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
    ax.axvline(0.45, color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
    ax.set_xlabel("Fraction of patients referred", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Fraction of diabetics captured", color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Cumulative gains on the test set", loc="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_subgroups(subgroups, path):
    categories = {
        "Age": ["Age <40", "Age 40–59", "Age 60+"],
        "Sex": ["Female", "Male"],
        "BMI": ["BMI <25", "BMI 25–30", "BMI 30+"],
    }
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey="row")
    width = 0.8 / len(BASE_MODELS)
    for column, (category, labels) in enumerate(categories.items()):
        positions = np.arange(len(labels))
        for row, metric in enumerate(["recall", "precision"]):
            axis = axes[row][column]
            for offset, name in enumerate(BASE_MODELS):
                values = []
                lower = []
                upper = []
                for label in labels:
                    subset = subgroups[(subgroups["subgroup"] == label) & (subgroups["model"] == name)]
                    if subset.empty:
                        values.append(np.nan); lower.append(np.nan); upper.append(np.nan)
                    else:
                        item = subset.iloc[0]
                        values.append(item[metric])
                        lower.append(item[metric] - item[f"{metric}_lower"])
                        upper.append(item[f"{metric}_upper"] - item[metric])
                axis.bar(positions + (offset - 1) * width, values, width,
                         color=MODEL_COLORS[name], label=name,
                         yerr=[lower, upper], capsize=2, error_kw={"elinewidth": 0.8})
            axis.set_xticks(positions, labels, fontsize=8, rotation=20, ha="right")
            axis.set_ylim(0, 1.02 if metric == "recall" else 0.25)
            axis.set_title(f"{category}: {metric}", loc="left", color=TEXT_PRIMARY, fontsize=10)
            style_axes(axis)
            axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_feature_sets(table, path):
    order = ["questionnaire", "equipment", "full"]
    if table.empty or set(order) - set(table["feature_set"]):
        return
    labels = [FEATURE_SETS_DISPLAY[name] for name in order]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    for axis, metric, title in zip(
        axes, ["pr_auc", "precision"], ["PR-AUC", "Precision at the frozen threshold"]
    ):
        for name, color in MODEL_COLORS.items():
            subset = table[table["model"] == name].set_index("feature_set").loc[order]
            axis.plot(range(len(order)), subset[metric], marker="o", linewidth=1.5,
                      color=color, label=name)
        axis.set_xticks(range(len(order)), labels, fontsize=8, rotation=15, ha="right")
        axis.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=10)
        style_axes(axis)
        axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--skip-subgroups", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y, test = load_test()

    models = {name: load_predictions(slug) for name, slug in BASE_MODELS.items()}
    logger.info("Loaded %d test rows for %d models", len(y), len(models))

    model_table(y, models).to_csv(DATA_DIR / "comparison_ranking.csv", index=False)

    operating = model_table(y, models).copy()
    operating["nns"] = 1 / operating["precision"]
    operating["fp_per_tp"] = operating["fp"] / operating["tp"]
    operating.to_csv(DATA_DIR / "comparison_operating_point.csv", index=False)

    fold_differences().to_csv(DATA_DIR / "comparison_fold_differences.csv", index=False)

    curves = threshold_curves(y, models)
    curves.to_csv(DATA_DIR / "comparison_thresholds.csv", index=False)
    plot_threshold_tradeoff(curves, models, FIGURES_DIR / "threshold_tradeoff.png")

    gains = gains_curve(y, models)
    gains.to_csv(DATA_DIR / "comparison_gains.csv", index=False)
    plot_gains(gains, FIGURES_DIR / "cumulative_gains.png")

    calibration = calibration_table(y, models)
    calibration.to_csv(DATA_DIR / "comparison_calibration.csv", index=False)
    calibration_deciles(y, models).to_csv(DATA_DIR / "comparison_calibration_deciles.csv", index=False)

    if not args.skip_subgroups:
        subgroups = subgroup_table(y, test, models)
        subgroups.to_csv(DATA_DIR / "comparison_subgroups.csv", index=False)
        plot_subgroups(subgroups, FIGURES_DIR / "subgroup_performance.png")
    else:
        subgroups = pd.DataFrame()

    if not args.skip_bootstrap:
        bootstrap = bootstrap_models(y, models)
        bootstrap.to_csv(DATA_DIR / "comparison_bootstrap_replicates.csv", index=False)
        summary = pair_summaries(bootstrap)
        summary.to_csv(DATA_DIR / "comparison_bootstrap.csv", index=False)
        plot_paired_differences(summary, FIGURES_DIR / "paired_differences.png")
    else:
        summary = pd.DataFrame()

    feature_sets = feature_set_table(y)
    feature_sets.to_csv(DATA_DIR / "comparison_feature_sets.csv", index=False)
    plot_feature_sets(feature_sets, FIGURES_DIR / "feature_set_comparison.png")

    print("\nTest ranking and operating point:")
    print(operating.round(4).to_string(index=False))
    if not summary.empty:
        print("\nPaired bootstrap differences:")
        print(summary.round(4).to_string(index=False))
    if not subgroups.empty:
        print("\nSubgroup performance:")
        print(subgroups.round(4).to_string(index=False))
    if not feature_sets.empty:
        print("\nFeature-set comparison:")
        print(feature_sets.round(4).to_string(index=False))
    print("\nCalibration:")
    print(calibration.round(4).to_string(index=False))
    logger.info("Saved comparison tables to %s and figures to %s", DATA_DIR, FIGURES_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
