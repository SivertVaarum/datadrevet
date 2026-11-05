"""Final comparison for section 8: basic vs boosting vs bagging vs transfer."""
import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, TASK_DIR
from data_exploration import GRID, SERIES_1, SERIES_2, TEXT_PRIMARY, TEXT_SECONDARY, style_axes
from models import (
    MODEL_COLORS,
    metrics_at_threshold,
    plot_calibration,
    plot_pr_curves,
    plot_roc_curves,
)
from run_comparison import bootstrap_models, load_predictions, load_test, ranking_metrics

logger = logging.getLogger(__name__)
FIGURES_DIR = TASK_DIR / "figures" / "section8"
REFERENCE = "Basic GAM"
N_BOOT = 1_000

MODEL_SPECS = {
    "Basic GAM": ("gam", "questionnaire"),
    "Boosting (HGB)": ("hgb", "questionnaire"),
    "Bagging (RF)": ("rf", "questionnaire"),
    "Transfer (student)": ("transfer", "questionnaire"),
}
TEACHER = "Teacher (full GAM)"


def load_models():
    models = {}
    for name, (slug, feature_set) in MODEL_SPECS.items():
        models[name] = load_predictions(slug, feature_set)
    teacher = load_predictions("gam", "full")
    return models, teacher


def comparison_table(y, models):
    rows = []
    for name, values in models.items():
        point = {
            "model": name,
            **ranking_metrics(y, values["test"]),
            **metrics_at_threshold(y, values["test"], values["threshold"]),
            "threshold": values["threshold"],
        }
        point["nns"] = 1 / point["precision"] if point["precision"] else np.nan
        point["fp_per_tp"] = point["fp"] / point["tp"] if point["tp"] else np.nan
        rows.append(point)
    return pd.DataFrame(rows)


def timing_table():
    rows = []
    basic = pd.read_csv(DATA_DIR / "basic_model_test.csv")
    gam = basic[basic["model"] == "GAM"].iloc[0]
    rows.append({"model": "Basic GAM", "fit_seconds": gam["fit_seconds"],
                 "predict_ms_per_100k": gam["predict_ms_per_100k"]})
    ensembles = pd.read_csv(DATA_DIR / "ensemble_test.csv")
    for name in ["Boosting (HGB)", "Bagging (RF)"]:
        row = ensembles[ensembles["model"] == name].iloc[0]
        rows.append({"model": name, "fit_seconds": row["fit_seconds"],
                     "predict_ms_per_100k": row["predict_ms_per_100k"]})
    transfer = pd.read_csv(DATA_DIR / "transfer_test.csv").iloc[0]
    rows.append({"model": "Transfer (student)", "fit_seconds": transfer["fit_seconds"],
                 "predict_ms_per_100k": transfer["predict_ms_per_100k"]})
    return pd.DataFrame(rows)


def pair_summaries(bootstrap):
    pairs = [
        ("Boosting (HGB)", REFERENCE),
        ("Bagging (RF)", REFERENCE),
        ("Transfer (student)", REFERENCE),
        ("Boosting (HGB)", "Transfer (student)"),
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


def plot_pairs(summary, path):
    metrics = {"pr_auc": "PR-AUC", "precision": "Precision"}
    comparisons = [
        "Boosting (HGB) − Basic GAM",
        "Bagging (RF) − Basic GAM",
        "Transfer (student) − Basic GAM",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), sharey=True)
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
        axis.set_xlabel(f"Δ{label} vs basic GAM", color=TEXT_SECONDARY, fontsize=9)
        style_axes(axis)
        axis.yaxis.grid(False)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_metric_bars(table, path):
    metrics = [("pr_auc", "PR-AUC"), ("precision", "Precision"), ("f2", "F2")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=False)
    order = table["model"].tolist()
    for axis, (metric, title) in zip(axes, metrics):
        values = table[metric].to_numpy()
        colors = [MODEL_COLORS.get(name, SERIES_1) for name in order]
        axis.barh(np.arange(len(order)), values, color=colors, height=0.6)
        axis.set_yticks(np.arange(len(order)), order, fontsize=8)
        axis.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=10)
        style_axes(axis)
        axis.yaxis.grid(False)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_time_performance(table, timing, path):
    merged = table.merge(timing, on="model", how="left")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for _, row in merged.iterrows():
        ax.scatter(row["fit_seconds"], row["pr_auc"], s=60,
                   color=MODEL_COLORS.get(row["model"], SERIES_1))
        ax.annotate(row["model"], (row["fit_seconds"], row["pr_auc"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=8,
                    color=TEXT_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlabel("Fit time (s, log scale)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Test PR-AUC", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.set_title("Performance against training cost", loc="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_transfer(table, teacher_name, path):
    names = ["Basic GAM", "Transfer (student)", teacher_name]
    metrics = [("pr_auc", "PR-AUC"), ("precision", "Precision")]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    colors = [MODEL_COLORS.get(name, SERIES_1) for name in names]
    for axis, (metric, title) in zip(axes, metrics):
        values = [table.loc[table["model"] == name, metric].iloc[0] for name in names]
        axis.bar(np.arange(len(names)), values, color=colors, width=0.55)
        axis.set_xticks(np.arange(len(names)), names, fontsize=8, rotation=15, ha="right")
        axis.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=10)
        style_axes(axis)
        axis.yaxis.grid(True, color=GRID, linewidth=0.8)
    fig.suptitle("Transfer: basic student, distilled student and lab teacher",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    y, _ = load_test()
    models, teacher = load_models()

    MODEL_COLORS.update({
        "Basic GAM": SERIES_2,
        "Boosting (HGB)": "#2a78d6",
        "Bagging (RF)": "#2e8b57",
        "Transfer (student)": "#7b4fa0",
        TEACHER: "#9a9a9a",
    })

    table = comparison_table(y, models)
    table.to_csv(DATA_DIR / "final_comparison.csv", index=False)
    timing = timing_table()
    timing.to_csv(DATA_DIR / "final_timing.csv", index=False)

    teacher_point = {
        "model": TEACHER,
        **ranking_metrics(y, teacher["test"]),
        **metrics_at_threshold(y, teacher["test"], teacher["threshold"]),
    }
    teacher_table = pd.DataFrame([teacher_point])
    teacher_table["nns"] = 1 / teacher_table["precision"]
    teacher_table["fp_per_tp"] = teacher_table["fp"] / teacher_table["tp"]
    teacher_table.to_csv(DATA_DIR / "teacher_comparison.csv", index=False)

    predictions = {name: values["test"] for name, values in models.items()}
    predictions[TEACHER] = teacher["test"]
    thresholds = {name: values["threshold"] for name, values in models.items()}
    thresholds[TEACHER] = teacher["threshold"]

    plot_pr_curves(y, predictions, thresholds, FIGURES_DIR / "pr_curves.png")
    plot_roc_curves(y, predictions, thresholds, FIGURES_DIR / "roc_curves.png")
    plot_calibration(y, predictions, FIGURES_DIR / "calibration.png")
    plot_metric_bars(table, FIGURES_DIR / "metric_bars.png")
    plot_time_performance(table, timing, FIGURES_DIR / "time_performance.png")
    plot_transfer(
        pd.concat([table[["model", "pr_auc", "precision"]], teacher_table[["model", "pr_auc", "precision"]]],
                  ignore_index=True),
        TEACHER, FIGURES_DIR / "transfer_comparison.png",
    )

    bootstrap = bootstrap_models(y, models, n_boot=N_BOOT)
    bootstrap.to_csv(DATA_DIR / "final_bootstrap_replicates.csv", index=False)
    summary = pair_summaries(bootstrap)
    summary.to_csv(DATA_DIR / "final_bootstrap.csv", index=False)
    plot_pairs(summary, FIGURES_DIR / "paired_differences.png")

    print("\nFinal comparison:")
    print(table.round(4).to_string(index=False))
    print("\nTeacher reference:")
    print(teacher_table.round(4).to_string(index=False))
    print("\nTiming:")
    print(timing.round(4).to_string(index=False))
    print("\nPaired bootstrap vs basic GAM:")
    print(summary.round(4).to_string(index=False))
    logger.info("Saved section 8 tables and figures")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
