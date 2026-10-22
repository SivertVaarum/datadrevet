"""Exploration of the raw dataset that motivates the preprocessing decisions (report section 2.1).

Prints the statistics cited in the report and saves figures to figures/.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cleaning import load_raw
from config import (
    DIABETES_THRESHOLD_MG_DL,
    GLUCOSE_COLUMN,
    PLAUSIBLE_RANGES,
    SIGHT_BLIND_CODE,
    SKEWED_FEATURES,
    TASK_DIR,
)

FIGURES_DIR = TASK_DIR / "figures"

# Reference palette slots 1-2 (validated adjacent pair) and recessive ink for text and axes.
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def report_overview(df):
    print(f"Rows: {len(df)}, columns: {df.shape[1]}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Exact duplicate rows: {df.duplicated().sum()}")
    diabetes = df[GLUCOSE_COLUMN] >= DIABETES_THRESHOLD_MG_DL
    print(f"Diabetes prevalence (BLDS >= {DIABETES_THRESHOLD_MG_DL}): {100 * diabetes.mean():.2f}% "
          f"({diabetes.sum()} of {len(df)})")


def report_measurement_resolution(df):
    for column in ("height", "weight"):
        print(f"{column}: {100 * (df[column] % 5 == 0).mean():.1f}% of values are multiples of 5")
    print(f"waistline: {100 * (df['waistline'] % 1 != 0).mean():.1f}% of values have decimals")


def report_sentinels_and_implausible(df):
    for side in ("left", "right"):
        print(f"sight_{side} == {SIGHT_BLIND_CODE} (blindness code): "
              f"{np.isclose(df[f'sight_{side}'], SIGHT_BLIND_CODE).sum()}")
    total = 0
    for column, (low, high) in PLAUSIBLE_RANGES.items():
        outside = ~df[column].between(low, high)
        total += outside.sum()
        if outside.any():
            print(f"{column}: {outside.sum()} outside [{low}, {high}] "
                  f"(min {df[column].min()}, max {df[column].max()})")
    print(f"Total implausible values: {total}")


def report_categoricals(df):
    for column in ("sex", "DRK_YN", "SMK_stat_type_cd", "hear_left", "urine_protein"):
        shares = (100 * df[column].value_counts(normalize=True).sort_index()).round(1)
        print(f"{column}: {shares.to_dict()}")


def skewness_table(df):
    table = pd.DataFrame({
        "raw": [df[c].skew() for c in SKEWED_FEATURES],
        "log1p": [np.log1p(df[c]).skew() for c in SKEWED_FEATURES],
    }, index=SKEWED_FEATURES)
    print("Skewness of lab values (implausible values removed):")
    print(table.round(2).to_string())
    return table


def plot_skewness(table, path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = np.arange(len(table))
    height = 0.36
    ax.barh(y - height / 2 - 0.01, table["raw"], height, color=SERIES_1, label="Raw")
    ax.barh(y + height / 2 + 0.01, table["log1p"], height, color=SERIES_2, label="After log(1 + x)")
    for i, (raw, logged) in enumerate(zip(table["raw"], table["log1p"])):
        ax.text(raw + 0.3, i - height / 2, f"{raw:.1f}", va="center", fontsize=8, color=TEXT_SECONDARY)
        ax.text(logged + 0.3, i + height / 2, f"{logged:.1f}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(y, table.index)
    ax.invert_yaxis()
    ax.set_xlabel("Skewness", color=TEXT_SECONDARY, fontsize=9)
    ax.set_title("Skewness of lab values before and after log transform", loc="left",
                 color=TEXT_PRIMARY, fontsize=11)
    style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def prevalence_by(df, column, bins, labels):
    groups = pd.cut(df[column], bins=bins, labels=labels, right=False)
    return 100 * df.groupby(groups, observed=True)["diabetes"].mean()


def plot_prevalence(df, path):
    """Diabetes prevalence by questionnaire risk factor, on one shared percentage axis."""
    df = df.assign(
        diabetes=df[GLUCOSE_COLUMN] >= DIABETES_THRESHOLD_MG_DL,
        bmi=df["weight"] / (df["height"] / 100) ** 2,
    )
    panels = [
        ("Age (years)", prevalence_by(df, "age", [0, 30, 40, 50, 60, 70, 200],
                                      ["<30", "30-39", "40-49", "50-59", "60-69", "70+"])),
        # WHO Asia-Pacific BMI categories, appropriate for the Korean population.
        ("BMI (kg/m²)", prevalence_by(df, "bmi", [0, 18.5, 23, 25, 30, 200],
                                      ["<18.5", "18.5-23", "23-25", "25-30", "30+"])),
        ("Waist (cm)", prevalence_by(df, "waistline", [0, 70, 80, 90, 100, 1000],
                                     ["<70", "70-79", "80-89", "90-99", "100+"])),
    ]
    base_rate = 100 * df["diabetes"].mean()

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, (title, rates) in zip(axes, panels):
        ax.bar(rates.index.astype(str), rates.values, width=0.6, color=SERIES_1)
        for x, rate in enumerate(rates.values):
            ax.text(x, rate + 0.4, f"{rate:.0f}%", ha="center", fontsize=8, color=TEXT_SECONDARY)
        ax.axhline(base_rate, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontsize=10)
        style_axes(ax)
    axes[0].set_ylabel("Diabetes prevalence (%)", color=TEXT_SECONDARY, fontsize=9)
    axes[0].text(-0.4, base_rate + 0.4, f"overall {base_rate:.1f}%",
                 ha="left", fontsize=8, color=TEXT_SECONDARY)
    fig.suptitle("Diabetes prevalence rises with age, BMI and waist circumference",
                 x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    df = load_raw()
    FIGURES_DIR.mkdir(exist_ok=True)

    report_overview(df)
    report_measurement_resolution(df)
    report_sentinels_and_implausible(df)
    report_categoricals(df)

    plausible = df.copy()
    for column in SKEWED_FEATURES:
        low, high = PLAUSIBLE_RANGES[column]
        plausible.loc[~plausible[column].between(low, high), column] = np.nan
    plot_skewness(skewness_table(plausible), FIGURES_DIR / "skewness.png")

    valid_waist = df["waistline"].between(*PLAUSIBLE_RANGES["waistline"])
    valid_glucose = df[GLUCOSE_COLUMN].between(*PLAUSIBLE_RANGES[GLUCOSE_COLUMN])
    plot_prevalence(df[valid_waist & valid_glucose], FIGURES_DIR / "prevalence_by_risk_factor.png")
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
