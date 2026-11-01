"""Feature selection and dimensionality reduction for the questionnaire model (report section 4).

Four selection methods (mutual information, permutation importance, L1 logistic regression and
RFECV) and one reduction method (PCA) are fitted on the training split only, then compared in a
single ablation. Selection is refitted inside every cross-validation fold, so the validation fold
never influences which features are kept and the test set is never touched.

Outputs:
    figures/section4/feature_correlation.png   Spearman correlation of the ten questionnaire features
    figures/section4/pca_scree.png             explained variance per component and cumulative
    figures/section4/pca_scatter.png           first two components, colored by label
    figures/section4/feature_ranking.png       mutual information and permutation importance
    figures/section4/rfecv_curve.png           CV PR-AUC against the number of selected features
    figures/section4/pca_truncation.png        CV PR-AUC against the number of retained components
    data/feature_scores.csv           per-feature MI, permutation importance and L1 entry point
    data/l1_path.csv                  non-zero coefficients per regularization strength
    data/pca_variance.csv             explained variance ratio and cumulative variance
    data/selection_ablation.csv       PR-AUC / ROC-AUC / recall at 20 % precision per variant
    data/knn_pca.csv                  KNN on raw features versus rotated components
    data/full_feature_mi.csv          mutual information for the 28-feature comparison model
"""
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_DIR, RANDOM_STATE, TARGET, TASK_DIR
from data_exploration import GRID, SERIES_1, SERIES_2, TEXT_PRIMARY, TEXT_SECONDARY, style_axes

logger = logging.getLogger(__name__)
FIGURES_DIR = TASK_DIR / "figures" / "section4"

N_SPLITS = 5
KNN_SPLITS = 3
PRECISION_FLOOR = 0.20  # screening operating point: at least one referral in five is a true case
SUBSAMPLE = 100_000  # mutual information uses kNN density estimation, so it is subsampled
L1_SUBSAMPLE = 200_000  # liblinear is single-core, so the L1 path is fitted on a subsample
KNN_SUBSAMPLE = 30_000  # KNN prediction is linear in the training set size
SCATTER_SAMPLE = 20_000
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
REFERENCE_C = 0.1  # sparsest strength on the L1 path that is still stable across the grid

# One-hot columns are meaningless to a reader, so give them a clinical label in tables and figures.
DISPLAY_NAMES = {
    "smoking_status_2": "smoking (former)",
    "smoking_status_3": "smoking (current)",
}


def display_name(feature):
    return DISPLAY_NAMES.get(feature, feature.replace("_", " "))


def load_questionnaire():
    """Return the ten questionnaire inputs and the label from the training split.

    The processed file is used because it already applies every transform from section 2.2. All
    ten columns are then put on unit variance for this analysis, including the 0/1 indicators.
    PCA maximizes variance, so an indicator with standard deviation 0.5 would otherwise be
    down-weighted relative to the standardized continuous features purely because of its scale.
    The final model pipeline (2.2.11) still leaves the indicators unscaled.
    """
    processed = pd.read_csv(DATA_DIR / "train_processed_questionnaire.csv")
    y = processed[TARGET].to_numpy()
    X = processed.drop(columns=[TARGET])
    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    logger.info("Questionnaire design matrix: %d rows, %d columns", *X.shape)
    return X, y


def load_full_features():
    """Return the 28-feature comparison design matrix, used only for the MI cross-check."""
    processed = pd.read_csv(DATA_DIR / "train_processed_full.csv")
    X = processed.drop(columns=[TARGET])
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)


def score_predictions(y, probabilities):
    """Primary metrics for a set of out-of-fold probabilities.

    PR-AUC is threshold-free and is the primary metric for an imbalanced target. The recall
    figure is taken at a fixed precision floor so that every variant is compared at the same
    operating point; the final threshold itself is chosen in section 5/6.
    """
    precision, recall, _ = precision_recall_curve(y, probabilities)
    eligible = precision >= PRECISION_FLOOR
    return {
        "pr_auc": average_precision_score(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "recall_at_20pct_precision": float(recall[eligible].max()) if eligible.any() else 0.0,
    }


def correlation_heatmap(X):
    """Spearman correlation of the questionnaire features.

    Spearman rather than Pearson because several columns (weight, waist, BMI) are skewed and
    rounded; the ranking question is whether two features carry the same ordering of patients.
    """
    correlations = X.corr(method="spearman")
    labels = [display_name(column) for column in correlations.columns]

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    image = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = correlations.iloc[row, column]
            color = "white" if abs(value) > 0.55 else TEXT_PRIMARY
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.tick_params(colors=TEXT_SECONDARY)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Spearman correlation, questionnaire features", loc="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_correlation.png", dpi=200)
    plt.close(fig)
    return correlations


def pca_analysis(X, y):
    """Fit PCA on the standardized questionnaire features and return scores and variance table."""
    pca = PCA()
    scores = pca.fit_transform(X)
    variance = pd.DataFrame({
        "component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    })
    variance.to_csv(DATA_DIR / "pca_variance.csv", index=False)

    loadings = pd.DataFrame(pca.components_[:3].T, index=X.columns, columns=["PC1", "PC2", "PC3"])
    print("\nPCA loadings (first three components):")
    print(loadings.round(2).to_string())
    for threshold in (0.90, 0.95):
        needed = int(np.argmax(variance["cumulative_variance"].to_numpy() >= threshold) + 1)
        print(f"Components needed for {threshold:.0%} of the variance: {needed} of {X.shape[1]}")

    _plot_scree(variance)
    _plot_pca_scatter(scores, y)
    return pca, scores


def _plot_scree(variance):
    components = variance["component"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(components, variance["explained_variance_ratio"], width=0.6, color=SERIES_1,
           label="Individual")
    ax.set_xlabel("Principal component", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("Explained variance ratio", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)

    cumulative_axis = ax.twinx()
    cumulative_axis.plot(components, variance["cumulative_variance"], marker="o", linewidth=1.5,
                         color=SERIES_2, label="Cumulative")
    cumulative_axis.set_ylim(0, 1.03)
    cumulative_axis.set_ylabel("Cumulative variance", color=TEXT_SECONDARY, fontsize=9)
    cumulative_axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    for threshold in (0.90, 0.95):
        cumulative_axis.axhline(threshold, color=TEXT_SECONDARY, linewidth=0.8, linestyle="--")

    ax.set_title("PCA on the ten questionnaire features", loc="left", color=TEXT_PRIMARY,
                 fontsize=11)
    handles = ax.get_legend_handles_labels()[0] + cumulative_axis.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + cumulative_axis.get_legend_handles_labels()[1]
    ax.legend(handles, labels, frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_scree.png", dpi=200)
    plt.close(fig)


def _plot_pca_scatter(scores, y):
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.choice(len(y), size=min(SCATTER_SAMPLE, len(y)), replace=False)
    negative = y[indices] == 0
    positive = ~negative

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(scores[indices[negative], 0], scores[indices[negative], 1], s=4, alpha=0.30,
               linewidths=0, color=SERIES_1, label="No diabetes")
    ax.scatter(scores[indices[positive], 0], scores[indices[positive], 1], s=5, alpha=0.55,
               linewidths=0, color=SERIES_2, label="Diabetes")
    ax.set_xlabel("PC1", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("PC2", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=9, markerscale=3)
    ax.set_title("The classes overlap in the first two components", loc="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_scatter.png", dpi=200)
    plt.close(fig)


def mutual_information_scores(X, y):
    """Univariate, non-linear dependence between each feature and the label. Blind to redundancy."""
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=SUBSAMPLE, stratify=y, random_state=RANDOM_STATE
    )
    scores = mutual_info_classif(X_sample, y_sample, random_state=RANDOM_STATE)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)


def permutation_scores(X, y):
    """Drop in PR-AUC when one feature is shuffled, measured on a held-out fifth of the train set."""
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    model = LogisticRegression(max_iter=1000).fit(X_fit, y_fit)
    result = permutation_importance(
        model, X_validation, y_validation, scoring="average_precision",
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    mean = pd.Series(result.importances_mean, index=X.columns)
    spread = pd.Series(result.importances_std, index=X.columns)
    return mean, spread


def l1_path(X, y):
    """L1 logistic regression: how many coefficients survive at each regularization strength?

    The result is non-monotone in C: among correlated body-size features, L1 keeps whichever
    enters first and can drop a different one again as C grows. The returned count of selected
    strengths therefore summarizes stability better than the first non-zero C.
    """
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=L1_SUBSAMPLE, stratify=y, random_state=RANDOM_STATE
    )
    rows = []
    coefficients_by_strength = {}
    for C in C_GRID:
        model = LogisticRegression(solver="liblinear", l1_ratio=1.0, C=C, max_iter=2000)
        model.fit(X_sample, y_sample)
        coefficients = pd.Series(model.coef_[0], index=X.columns)
        coefficients_by_strength[C] = coefficients
        rows.append({
            "C": C,
            "n_nonzero": int((coefficients.abs() > 1e-8).sum()),
            **{column: round(coefficients[column], 4) for column in X.columns},
        })

    stability = pd.Series({
        column: int(sum(abs(coefficients_by_strength[C][column]) > 1e-8 for C in C_GRID))
        for column in X.columns
    })
    return pd.DataFrame(rows), stability, coefficients_by_strength[REFERENCE_C]


def rfecv_selection(X, y):
    """Recursive feature elimination with 5-fold CV, scored on PR-AUC."""
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    selector = RFECV(
        LogisticRegression(max_iter=1000),
        step=1,
        cv=folds,
        scoring="average_precision",
        min_features_to_select=1,
        n_jobs=-1,
    )
    selector.fit(X, y)

    scores = selector.cv_results_["mean_test_score"]
    counts = np.arange(1, X.shape[1] + 1)
    curve = pd.DataFrame({"n_features": counts, "mean_pr_auc": scores})
    curve.to_csv(DATA_DIR / "rfecv_curve.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(counts, scores, marker="o", linewidth=1.5, color=SERIES_1)
    ax.scatter([selector.n_features_], [scores[selector.n_features_ - 1]], s=70, zorder=3,
               color=SERIES_2, label=f"Optimum: {selector.n_features_} features")
    ax.set_xlabel("Number of features kept", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("CV PR-AUC", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("RFECV: PR-AUC against the number of features", loc="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rfecv_curve.png", dpi=200)
    plt.close(fig)

    support = pd.Series(selector.support_, index=X.columns)
    ranking = pd.Series(selector.ranking_, index=X.columns).sort_values()
    return support, ranking


def pca_truncation(X, y):
    """CV PR-AUC for a logistic regression on k rotated components, k = 1..10."""
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for k in range(1, X.shape[1] + 1):
        pipeline = Pipeline([
            ("pca", PCA(n_components=k)),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        started = time.perf_counter()
        probabilities = cross_val_predict(
            pipeline, X, y, cv=folds, method="predict_proba", n_jobs=N_SPLITS
        )[:, 1]
        rows.append({"n_components": k, **score_predictions(y, probabilities),
                     "seconds": time.perf_counter() - started})
    return pd.DataFrame(rows)


def _plot_pca_truncation(curve, baseline, pca_k):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(curve["n_components"], curve["pr_auc"], marker="o", linewidth=1.5, color=SERIES_2,
            label="Logistic regression on PCs")
    ax.axhline(baseline, color=SERIES_1, linewidth=1.2, linestyle="--",
               label="All ten original features")
    ax.axvline(pca_k, color=TEXT_SECONDARY, linewidth=0.9, linestyle=":",
               label=f"{pca_k} components (95 % variance)")
    ax.set_xlabel("Number of components kept", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("CV PR-AUC", color=TEXT_SECONDARY, fontsize=9)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("PR-AUC against the number of retained components", loc="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pca_truncation.png", dpi=200)
    plt.close(fig)


def plot_dimension_comparison():
    """One two-panel figure: CV PR-AUC against features kept and against components retained.

    Reads the saved result tables, so the figure can be regenerated without refitting the models.
    """
    rfecv_curve = pd.read_csv(DATA_DIR / "rfecv_curve.csv")
    pca_curve = pd.read_csv(DATA_DIR / "pca_truncation.csv")
    variance = pd.read_csv(DATA_DIR / "pca_variance.csv")
    ablation_table = pd.read_csv(DATA_DIR / "selection_ablation.csv")
    pca_k = int(variance.loc[variance["cumulative_variance"] >= 0.95, "component"].iloc[0])
    baseline = ablation_table.loc[
        ablation_table["variant"].str.startswith("All "), "pr_auc"
    ].iloc[0]
    best_features = int(rfecv_curve.loc[rfecv_curve["mean_pr_auc"].idxmax(), "n_features"])

    fig, (left, right) = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)

    left.plot(rfecv_curve["n_features"], rfecv_curve["mean_pr_auc"], marker="o", linewidth=1.4,
              color=SERIES_1)
    left.scatter([best_features], [rfecv_curve["mean_pr_auc"].max()], s=70, zorder=3,
                 color=SERIES_2, label=f"Optimum: {best_features} features")
    left.set_xlabel("Features kept", color=TEXT_SECONDARY, fontsize=9)
    left.set_ylabel("CV PR-AUC", color=TEXT_SECONDARY, fontsize=9)
    left.set_title("Feature selection (RFECV)", loc="left", color=TEXT_PRIMARY, fontsize=10)
    left.legend(frameon=False, fontsize=8, loc="lower right")

    right.plot(pca_curve["n_components"], pca_curve["pr_auc"], marker="o", linewidth=1.4,
               color=SERIES_1)
    right.axhline(baseline, color=TEXT_SECONDARY, linewidth=1, linestyle="--",
                  label="All ten original features")
    right.axvline(pca_k, color=SERIES_2, linewidth=1, linestyle=":",
                  label=f"{pca_k} components (95 % variance)")
    right.set_xlabel("Components kept", color=TEXT_SECONDARY, fontsize=9)
    right.set_title("Dimensionality reduction (PCA)", loc="left", color=TEXT_PRIMARY, fontsize=10)
    right.legend(frameon=False, fontsize=8, loc="lower right")

    for axis in (left, right):
        style_axes(axis)
    left.set_ylim(0.105, 0.172)
    fig.suptitle("Both methods plateau at about five dimensions", x=0.01, ha="left",
                 color=TEXT_PRIMARY, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES_DIR / "feature_dimension_comparison.png", dpi=200)
    plt.close(fig)


def _evaluate_subset(X, y, columns, folds):
    started = time.perf_counter()
    probabilities = cross_val_predict(
        LogisticRegression(max_iter=1000), X[columns], y, cv=folds,
        method="predict_proba", n_jobs=N_SPLITS,
    )[:, 1]
    return {**score_predictions(y, probabilities), "seconds": time.perf_counter() - started}


def ablation(X, y, selected, mi_ranking, pca_curve, pca_k):
    """Compare the full set, the selected subsets and the top-k components on identical folds.

    The number of components is fixed in advance by the 95 % variance rule rather than by the
    PR-AUC maximum, so the comparison does not choose k on the same scores it reports.
    """
    folds = StratifiedKFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    best_k = int(selected.sum())
    top_mi = list(mi_ranking.index[:best_k])

    variants = {
        f"All {X.shape[1]} features": list(X.columns),
        f"RFECV: {best_k} features": [column for column in X.columns if selected[column]],
        f"Top {best_k} by mutual information": top_mi,
        f"PCA: {pca_k} components": list(X.columns),  # transformed separately below
    }

    rows = []
    for name, columns in variants.items():
        if name.startswith("PCA"):
            scores = pca_curve.loc[pca_curve["n_components"] == pca_k].iloc[0]
            rows.append({
                "variant": name, "n_inputs": pca_k,
                "pr_auc": scores["pr_auc"], "roc_auc": scores["roc_auc"],
                "recall_at_20pct_precision": scores["recall_at_20pct_precision"],
                "seconds": scores["seconds"],
            })
            continue
        metrics = _evaluate_subset(X, y, columns, folds)
        rows.append({"variant": name, "n_inputs": len(columns), **metrics})

    result = pd.DataFrame(rows).drop_duplicates(subset=["variant"])
    result.to_csv(DATA_DIR / "selection_ablation.csv", index=False)
    return result, variants


def knn_comparison(X, y):
    """KNN is the only model family where decorrelating the inputs could plausibly help."""
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=KNN_SUBSAMPLE, stratify=y, random_state=RANDOM_STATE
    )
    folds = StratifiedKFold(KNN_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    neighbors = KNeighborsClassifier(n_neighbors=15, n_jobs=-1)

    variants = {
        "KNN, ten raw features": Pipeline([("model", neighbors)]),
        "KNN, five components": Pipeline([("pca", PCA(n_components=5)), ("model", neighbors)]),
    }
    rows = []
    for name, pipeline in variants.items():
        started = time.perf_counter()
        probabilities = cross_val_predict(
            pipeline, X_sample, y_sample, cv=folds, method="predict_proba", n_jobs=KNN_SPLITS
        )[:, 1]
        rows.append({"variant": name, **score_predictions(y_sample, probabilities),
                     "seconds": time.perf_counter() - started})
    result = pd.DataFrame(rows)
    result.to_csv(DATA_DIR / "knn_pca.csv", index=False)
    return result


def _plot_ranking(mi_ranking, importance, spread):
    order = mi_ranking.index[::-1]
    labels = [display_name(column) for column in order]
    positions = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    axes[0].barh(positions, mi_ranking.loc[order], height=0.6, color=SERIES_1)
    axes[0].set_title("Mutual information", loc="left", color=TEXT_PRIMARY, fontsize=10)
    axes[0].set_xlabel("nats", color=TEXT_SECONDARY, fontsize=9)

    axes[1].barh(positions, importance.loc[order], height=0.6, color=SERIES_2,
                 xerr=spread.loc[order], error_kw={"elinewidth": 0.8, "ecolor": TEXT_SECONDARY})
    axes[1].set_title("Permutation importance (drop in PR-AUC)", loc="left",
                      color=TEXT_PRIMARY, fontsize=10)
    axes[1].set_xlabel("PR-AUC drop", color=TEXT_SECONDARY, fontsize=9)

    for axis in axes:
        axis.set_yticks(positions, labels, fontsize=8)
        style_axes(axis)
        axis.yaxis.grid(False)
        axis.xaxis.grid(True, color=GRID, linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_ranking.png", dpi=200)
    plt.close(fig)


def full_model_information():
    """MI ranking for the 28-feature comparison model: do the lab values dominate?"""
    X = load_full_features()
    y = pd.read_csv(DATA_DIR / "train_processed_full.csv")[TARGET].to_numpy()
    ranking = mutual_information_scores(X, y)
    ranking.rename("mutual_information").to_csv(DATA_DIR / "full_feature_mi.csv")
    print("\nMutual information, comparison model (top 10):")
    print(ranking.head(10).round(4).to_string())
    print("Bottom 5:")
    print(ranking.tail(5).round(4).to_string())
    return ranking


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    X, y = load_questionnaire()

    correlations = correlation_heatmap(X)
    strong = [
        (display_name(row), display_name(column), round(correlations.loc[row, column], 2))
        for row in correlations.index for column in correlations.columns
        if row < column and abs(correlations.loc[row, column]) >= 0.5
    ]
    print("Correlated pairs (|Spearman| >= 0.5):")
    for pair in strong:
        print(f"  {pair[0]:22s} {pair[1]:22s} {pair[2]:+.2f}")

    _, _ = pca_analysis(X, y)
    variance = pd.read_csv(DATA_DIR / "pca_variance.csv")
    pca_k = int(variance.loc[variance["cumulative_variance"] >= 0.95, "component"].iloc[0])

    mi_ranking = mutual_information_scores(X, y)
    importance, spread = permutation_scores(X, y)
    l1_table, l1_stability, l1_reference = l1_path(X, y)
    l1_table.to_csv(DATA_DIR / "l1_path.csv", index=False)

    scores = pd.DataFrame({
        "mutual_information": mi_ranking,
        "permutation_importance": importance,
        "permutation_std": spread,
        "l1_selected_count_of_7": l1_stability,
        f"l1_coefficient_C{REFERENCE_C}": l1_reference,
    })
    scores["mi_rank"] = scores["mutual_information"].rank(ascending=False).astype(int)
    scores["permutation_rank"] = scores["permutation_importance"].rank(ascending=False).astype(int)
    scores.sort_values("mutual_information", ascending=False).to_csv(
        DATA_DIR / "feature_scores.csv", index_label="feature"
    )
    _plot_ranking(mi_ranking, importance, spread)

    print("\nMutual information ranking:")
    print(mi_ranking.round(4).to_string())
    print("\nL1 path (negative coefficient means lower risk):")
    print(l1_table[["C", "n_nonzero"]].to_string(index=False))
    print(f"L1 stability at C={REFERENCE_C}:")
    print(l1_stability.sort_values(ascending=False).to_string())

    support, ranking = rfecv_selection(X, y)
    print("\nRFECV selection:")
    print(ranking.round(0).to_string())

    pca_curve = pca_truncation(X, y)
    pca_curve.to_csv(DATA_DIR / "pca_truncation.csv", index=False)

    ablation_table, _ = ablation(X, y, support, mi_ranking, pca_curve, pca_k)
    baseline = ablation_table.loc[
        ablation_table["variant"] == f"All {X.shape[1]} features", "pr_auc"
    ].iloc[0]
    _plot_pca_truncation(pca_curve, baseline, pca_k)
    plot_dimension_comparison()
    print("\nAblation (identical folds):")
    print(ablation_table.round(4).to_string(index=False))

    knn_table = knn_comparison(X, y)
    print("\nKNN comparison:")
    print(knn_table.round(4).to_string(index=False))

    full_model_information()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
