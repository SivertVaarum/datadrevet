"""Error analysis: trade-off between PCA compression and reconstruction error.

Usage:
    python error_analysis.py --data cars.npy
    python error_analysis.py --data cars.npy --holdout 10     # also test on unseen images

Needs pca_steps.py in the same folder (it reuses run_pca).

Outputs (in --outdir, default error_analysis/):
    error_table.csv           k, storage, compression, MSE, PSNR, marginal gains
    mse_vs_k.png              MSE and cumulative variance against k
    mse_vs_storage.png        the trade-off curve (MSE against storage)
    marginal_gain.png         MSE reduction per extra component
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pca_steps import run_pca, explained_variance_ratio


# ---------------------------------------------------------------- data
def split_data(X, holdout, seed=0):
    """Split rows into train and test. holdout = number of test images."""
    if holdout <= 0:
        return X, None
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    return X[idx[holdout:]], X[idx[:holdout]]


def choose_ks(max_k):
    """k values to evaluate: dense at the start, then spaced out."""
    base = [1, 2, 3, 5, 10, 15, 20, 25, 31, 40, 45, 60, 80, 100]
    ks = sorted({k for k in base if k < max_k} | {max_k})
    return ks


# ---------------------------------------------------------------- measures
def reconstruct_with_k(X, mean, components_all, k):
    """Project X on the first k components and reconstruct, clipped to [0, 1]."""
    comps = components_all[:, :k]
    Z = (X - mean) @ comps
    return np.clip(Z @ comps.T + mean, 0, 1)


def mean_mse(X, X_hat):
    """Mean squared error, averaged over all images and pixels."""
    return float(np.mean((X - X_hat) ** 2))


def psnr_from_mse(mse):
    """PSNR in dB for images in [0, 1]."""
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def storage_fraction(n, d, k):
    """Stored numbers (mean + k components + k coefficients per image)
    divided by the raw size n*d."""
    return (d + k * d + n * k) / (n * d)


def theory_mse(eigenvalues, k, n, d):
    """MSE predicted by the discarded eigenvalues (no clipping)."""
    return float(eigenvalues[k:].clip(0).sum() * (n - 1) / n / d)


# ---------------------------------------------------------------- table
def build_table(X_train, X_test, result, ks):
    """One row per k with all quantities of interest."""
    n, d = X_train.shape
    mean, comps, eig = result["mean"], result["components"], result["eigenvalues"]

    base_mse = mean_mse(X_train, np.tile(mean, (n, 1)))   # k = 0 baseline
    prev_mse, prev_k, prev_storage = base_mse, 0, storage_fraction(n, d, 0)
    table = [{"k": 0, "storage_pct": 100 * prev_storage, "compression": 1 / prev_storage,
              "train_mse": base_mse, "train_psnr": psnr_from_mse(base_mse),
              "theory_mse": theory_mse(eig, 0, n, d),
              "cum_variance_pct": 0.0,
              "gain_per_component": np.nan, "gain_per_1pct_storage": np.nan}]
    if X_test is not None:
        mu = np.tile(mean, (len(X_test), 1))
        table[0]["test_mse"] = mean_mse(X_test, mu)

    ratio = explained_variance_ratio(eig)
    for k in ks:
        mse = mean_mse(X_train, reconstruct_with_k(X_train, mean, comps, k))
        stor = storage_fraction(n, d, k)
        gain = (prev_mse - mse) / (k - prev_k)
        gain_pct = (prev_mse - mse) / (100 * (stor - prev_storage)) if stor > prev_storage else np.nan
        row = {"k": k, "storage_pct": 100 * stor, "compression": 1 / stor,
               "train_mse": mse, "train_psnr": psnr_from_mse(mse),
               "theory_mse": theory_mse(eig, k, n, d),
               "cum_variance_pct": 100 * ratio[:k].sum(),
               "gain_per_component": gain, "gain_per_1pct_storage": gain_pct}
        if X_test is not None:
            row["test_mse"] = mean_mse(X_test, reconstruct_with_k(X_test, mean, comps, k))
        table.append(row)
        prev_mse, prev_k, prev_storage = mse, k, stor
    return table


# ---------------------------------------------------------------- saving
def save_table(path, table):
    """Save the table as CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[-1].keys()))
        writer.writeheader()
        for row in table:
            writer.writerow({key: row.get(key, "") for key in writer.fieldnames})


def print_table(table, has_test):
    """Print a compact summary."""
    head = f"{'k':>4} {'storage%':>9} {'ratio':>7} {'MSE':>9} {'PSNR':>7} {'var%':>6}"
    if has_test:
        head += f" {'test MSE':>9}"
    print(head)
    for r in table:
        line = (f"{r['k']:>4} {r['storage_pct']:>9.1f} {r['compression']:>7.2f} "
                f"{r['train_mse']:>9.5f} {r['train_psnr']:>7.2f} {r['cum_variance_pct']:>6.1f}")
        if has_test:
            line += f" {r['test_mse']:>9.5f}"
        print(line)


def plot_mse_vs_k(path, table, has_test):
    """MSE against k, with cumulative variance on a second axis."""
    ks = [r["k"] for r in table]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, [r["train_mse"] for r in table], "o-", label="MSE (training images)")
    if has_test:
        ax.plot(ks, [r["test_mse"] for r in table], "s--", label="MSE (held-out images)")
    ax.set(xlabel="Number of components k", ylabel="Mean MSE")
    ax2 = ax.twinx()
    ax2.plot(ks, [r["cum_variance_pct"] for r in table], "g:", label="Cumulative variance")
    ax2.set_ylabel("Cumulative variance explained (%)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mse_vs_storage(path, table, has_test):
    """The trade-off curve: error against storage, points labelled with k."""
    stor = [r["storage_pct"] for r in table]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(stor, [r["train_mse"] for r in table], "o-", label="Training images")
    if has_test:
        ax.plot(stor, [r["test_mse"] for r in table], "s--", label="Held-out images")
        ax.legend()
    for r in table:
        if r["k"] in (0, 1, 5, 10, 20, 31, 45, 60, 80):
            ax.annotate(f"k={r['k']}", (r["storage_pct"], r["train_mse"]),
                        textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax.axvline(100, color="red", ls=":", lw=1)
    ax.text(100, ax.get_ylim()[1] * 0.9, " no compression", color="red", fontsize=8)
    ax.set(xlabel="Storage (% of original)", ylabel="Mean MSE",
           title="Compression vs reconstruction error")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_marginal_gain(path, table):
    """MSE reduction per extra component (diminishing returns)."""
    rows = [r for r in table if r["k"] > 0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(r["k"]) for r in rows], [r["gain_per_component"] for r in rows])
    ax.set(xlabel="k (up to this value)", ylabel="MSE reduction per added component",
           title="Diminishing returns")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help=".npy matrix (N, D)")
    parser.add_argument("--holdout", type=int, default=0,
                        help="number of images kept out of the PCA fit for testing")
    parser.add_argument("--outdir", default="error_analysis")
    args = parser.parse_args()

    X = np.load(args.data)
    X_train, X_test = split_data(X, args.holdout)
    max_k = len(X_train) - 1
    ks = choose_ks(max_k)

    result = run_pca(X_train, max_k)          # fit once with all components
    table = build_table(X_train, X_test, result, ks)
    has_test = X_test is not None

    os.makedirs(args.outdir, exist_ok=True)
    save_table(os.path.join(args.outdir, "error_table.csv"), table)
    plot_mse_vs_k(os.path.join(args.outdir, "mse_vs_k.png"), table, has_test)
    plot_mse_vs_storage(os.path.join(args.outdir, "mse_vs_storage.png"), table, has_test)
    plot_marginal_gain(os.path.join(args.outdir, "marginal_gain.png"), table)

    print(f"Images: {len(X_train)} for fitting"
          + (f", {len(X_test)} held out" if has_test else "") + f", pixels: {X.shape[1]}\n")
    print_table(table, has_test)
    print(f"\nSaved to '{args.outdir}/'")


if __name__ == "__main__":
    main()