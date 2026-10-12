"""PCA steps 2-6 on a 2D image matrix (rows = images, columns = pixels).

Usage:
    python pca_steps.py --data cars.npy --k 20
    python pca_steps.py --data cars.npy --k 20 --outdir pca_results

Input:  .npy file of shape (N, D), values in [0, 1] (output of preprocess_images.py)
Output (in --outdir):
    mean.npy              mean image, shape (D,)
    eigenvalues.npy       all eigenvalues, sorted descending, shape (D,)
    components.npy        top k eigenvectors as columns, shape (D, k)
    projected.npy         images projected onto the k components, shape (N, k)
    summary.txt           shapes and explained variance
    principal_components.png, explained_variance.png, reconstructions.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------- PCA steps
def center_data(X):
    """Subtract the mean image from every row (required before covariance)."""
    mean = X.mean(axis=0)
    return X - mean, mean


def compute_covariance(X_centered):
    """Step 2: covariance matrix (D x D) of the centered data."""
    n = X_centered.shape[0]
    return (X_centered.T @ X_centered) / (n - 1)


def compute_eigen(cov):
    """Step 3: eigenvalues and eigenvectors (eigh: cov is symmetric)."""
    return np.linalg.eigh(cov)


def sort_eigen(eigenvalues, eigenvectors):
    """Step 4: sort by eigenvalue, descending. Eigenvectors are the columns."""
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def select_top_k(eigenvectors, k):
    """Step 5: keep the first k eigenvectors -> principal components (D x k)."""
    return eigenvectors[:, :k]


def project(X_centered, components):
    """Step 6: project centered images onto the k components -> (N x k)."""
    return X_centered @ components


def run_pca(X, k):
    """Run steps 2-6 in order. Returns a dict with all results."""
    X_centered, mean = center_data(X)
    cov = compute_covariance(X_centered)
    eigenvalues, eigenvectors = compute_eigen(cov)
    eigenvalues, eigenvectors = sort_eigen(eigenvalues, eigenvectors)
    components = select_top_k(eigenvectors, k)
    projected = project(X_centered, components)
    return {"mean": mean, "eigenvalues": eigenvalues,
            "components": components, "projected": projected}


# ---------------------------------------------------------------- helpers
def explained_variance_ratio(eigenvalues):
    """Fraction of total variance carried by each component."""
    vals = np.clip(eigenvalues, 0, None)
    return vals / vals.sum()


def reconstruct(projected, components, mean):
    """Go back from k dimensions to pixel space (for checking quality)."""
    return np.clip(projected @ components.T + mean, 0, 1)


def image_side(D):
    """Side length of the square image, from the number of pixels."""
    side = int(round(np.sqrt(D)))
    if side * side != D:
        raise ValueError(f"{D} pixels is not a square image")
    return side


# ---------------------------------------------------------------- saving
def save_arrays(outdir, result):
    """Save the numeric results as .npy files."""
    for name in ("mean", "eigenvalues", "components", "projected"):
        np.save(os.path.join(outdir, f"{name}.npy"), result[name])


def save_summary(outdir, X, result, k):
    """Save shapes and explained variance as text."""
    ratio = explained_variance_ratio(result["eigenvalues"])
    with open(os.path.join(outdir, "summary.txt"), "w") as f:
        f.write(f"Input matrix:         {X.shape}  (images x pixels)\n")
        f.write(f"Components (D x k):   {result['components'].shape}\n")
        f.write(f"Projected data (N x k): {result['projected'].shape}\n")
        f.write(f"Variance explained by top {k}: {ratio[:k].sum():.4f}\n\n")
        f.write("Component, eigenvalue, variance ratio, cumulative\n")
        cum = np.cumsum(ratio)
        for i in range(k):
            f.write(f"{i + 1}, {result['eigenvalues'][i]:.6f}, "
                    f"{ratio[i]:.4f}, {cum[i]:.4f}\n")


def save_components_image(outdir, components, side, n=10):
    """Show the first n principal components as images."""
    n = min(n, components.shape[1])
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        axes[i].imshow(components[:, i].reshape(side, side), cmap="gray")
        axes[i].set_title(f"PC{i + 1}", fontsize=9)
    fig.savefig(os.path.join(outdir, "principal_components.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_variance_plot(outdir, eigenvalues, k, thresholds=(0.90, 0.95)):
    """Save per-component and cumulative explained variance.

    Only non-zero components are shown (at most N - 1 for N images).
    The chosen k and the variance thresholds are marked on the plots.
    """
    ratio = explained_variance_ratio(eigenvalues)
    n = min(int(np.sum(ratio > 1e-10)), 100)
    cum = np.cumsum(ratio)
    ks = np.arange(1, n + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(ks, ratio[:n] * 100, "o-", ms=3)
    ax[0].axhline(1, color="gray", ls=":", lw=1, label="1 % per component")
    ax[0].axvline(k, color="C3", ls="--", lw=1, label=f"chosen k = {k}")
    ax[0].set(title="Variance per component", xlabel="Component",
              ylabel="Explained variance (%)")
    ax[0].legend()

    ax[1].plot(ks, cum[:n] * 100, "o-", ms=3)
    for t, color in zip(thresholds, ("C2", "C1")):
        k_t = int(np.argmax(cum >= t)) + 1
        ax[1].axhline(t * 100, color=color, ls=":", lw=1,
                      label=f"{t:.0%} at k = {k_t}")
    ax[1].axvline(k, color="C3", ls="--", lw=1,
                  label=f"chosen k = {k} ({cum[k - 1]:.1%})")
    ax[1].set(title="Cumulative explained variance", xlabel="k",
              ylabel="Cumulative variance (%)")
    ax[1].grid(True, alpha=0.4)
    ax[1].legend(loc="lower right")
    fig.savefig(os.path.join(outdir, "explained_variance.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_reconstructions(outdir, X, result, side, ks, idx=0):
    """Reconstruct one image using different numbers of components."""
    Xc = X[idx] - result["mean"]
    panels, titles = [X[idx]], ["Original"]
    for k in ks:
        comps = result["components"][:, :k]
        panels.append(reconstruct(Xc @ comps, comps, result["mean"]))
        titles.append(f"k={k}")
    fig, axes = plt.subplots(1, len(panels), figsize=(2.2 * len(panels), 2.6))
    for ax, im, t in zip(axes, panels, titles):
        ax.imshow(im.reshape(side, side), cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    fig.savefig(os.path.join(outdir, "reconstructions.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_all(outdir, X, result, k):
    """Save every output file."""
    os.makedirs(outdir, exist_ok=True)
    side = image_side(X.shape[1])
    ks = sorted({1, max(k // 4, 1), max(k // 2, 1), k})
    save_arrays(outdir, result)
    save_summary(outdir, X, result, k)
    save_components_image(outdir, result["components"], side)
    save_variance_plot(outdir, result["eigenvalues"], k)
    save_reconstructions(outdir, X, result, side, ks)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help=".npy file, shape (N, D)")
    parser.add_argument("--k", type=int, default=20, help="number of components")
    parser.add_argument("--outdir", default="pca_results", help="output folder")
    args = parser.parse_args()

    X = np.load(args.data)
    if X.ndim != 2:
        raise ValueError(f"Expected a 2D matrix (N, D), got shape {X.shape}")
    k = min(args.k, X.shape[0] - 1, X.shape[1])

    result = run_pca(X, k)
    save_all(args.outdir, X, result, k)

    ratio = explained_variance_ratio(result["eigenvalues"])
    print(f"Input: {X.shape}, k = {k}")
    print(f"Projected shape: {result['projected'].shape}")
    print(f"Variance explained by top {k}: {ratio[:k].sum():.4f}")
    print(f"Saved to '{args.outdir}/'")


if __name__ == "__main__":
    main()