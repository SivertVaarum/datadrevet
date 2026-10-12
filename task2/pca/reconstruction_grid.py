"""Show original images next to their PCA reconstructions for several k.

Rows are the best, median and worst reconstructed image (by MSE at the
largest k), so the figure shows the full range of reconstruction quality.
Column k=0 is the mean image, i.e. the reconstruction with no components.

Usage:
    python reconstruction_grid.py
    python reconstruction_grid.py --ks 1 5 10 20 --out reconstruction_grid.png

Needs (from the earlier scripts):
    cars.npy, cars_files.txt, pca_results/mean.npy, components.npy, projected.npy
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_inputs(data_path, names_path, pca_dir):
    """Load the original matrix, file names and PCA results."""
    X = np.load(data_path)
    with open(names_path) as f:
        names = [os.path.splitext(line.strip())[0] for line in f if line.strip()]
    mean = np.load(os.path.join(pca_dir, "mean.npy"))
    components = np.load(os.path.join(pca_dir, "components.npy"))
    projected = np.load(os.path.join(pca_dir, "projected.npy"))
    return X, names, mean, components, projected


def reconstruct(mean, components, projected, k):
    """Rebuild all images from the first k components. Shape (N, D)."""
    return np.clip(projected[:, :k] @ components[:, :k].T + mean, 0, 1)


def per_image_mse(X, X_hat):
    """MSE for each image separately. Shape (N,)."""
    return np.mean((X - X_hat) ** 2, axis=1)


def pick_rows(errors):
    """Indices of the best, median and worst reconstructed image."""
    order = np.argsort(errors)
    return [order[0], order[len(order) // 2], order[-1]]


def save_grid(out_path, X, names, recons, ks, rows):
    """Draw rows = images, columns = original + one column per k."""
    side = int(round(np.sqrt(X.shape[1])))
    labels = ["Best", "Median", "Worst"]
    ncols = len(ks) + 1
    fig, axes = plt.subplots(len(rows), ncols,
                             figsize=(1.9 * ncols, 2.15 * len(rows)))
    for r, idx in enumerate(rows):
        axes[r, 0].imshow(X[idx].reshape(side, side), cmap="gray", vmin=0, vmax=1)
        axes[r, 0].set_title("Original" if r == 0 else "", fontsize=10)
        axes[r, 0].set_ylabel(f"{labels[r]}\n{names[idx]}", fontsize=9)
        for c, k in enumerate(ks, start=1):
            img = recons[k][idx]
            mse = np.mean((X[idx] - img) ** 2)
            axes[r, c].imshow(img.reshape(side, side), cmap="gray", vmin=0, vmax=1)
            if r == 0:
                axes[r, c].set_title("Mean (k=0)" if k == 0 else f"k={k}",
                                     fontsize=10)
            axes[r, c].set_xlabel(f"MSE {mse:.4f}", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="cars.npy", help="original matrix (N, D)")
    parser.add_argument("--names", default="cars_files.txt", help="file name list")
    parser.add_argument("--pca_dir", default="pca_results", help="folder with PCA outputs")
    parser.add_argument("--ks", type=int, nargs="+", default=[0, 1, 5, 10, 20],
                        help="numbers of components to show")
    parser.add_argument("--out", default="reconstruction_grid.png", help="output image")
    args = parser.parse_args()

    X, names, mean, components, projected = load_inputs(
        args.data, args.names, args.pca_dir)
    ks = sorted({min(k, components.shape[1]) for k in args.ks})
    recons = {k: reconstruct(mean, components, projected, k) for k in ks}

    rows = pick_rows(per_image_mse(X, recons[ks[-1]]))
    save_grid(args.out, X, names, recons, ks, rows)

    print(f"Rows: {[names[i] for i in rows]}")
    for k in ks:
        print(f"k={k:>2}: mean MSE over all images = "
              f"{per_image_mse(X, recons[k]).mean():.4f}")
    print(f"Saved '{args.out}'")


if __name__ == "__main__":
    main()
