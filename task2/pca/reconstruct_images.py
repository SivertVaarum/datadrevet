"""Reconstruct all images from PCA results and save them as PNG files.

Usage:
    python reconstruct_images.py
    python reconstruct_images.py --k 10 --outdir reconstructed_k10

Needs (from the earlier scripts):
    pca_results/mean.npy, components.npy, projected.npy
    cars_files.txt   (optional, used to name the output files)

Reconstruction: X_hat = projected[:, :k] @ components[:, :k].T + mean
"""
import argparse
import os

import numpy as np
from skimage import io


# ---------------------------------------------------------------- loading
def load_pca_results(pca_dir):
    """Load mean, components (D x K) and projected data (N x K)."""
    mean = np.load(os.path.join(pca_dir, "mean.npy"))
    components = np.load(os.path.join(pca_dir, "components.npy"))
    projected = np.load(os.path.join(pca_dir, "projected.npy"))
    return mean, components, projected


def load_names(names_path, n):
    """Read file names, or fall back to image_000, image_001, ..."""
    if names_path and os.path.isfile(names_path):
        with open(names_path) as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == n:
            return [os.path.splitext(name)[0] for name in names]
        print("Warning: name count does not match images, using numbers instead.")
    return [f"image_{i:03d}" for i in range(n)]


# ---------------------------------------------------------------- reconstruction
def reconstruct_all(mean, components, projected, k):
    """Rebuild all images using the first k components. Shape (N, D), in [0, 1]."""
    X_hat = projected[:, :k] @ components[:, :k].T + mean
    return np.clip(X_hat, 0, 1)


def image_side(D):
    """Side length of the square image from the pixel count."""
    side = int(round(np.sqrt(D)))
    if side * side != D:
        raise ValueError(f"{D} pixels is not a square image")
    return side


# ---------------------------------------------------------------- saving
def save_image(path, flat_img, side):
    """Save one flattened [0, 1] image as an 8-bit PNG."""
    img = (flat_img.reshape(side, side) * 255).round().astype(np.uint8)
    io.imsave(path, img, check_contrast=False)


def save_all(outdir, X_hat, names):
    """Save every reconstructed image to outdir."""
    os.makedirs(outdir, exist_ok=True)
    side = image_side(X_hat.shape[1])
    for flat_img, name in zip(X_hat, names):
        save_image(os.path.join(outdir, f"{name}.png"), flat_img, side)


def compute_mse(X, X_hat):
    """Mean squared error between original and reconstructed images."""
    return np.mean((X - X_hat) ** 2)


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_dir", default="pca_results", help="folder with PCA outputs")
    parser.add_argument("--k", type=int, default=None,
                        help="components to use (default: all saved)")
    parser.add_argument("--names", default="cars_files.txt", help="file name list")
    parser.add_argument("--data", default="cars.npy",
                        help="original matrix, only used to report the error")
    parser.add_argument("--outdir", default=None, help="output folder")
    args = parser.parse_args()

    mean, components, projected = load_pca_results(args.pca_dir)
    k = min(args.k or components.shape[1], components.shape[1])
    outdir = args.outdir or f"reconstructed_k{k}"

    X_hat = reconstruct_all(mean, components, projected, k)
    names = load_names(args.names, X_hat.shape[0])
    save_all(outdir, X_hat, names)

    print(f"Reconstructed {X_hat.shape[0]} images using k = {k} components")
    print(f"Saved to '{outdir}/'")
    if os.path.isfile(args.data):
        X = np.load(args.data)
        if X.shape == X_hat.shape:
            print(f"Reconstruction MSE vs original: {compute_mse(X, X_hat):.6f}")


if __name__ == "__main__":
    main()