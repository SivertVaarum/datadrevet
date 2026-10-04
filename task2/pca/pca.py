"""PCA on a set of grayscale images.

Usage:
    python pca_images.py --folder my_images --size 64 --k 20
If --folder is omitted, scikit-learn's built-in digits dataset is used as a demo.
All outputs are saved to the "results" folder.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")  # no GUI needed, we only save files
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize

OUT_DIR = "results"


# ---------------------------------------------------------------- loading
def load_gray01(path, size):
    """Load one image as grayscale float in [0, 1], resized to (size, size)."""
    img = io.imread(path)
    if img.ndim == 3:
        img = rgb2gray(img[..., :3])
    else:
        img = img_as_float(img)
    img = resize(img, (size, size), anti_aliasing=True)
    return np.clip(img, 0, 1)


def load_folder(folder, size):
    """Load all images in a folder. Returns (N, size, size) array."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    if not files:
        raise FileNotFoundError(f"No images found in {folder}")
    return np.array([load_gray01(os.path.join(folder, f), size) for f in files])


def load_demo_digits():
    """Offline demo dataset: 8x8 digits, normalized to [0, 1]."""
    from sklearn.datasets import load_digits
    data = load_digits().images
    return data / data.max()


# ---------------------------------------------------------------- PCA steps
def images_to_matrix(images):
    """Step 1: (N, H, W) -> (N, H*W). Each row is one image."""
    n = images.shape[0]
    return images.reshape(n, -1)


def center_data(X):
    """Subtract the mean image. Returns centered data and the mean."""
    mean = X.mean(axis=0)
    return X - mean, mean


def compute_covariance(X_centered):
    """Step 2: covariance matrix (D x D) of the centered data."""
    n = X_centered.shape[0]
    return (X_centered.T @ X_centered) / (n - 1)


def compute_eigen(cov):
    """Step 3: eigenvalues and eigenvectors (eigh, since cov is symmetric)."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    return eigenvalues, eigenvectors


def sort_eigen(eigenvalues, eigenvectors):
    """Step 4: sort by eigenvalue, descending. Columns are eigenvectors."""
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def select_components(eigenvectors, k):
    """Step 5: keep the top k eigenvectors. Shape (D, k)."""
    return eigenvectors[:, :k]


def project(X_centered, components):
    """Step 6: project onto the k-dim subspace. Shape (N, k)."""
    return X_centered @ components


def reconstruct(Z, components, mean):
    """Map projected data back to pixel space. Shape (N, D)."""
    return Z @ components.T + mean


def pca(images, k):
    """Run the full PCA pipeline. Returns a dict with all results."""
    X = images_to_matrix(images)
    Xc, mean = center_data(X)
    cov = compute_covariance(Xc)
    vals, vecs = compute_eigen(cov)
    vals, vecs = sort_eigen(vals, vecs)
    comps = select_components(vecs, k)
    Z = project(Xc, comps)
    return {"mean": mean, "eigenvalues": vals, "components": comps,
            "projected": Z, "shape": images.shape[1:]}


# ---------------------------------------------------------------- evaluation
def explained_variance_ratio(eigenvalues):
    """Fraction of total variance carried by each component."""
    vals = np.clip(eigenvalues, 0, None)
    return vals / vals.sum()


def mse_for_k(images, result, k):
    """Reconstruction MSE when using only the first k components."""
    X = images_to_matrix(images)
    comps = result["components"][:, :k]
    Z = (X - result["mean"]) @ comps
    X_hat = np.clip(reconstruct(Z, comps, result["mean"]), 0, 1)
    return np.mean((X - X_hat) ** 2)


# ---------------------------------------------------------------- saving
def save_array_as_png(path, img):
    """Save a float image as PNG, min-max scaled for display."""
    rng = img.max() - img.min()
    img = (img - img.min()) / rng if rng > 0 else np.zeros_like(img)
    io.imsave(path, (img * 255).astype(np.uint8), check_contrast=False)


def save_grid(path, imgs, shape, title, ncols=5, normalize=False):
    """Save a grid of images (each flattened or 2D) as one figure."""
    n = len(imgs)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, im in zip(np.atleast_1d(axes).ravel(), imgs):
        im = np.asarray(im).reshape(shape)
        if normalize:
            ax.imshow(im, cmap="gray")
        else:
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_variance_plot(path, eigenvalues):
    """Save scree plot and cumulative explained variance."""
    ratio = explained_variance_ratio(eigenvalues)
    cum = np.cumsum(ratio)
    n = min(len(ratio), 100)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(range(1, n + 1), ratio[:n], "o-", ms=3)
    ax[0].set(title="Variance per component", xlabel="Component", ylabel="Ratio")
    ax[1].plot(range(1, n + 1), cum[:n], "o-", ms=3)
    ax[1].set(title="Cumulative explained variance", xlabel="k", ylabel="Ratio")
    ax[1].grid(True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_projection_scatter(path, Z):
    """Save a 2D scatter of the first two principal components."""
    if Z.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(Z[:, 0], Z[:, 1], s=8)
    ax.set(title="Projection onto PC1 and PC2", xlabel="PC1", ylabel="PC2")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_reconstructions(path, images, result, ks, idx=0):
    """Save one image reconstructed with different numbers of components."""
    X = images_to_matrix(images)
    shape = result["shape"]
    recons = [X[idx]]
    for k in ks:
        comps = result["components"][:, :k]
        z = (X[idx] - result["mean"]) @ comps
        recons.append(np.clip(reconstruct(z, comps, result["mean"]), 0, 1))
    save_grid(path, recons, shape,
              "Original, then k = " + ", ".join(map(str, ks)),
              ncols=len(recons))


def save_results(images, result, ks):
    """Save every output file to OUT_DIR."""
    os.makedirs(OUT_DIR, exist_ok=True)
    shape = result["shape"]
    n_show = min(10, len(images))

    save_grid(f"{OUT_DIR}/originals.png", images[:n_show], shape,
              "Original images (normalized to [0, 1])")
    save_grid(f"{OUT_DIR}/mean_image.png", [result["mean"]], shape,
              "Mean image", ncols=1)
    save_grid(f"{OUT_DIR}/principal_components.png",
              result["components"].T[:10], shape,
              "Top principal components (eigenfaces)", normalize=True)
    save_variance_plot(f"{OUT_DIR}/explained_variance.png", result["eigenvalues"])
    save_projection_scatter(f"{OUT_DIR}/projection_pc1_pc2.png", result["projected"])
    save_reconstructions(f"{OUT_DIR}/reconstructions.png", images, result, ks)

    np.save(f"{OUT_DIR}/projected.npy", result["projected"])
    np.save(f"{OUT_DIR}/components.npy", result["components"])
    np.save(f"{OUT_DIR}/mean.npy", result["mean"])

    with open(f"{OUT_DIR}/summary.txt", "w") as f:
        ratio = explained_variance_ratio(result["eigenvalues"])
        f.write(f"Images: {len(images)}, shape: {shape}\n")
        f.write(f"Projected data shape: {result['projected'].shape}\n\n")
        f.write("k, cumulative variance, reconstruction MSE\n")
        for k in ks:
            f.write(f"{k}, {ratio[:k].sum():.4f}, {mse_for_k(images, result, k):.6f}\n")


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=None, help="folder with images")
    parser.add_argument("--size", type=int, default=64, help="resize to size x size")
    parser.add_argument("--k", type=int, default=20, help="number of components")
    args = parser.parse_args()

    images = load_folder(args.folder, args.size) if args.folder else load_demo_digits()
    k = min(args.k, images.shape[1] * images.shape[2])

    result = pca(images, k)
    ks = sorted({1, 5, k // 2 or 1, k})
    save_results(images, result, ks)
    print(f"Done. {len(images)} images, projected to shape {result['projected'].shape}.")
    print(f"Files saved in '{OUT_DIR}/'")


if __name__ == "__main__":
    main()