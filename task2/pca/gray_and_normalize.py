"""Convert all images in a folder to 64x64 grayscale, normalize to [0, 1],
flatten each image to one row, and save everything as a .npy file.

Usage:
    python preprocess_images.py --folder car_images
    python preprocess_images.py --folder car_images --out cars.npy --size 64

Outputs:
    <out>.npy            array of shape (N, size*size), values in [0, 1]
    <out>_files.txt      file names, one per row of the array (same order)
"""
import argparse
import os

import numpy as np
from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.transform import resize

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_image_files(folder):
    """Return sorted list of image file names in the folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(IMAGE_EXTENSIONS))
    if not files:
        raise FileNotFoundError(f"No images found in: {folder}")
    return files


def to_grayscale01(img):
    """Convert any loaded image (gray, RGB, RGBA) to a float grayscale in [0, 1]."""
    if img.ndim == 3:
        img = rgb2gray(img[..., :3])      # drops alpha, returns floats in [0, 1]
    else:
        img = img_as_float(img)           # scales by dtype (uint8 -> /255, etc.)
    return np.clip(img, 0, 1)


def resize_image(img, size):
    """Resize (stretch) to size x size."""
    return np.clip(resize(img, (size, size), anti_aliasing=True), 0, 1)


def process_image(path, size):
    """Load one image, grayscale, resize, normalize, flatten to 1D."""
    img = io.imread(path)
    img = to_grayscale01(img)
    img = resize_image(img, size)
    return img.flatten()


def process_folder(folder, size):
    """Process every image in the folder. Returns (matrix, file names).

    Files that cannot be read are skipped with a warning.
    """
    rows, names = [], []
    for name in list_image_files(folder):
        try:
            rows.append(process_image(os.path.join(folder, name), size))
            names.append(name)
        except Exception as e:
            print(f"Skipping {name}: {e}")
    if not rows:
        raise RuntimeError("No images could be processed.")
    return np.array(rows), names


def save_outputs(matrix, names, out_path):
    """Save the matrix as .npy and the file names as a text file."""
    np.save(out_path, matrix)
    names_path = os.path.splitext(out_path)[0] + "_files.txt"
    with open(names_path, "w") as f:
        f.write("\n".join(names))
    return names_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="folder with images")
    parser.add_argument("--out", default="images_flat.npy", help="output .npy path")
    parser.add_argument("--size", type=int, default=64, help="output size (size x size)")
    args = parser.parse_args()

    matrix, names = process_folder(args.folder, args.size)
    names_path = save_outputs(matrix, names, args.out)

    print(f"Processed {len(names)} images")
    print(f"Matrix shape: {matrix.shape}  (images x pixels)")
    print(f"Value range: {matrix.min():.3f} to {matrix.max():.3f}")
    print(f"Saved: {args.out}, {names_path}")


if __name__ == "__main__":
    main()