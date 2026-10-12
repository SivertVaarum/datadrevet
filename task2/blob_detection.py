import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage


def detect_blobs(
    img,
    min_sigma=2,
    max_sigma=30,
    num_sigma=10,
    threshold=0.1
):
    img_gray = skimage.color.rgb2gray(img)

    blobs = skimage.feature.blob_log(
        img_gray,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold
    )

    # blob_log returns y-position, x-position and sigma.
    # For a 2D Laplacian of Gaussian blob, the approximate
    # radius is sqrt(2) * sigma.
    blobs[:, 2] *= np.sqrt(2)

    return img_gray, blobs


def print_blob_statistics(blobs):
    print(f"Number of detected blobs: {len(blobs)}")

    if len(blobs) == 0:
        return

    radii = blobs[:, 2]
    areas = np.pi * radii ** 2

    print(f"Smallest radius: {radii.min():.2f}")
    print(f"Largest radius:  {radii.max():.2f}")
    print(f"Average radius:  {radii.mean():.2f}")
    print(f"Average area:    {areas.mean():.2f}")
    print()

    for index, (y, x, radius) in enumerate(blobs, start=1):
        area = np.pi * radius ** 2

        print(
            f"Blob {index}: "
            f"position=({x:.1f}, {y:.1f}), "
            f"radius={radius:.1f}, "
            f"estimated area={area:.1f}"
        )


def display_figures(
    img,
    min_sigma=2,
    max_sigma=30,
    num_sigma=10,
    threshold=0.1
):
    img_gray, blobs = detect_blobs(
        img,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold
    )

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(img_gray, cmap="gray")
    axs[1].set_title("Grayscale Image")
    axs[1].axis("off")

    axs[2].imshow(img)
    axs[2].set_title(f"Detected Blobs: {len(blobs)}")
    axs[2].axis("off")

    for y, x, radius in blobs:
        circle = plt.Circle(
            (x, y),
            radius,
            color="red",
            fill=False,
            linewidth=1.5
        )
        axs[2].add_patch(circle)

    parameter_text = (
        f"min sigma: {min_sigma}\n"
        f"max sigma: {max_sigma}\n"
        f"num sigma: {num_sigma}\n"
        f"threshold: {threshold}"
    )

    axs[2].text(
        0.02,
        0.98,
        parameter_text,
        transform=axs[2].transAxes,
        verticalalignment="top",
        color="white",
        bbox={
            "facecolor": "black",
            "alpha": 0.7,
            "pad": 5
        }
    )

    fig.tight_layout()

    print_blob_statistics(blobs)


def load_image(file_num):
    match file_num:
        case "1":
            path = "./task2/r7bthvstxw-1/pickup/PIC_87.jpg"
        case "2":
            path = "./task2/r7bthvstxw-1/motorcycle/PIC_153.jpg"
        case "3":
            path = "./task2/r7bthvstxw-1/sedan/PIC_208.jpg"
        case _:
            print(f"Unknown image number: {file_num}")
            print("Using image 1 by default.")
            path = "./task2/r7bthvstxw-1/pickup/PIC_87.jpg"

    print(f"Loading: {path}")
    return skimage.io.imread(path)


def main():
    file_num = sys.argv[1] if len(sys.argv) > 1 else "1"

    img = load_image(file_num)

    display_figures(
        img,
        min_sigma=2,
        max_sigma=30,
        num_sigma=10,
        threshold=0.1
    )

    plt.show()


if __name__ == "__main__":
    main()