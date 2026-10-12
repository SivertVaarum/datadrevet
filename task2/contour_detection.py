import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage


def detect_contours(
    img,
    gaussian_sigma=1.0,
    threshold=None,
    min_object_size=50
):
    img_gray = skimage.color.rgb2gray(img)

    img_blurred = skimage.filters.gaussian(
        img_gray,
        sigma=gaussian_sigma
    )

    if threshold is None:
        threshold = skimage.filters.threshold_otsu(img_blurred)

    binary_image = img_blurred > threshold

    binary_image = skimage.morphology.remove_small_objects(
        binary_image,
        min_size=min_object_size
    )

    contours = skimage.measure.find_contours(
        binary_image,
        level=0.5
    )

    return img_gray, binary_image, contours, threshold


def calculate_contour_statistics(contour):
    y = contour[:, 0]
    x = contour[:, 1]

    # Distance between consecutive contour points
    dx = np.diff(x, append=x[0])
    dy = np.diff(y, append=y[0])

    perimeter = np.sum(np.sqrt(dx ** 2 + dy ** 2))

    # Shoelace formula
    area = 0.5 * np.abs(
        np.dot(x, np.roll(y, 1))
        - np.dot(y, np.roll(x, 1))
    )

    return area, perimeter


def print_contour_statistics(contours):
    print(f"Number of detected contours: {len(contours)}")

    if len(contours) == 0:
        return

    measurements = []

    for index, contour in enumerate(contours, start=1):
        area, perimeter = calculate_contour_statistics(contour)
        measurements.append((area, perimeter))

        print(
            f"Contour {index}: "
            f"area={area:.1f}, "
            f"perimeter={perimeter:.1f}"
        )

    areas = np.array([measurement[0] for measurement in measurements])
    perimeters = np.array(
        [measurement[1] for measurement in measurements]
    )

    print()
    print(f"Average area:      {areas.mean():.2f}")
    print(f"Largest area:      {areas.max():.2f}")
    print(f"Average perimeter: {perimeters.mean():.2f}")


def display_figures(
    img,
    gaussian_sigma=1.0,
    threshold=None,
    min_object_size=50
):
    img_gray, binary_image, contours, used_threshold = detect_contours(
        img,
        gaussian_sigma=gaussian_sigma,
        threshold=threshold,
        min_object_size=min_object_size
    )

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(binary_image, cmap="gray")
    axs[1].set_title("Binary Image")
    axs[1].axis("off")

    axs[2].imshow(img)
    axs[2].set_title(f"Detected Contours: {len(contours)}")
    axs[2].axis("off")

    colors = plt.cm.hsv(
        np.linspace(0, 1, max(len(contours), 1))
    )

    for contour, color in zip(contours, colors):
        axs[2].plot(
            contour[:, 1],
            contour[:, 0],
            color=color,
            linewidth=2
        )

    parameter_text = (
        f"Gaussian sigma: {gaussian_sigma}\n"
        f"Threshold: {used_threshold:.3f}\n"
        f"Minimum size: {min_object_size}"
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

    print_contour_statistics(contours)


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
        gaussian_sigma=1.0,
        threshold=None,
        min_object_size=50
    )

    plt.show()


if __name__ == "__main__":
    main()