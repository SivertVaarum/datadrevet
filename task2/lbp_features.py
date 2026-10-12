import sys

import matplotlib.pyplot as plt
import numpy as np
import skimage


def calculate_lbp(img_gray):
    rows, cols = img_gray.shape
    lbp_image = np.zeros((rows, cols), dtype=np.uint8)

    # Clockwise ordering of the eight neighbours
    neighbours = [
        (-1, -1),  # Top-left
        (-1, 0),   # Top
        (-1, 1),   # Top-right
        (0, 1),    # Right
        (1, 1),    # Bottom-right
        (1, 0),    # Bottom
        (1, -1),   # Bottom-left
        (0, -1)    # Left
    ]

    # Ignore the outermost pixels because they do not
    # have eight surrounding neighbours
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            center_pixel = img_gray[y, x]
            lbp_value = 0

            for bit, (dy, dx) in enumerate(neighbours):
                neighbour_pixel = img_gray[y + dy, x + dx]

                if neighbour_pixel >= center_pixel:
                    lbp_value |= 1 << bit

            lbp_image[y, x] = lbp_value

    return lbp_image


def calculate_histogram(lbp_image):
    # Ignore the border pixels, since no LBP value was calculated there
    valid_pixels = lbp_image[1:-1, 1:-1]

    histogram, _ = np.histogram(
        valid_pixels,
        bins=256,
        range=(0, 256)
    )

    # Normalize the histogram to make images of different sizes comparable
    histogram = histogram.astype(float)
    histogram /= histogram.sum()

    return histogram


def display_figures(img):
    img_gray = skimage.color.rgb2gray(img)

    lbp_image = calculate_lbp(img_gray)
    histogram = calculate_histogram(lbp_image)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    axs[0].imshow(img_gray, cmap="gray")
    axs[0].set_title("Original Grayscale Image")
    axs[0].axis("off")

    axs[1].imshow(lbp_image, cmap="gray", vmin=0, vmax=255)
    axs[1].set_title("LBP Image")
    axs[1].axis("off")

    axs[2].bar(
        np.arange(256),
        histogram,
        width=1.0,
        color="black"
    )
    axs[2].set_title("Normalized LBP Histogram")
    axs[2].set_xlabel("LBP value")
    axs[2].set_ylabel("Relative frequency")
    axs[2].set_xlim(0, 255)

    fig.tight_layout()


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
    display_figures(img)

    plt.show()


if __name__ == "__main__":
    main()
