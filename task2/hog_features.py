import skimage

import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 3, figsize=(15, 5))


def display_figures(img, y):
	img_gray = skimage.color.rgb2gray(img) 

	gradient = skimage.filters.sobel(img_gray)

	fd, hog_image = skimage.feature.hog(img, orientations=9, pixels_per_cell=(4, 4),
						cells_per_block=(8, 8), visualize=True, channel_axis=-1)

	axs[y][0].imshow(img)
	axs[y][0].set_title(f"Original Image {y+1}")
	axs[y][0].axis("off")

	axs[y][1].imshow(gradient, cmap="gray")
	axs[y][1].set_title(f"Gradient Image {y+1}")
	axs[y][1].axis("off")

	axs[y][2].imshow(hog_image, cmap="gray")
	axs[y][2].set_title(f"HOG Feature Image {y+1}")
	axs[y][2].axis("off")


img_1 = skimage.io.imread("./task2/r7bthvstxw-1/pickup/PIC_87.jpg")
img_2 = skimage.io.imread("./task2/r7bthvstxw-1/motorcycle/PIC_153.jpg")
img_3 = skimage.io.imread("./task2/r7bthvstxw-1/sedan/PIC_208.jpg")

display_figures(img_1, 0)
display_figures(img_2, 1)
display_figures(img_3, 2)

plt.show()