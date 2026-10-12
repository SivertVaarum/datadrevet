import skimage
import matplotlib.pyplot as plt
import sys

fig, axs = plt.subplots(1, 3, figsize=(15, 5))


def display_figures(img):
	img_gray = skimage.color.rgb2gray(img) 

	gradient = skimage.filters.sobel(img_gray)

	fd, hog_image = skimage.feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
						cells_per_block=(2, 2), visualize=True, channel_axis=-1)

	axs[0].imshow(img)
	axs[0].set_title(f"Original Image")
	axs[0].axis("off")

	axs[1].imshow(gradient, cmap="gray")
	axs[1].set_title(f"Gradient Image")
	axs[1].axis("off")

	axs[2].imshow(hog_image, cmap="gray")
	axs[2].set_title(f"HOG Feature Image")
	axs[2].axis("off")


def main():
	file_num = sys.argv[1]
	match file_num:
		case "1":
			img = skimage.io.imread("./task2/r7bthvstxw-1/pickup/PIC_87.jpg")
		case "2":
			img = skimage.io.imread("./task2/r7bthvstxw-1/motorcycle/PIC_153.jpg")
		case "3":
			img = skimage.io.imread("./task2/r7bthvstxw-1/sedan/PIC_208.jpg")
		case _:
			img = skimage.io.imread("./task2/r7bthvstxw-1/pickup/PIC_87.jpg")

	display_figures(img)
	plt.show()

if __name__ == "__main__":
	main()