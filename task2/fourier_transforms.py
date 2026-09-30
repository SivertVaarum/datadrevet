import cv2, numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("r7bthvstxw-1/suv/PIC_0.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256)).astype(np.float64) / 255.0

def display_image(image, title="Image"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

display_image(img, "Gray Scale Image")



rows, cols = img.shape
u, v = np.meshgrid(np.arange(cols) - cols//2, np.arange(rows) - rows//2)
D = np.sqrt(u**2 + v**2)   # distance from centre (zero frequency)