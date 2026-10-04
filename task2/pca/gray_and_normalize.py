import numpy as np
from skimage import io

def load_gray01(path):
    """Load an image, convert to grayscale, and return a float array in [0, 1]."""
    img = io.imread(path)

    # Drop alpha channel if present (RGBA -> RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]

    # Convert to grayscale (luminance weights) if the image has color channels
    if img.ndim == 3:
        img = img.astype(np.float64)
        img = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    else:
        img = img.astype(np.float64)

    # Normalize by the data type's maximum value
    # (done before conversion would need dtype; here we infer from the values)
    if img.max() > 255:
        img = img / 65535.0      # 16-bit
    elif img.max() > 1:
        img = img / 255.0        # 8-bit
    # else: already in [0, 1]

    return np.clip(img, 0, 1)

print(load_gray01("PIC_0.jpg"))