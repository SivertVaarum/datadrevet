import cv2, numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("r7bthvstxw-1/suv/PIC_0.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (256, 256)).astype(np.float64) / 255.0

def display_image(image, title="Image"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

# display_image(img, "Gray Scale Image")



rows, cols = img.shape
u, v = np.meshgrid(np.arange(cols) - cols//2, np.arange(rows) - rows//2)
D = np.sqrt(u**2 + v**2)   # distance from centre (zero frequency)



# 2D DFT
F = np.fft.fft2(img)
Fshift = np.fft.fftshift(F)
magnitude = np.log(1 + np.abs(Fshift))

# Visualisering
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original image (grayscale, 256×256)")
axes[0].axis("off")

axes[1].imshow(magnitude, cmap="gray")
axes[1].set_title("Magnitude spectrum (log scale, centred)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("task1_spectrum.png", dpi=150)
plt.show()




# Low pass filter

# --- Hjelpefunksjoner ---
def apply_filter(image, H):
    Fs = np.fft.fftshift(np.fft.fft2(image))
    return np.real(np.fft.ifft2(np.fft.ifftshift(Fs * H)))

def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    return 10 * np.log10(1.0 / mse(a, b))

# --- Spekteret og total energi ---
Fshift = np.fft.fftshift(np.fft.fft2(img))
total_energy = np.sum(np.abs(Fshift) ** 2)

# --- Filtrering ---
cutoffs = [10, 30, 60]
masks, results = {}, {}
for D0 in cutoffs:
    masks[("Ideal", D0)] = (D <= D0).astype(float)
    masks[("Gaussian", D0)] = np.exp(-(D ** 2) / (2 * D0 ** 2))
for key, H in masks.items():
    results[key] = apply_filter(img, H)

# --- Tall ---
for (name, D0), out in results.items():
    kept = np.sum(np.abs(Fshift * masks[(name, D0)]) ** 2) / total_energy * 100
    print(f"{name:8s} D0={D0:3d}: MSE = {mse(img, out):.5f}, "
          f"PSNR = {psnr(img, out):.2f} dB, energy kept = {kept:.2f} %")

# --- Figur 1: filtermaskene ---
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, name in enumerate(["Ideal", "Gaussian"]):
    for j, D0 in enumerate(cutoffs):
        axes[i, j].imshow(masks[(name, D0)], cmap="gray", vmin=0, vmax=1)
        axes[i, j].set_title(f"{name} LPF mask, D0 = {D0}")
        axes[i, j].axis("off")
plt.tight_layout(); plt.savefig("task2_masks.png", dpi=150); plt.show()

# --- Figur 2: filtrerte bilder ---
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, name in enumerate(["Ideal", "Gaussian"]):
    axes[i, 0].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[i, 0].set_title("Original")
    for j, D0 in enumerate(cutoffs):
        out = results[(name, D0)]
        axes[i, j + 1].imshow(out, cmap="gray", vmin=0, vmax=1)
        axes[i, j + 1].set_title(f"{name}, D0 = {D0}\nPSNR {psnr(img, out):.2f} dB")
    for ax in axes[i]: ax.axis("off")
plt.tight_layout(); plt.savefig("task2_filtered.png", dpi=150); plt.show()

# --- Figur 3: det som ble fjernet (original minus filtrert) ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, D0 in zip(axes, cutoffs):
    removed = img - results[("Gaussian", D0)]
    ax.imshow(np.abs(removed), cmap="gray")
    ax.set_title(f"Removed content, Gaussian D0 = {D0}")
    ax.axis("off")
plt.tight_layout(); plt.savefig("task2_removed.png", dpi=150); plt.show()

# --- Figur 4: utsnitt for å se ringing ---
crop = (slice(140, 210), slice(130, 230))  # rundt skiltet, juster ved behov
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (title, im) in zip(axes, [("Original", img),
                                  ("Ideal, D0 = 30", results[("Ideal", 30)]),
                                  ("Gaussian, D0 = 30", results[("Gaussian", 30)])]):
    ax.imshow(im[crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.savefig("task2_ringing_zoom.png", dpi=150); plt.show()




# ===== TASK 3: HIGH-PASS FILTER =====

cutoffs = [10, 30, 60]
hp_masks, hp_results = {}, {}

# --- Lag høypassmasker (det motsatte av lavpass) ---
for D0 in cutoffs:
    hp_masks[("Ideal", D0)] = 1 - (D <= D0).astype(float)
    hp_masks[("Gaussian", D0)] = 1 - np.exp(-(D ** 2) / (2 * D0 ** 2))
for key, H in hp_masks.items():
    hp_results[key] = apply_filter(img, H)

# --- Figur 1: høypassmaskene ---
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, name in enumerate(["Ideal", "Gaussian"]):
    for j, D0 in enumerate(cutoffs):
        axes[i, j].imshow(hp_masks[(name, D0)], cmap="gray", vmin=0, vmax=1)
        axes[i, j].set_title(f"{name} HPF mask, D0 = {D0}")
        axes[i, j].axis("off")
plt.tight_layout(); plt.savefig("task3_masks.png", dpi=150); plt.show()

# --- Figur 2: høypassresultatene (kantene) ---
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, name in enumerate(["Ideal", "Gaussian"]):
    for j, D0 in enumerate(cutoffs):
        edges = np.abs(hp_results[(name, D0)])
        axes[i, j].imshow(edges, cmap="gray", vmin=0, vmax=np.percentile(edges, 99))
        axes[i, j].set_title(f"{name} HPF, D0 = {D0}")
        axes[i, j].axis("off")
plt.tight_layout(); plt.savefig("task3_highpass.png", dpi=150); plt.show()

# --- Figur 3: kantforsterkning (high-boost) ---
k = 1.5
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
axes[0].imshow(img, cmap="gray", vmin=0, vmax=1); axes[0].set_title("Original")
for ax, D0 in zip(axes[1:], cutoffs):
    enhanced = np.clip(img + k * hp_results[("Gaussian", D0)], 0, 1)
    ax.imshow(enhanced, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Enhanced (Gaussian D0 = {D0}, k = {k})")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.savefig("task3_enhanced.png", dpi=150); plt.show()

# --- Figur 4: utsnitt av skiltet, før og etter ---
crop = (slice(140, 210), slice(130, 230))  # samme utsnitt som i Task 2
enhanced_30 = np.clip(img + k * hp_results[("Gaussian", 30)], 0, 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(img[crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes[0].set_title("Original (zoom)")
axes[1].imshow(enhanced_30[crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
axes[1].set_title(f"Enhanced, Gaussian D0 = 30, k = {k} (zoom)")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.savefig("task3_enhanced_zoom.png", dpi=150); plt.show()