import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image (grayscale)
img = cv2.imread("noisyimage.jpeg", cv2.IMREAD_GRAYSCALE)

# Apply Median Filter
median_denoised = cv2.medianBlur(img, 3)

# Apply Gaussian Filter
gaussian_denoised = cv2.GaussianBlur(img, (5, 5), 1.5)

# Apply Non-Local Means Denoising (preserves edges)
nlm_denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

# Show results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original Noisy Image")
plt.imshow(img, cmap='gray')

plt.subplot(2, 2, 2)
plt.title("Median Filter")
plt.imshow(median_denoised, cmap='gray')

plt.subplot(2, 2, 3)
plt.title("Gaussian Filter")
plt.imshow(gaussian_denoised, cmap='gray')

plt.subplot(2, 2, 4)
plt.title("Non-local Means")
plt.imshow(nlm_denoised, cmap='gray')

plt.tight_layout()
plt.show()