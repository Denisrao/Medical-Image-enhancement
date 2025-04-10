import cv2
import numpy as np
import matplotlib.pyplot as plt

# grayscale
img = cv2.imread("noisyimage.jpeg", cv2.IMREAD_GRAYSCALE)

# Apply Median Filter
median_denoised = cv2.medianBlur(img, 3)    #applying median filtering

# Apply Gaussian Filter
gaussian_denoised = cv2.GaussianBlur(img, (5, 5), 1.5)   # smoothing image by gaussian kernel

# Apply Non-Local Means Denoising (preserves edges)
nlm_denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)    #removing noise and preserving edges

# Applyinh histogram equalization to maintain the contrast (Global contrast)
equalization = cv2.equalizeHist(img)

# Show results
plt.figure(figsize=(12, 8))

plt.subplot(3, 3, 1)
plt.title("Original Noised Image")
plt.imshow(img, cmap='gray')

# plt.subplot(2, 2, 2)
# plt.title("Median Filter")
# plt.imshow(median_denoised, cmap='gray')

# plt.subplot(2, 2, 3)
# plt.title("Gaussian Filter")
# plt.imshow(gaussian_denoised, cmap='gray')

plt.subplot(3, 3, 2)
plt.title("Denoised Image")
plt.imshow(nlm_denoised, cmap='gray')

plt.subplot(3,3,3)
plt.title("Equalized Image")
plt.imshow(equalization,cmap='gray')

plt.show()