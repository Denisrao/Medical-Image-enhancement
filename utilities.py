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

def enhance_image_cv2(img_path, apply_gamma=False, gamma=1.2):
    """
    Enhance an image using OpenCV: denoising, contrast adjustment with CLAHE, and sharpening.
    
    Parameters:
        img_path (str): Path to the input image.
        apply_gamma (bool): Whether to apply gamma correction.
        gamma (float): Gamma value for brightness control.
    
    Returns:
        enhanced (np.array): The enhanced image.
    """
    
    # Load image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at path: {img_path}")

    # Step 1: Denoise using Non-local Means Denoising
    denoised = cv2.fastNlMeansDenoising(img, h=10)

    # Step 2: Apply CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(denoised)

    # Step 3: Apply sharpening filter
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5,-1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_img, -1, sharpen_kernel)

    # Step 4: Optional Gamma Correction
    if apply_gamma:
        gamma_corrected = np.power(sharpened / 255.0, gamma)
        sharpened = np.uint8(gamma_corrected * 255)

    return sharpened
# Use the function
enhanced_img = enhance_image_cv2("noisyimage.jpeg", apply_gamma=True, gamma=1.2)

plt.figure(figsize=(12,8))
plt.subplots_adjust(hspace=0.8,wspace=0.3)   #hspace = space b/w rows & wspace = space b/w coloumns

# Show original image
plt.subplot(3, 3, 1)
plt.title("Original Noised Image")
plt.imshow(img, cmap='gray')

# Show Denoised
plt.subplot(3, 3, 2)
plt.title("Denoised Image")
plt.imshow(nlm_denoised, cmap='gray')

# Show equalized
plt.subplot(3, 3, 3)
plt.title("Equalized Image")
plt.imshow(equalization, cmap='gray')

# Show the enhanced image
plt.subplot(3,3,4)
plt.title("Enhanced Image")
plt.imshow(enhanced_img,cmap='gray')
plt.show()