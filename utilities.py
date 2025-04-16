import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("brainmodel.h5")

# Enhance the image (CLAHE + Denoise + Sharpen)
def enhance_image(img):
    denoised = cv2.fastNlMeansDenoising(img, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(contrast, -1, sharpen_kernel)
    return sharpened

# Preprocess for model input
def preprocess_for_model(img):
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img.reshape(1, 128, 128, 1)

# Get image from user
image_path = input("Enter the path of the MRI image: ").strip().strip('"')
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(" Image not found. Check the path and try again !!! ")
    exit()

print(" Image Fetched Successfully. Working on it...")

# Enhance the image
enhanced_img = enhance_image(img)

# Predict using the model
input_img = preprocess_for_model(enhanced_img)
prediction = model.predict(input_img)[0][0]

#  Threshold decision
print(f"🔍 Prediction score: {prediction:.4f}")
if prediction > 0.5:
    print(" Tumor Detected! ")

    # Basic segmentation using threshold
    _, mask = cv2.threshold(enhanced_img, 150, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

    # Calculate tumor % area
    brain_area = np.sum(enhanced_img > 10)
    tumor_area = np.sum(mask > 0)
    tumor_percent = (tumor_area / brain_area) * 100 if brain_area > 0 else 0

    # Red overlay on tumor
    overlay = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    overlay[mask > 0] = [0, 0, 255]
    cv2.putText(overlay, f"Tumor: {tumor_percent:.2f}%", (10, 30),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
# It is developed by Denis Yadav
else:
    print(" No Tumor Detected.")
    overlay = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(overlay, "No Tumor Detected", (10, 10),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)

#  Show the result
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title("Tumor Detection Result")
plt.axis('off')
plt.tight_layout()
plt.show()

#  Optional: Save the result
cv2.imwrite("tumor_overlay_result.jpg", overlay)
print("Output saved as 'tumor_overlay_result.jpg'")
