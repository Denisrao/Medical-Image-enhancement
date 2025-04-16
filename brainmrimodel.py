import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
def enhance_image(img):
    denoised = cv2.fastNlMeansDenoising(img, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_img = clahe.apply(denoised)
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(contrast_img, -1, sharpen_kernel)
    return sharpened
data = []
labels = []
path = "brain_mri_dataset"

for label, folder in enumerate(["no", "yes"]):  # no = 0, yes = 1
    folder_path = os.path.join(path, folder)
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            enhanced = enhance_image(img)                # Apply enhancement
            resized = cv2.resize(enhanced, (128, 128))   # Resize
            data.append(resized)
            labels.append(label)
X = np.array(data).reshape(-1, 128, 128, 1) / 255.0  # Normalize
y = np.array(labels)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Optional: Save best model during training
checkpoint = ModelCheckpoint("brainmodel.h5", monitor='val_accuracy', save_best_only=True)

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=16,
                    validation_data=(X_test, y_test),
                    callbacks=[checkpoint])
loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Final Test Accuracy: {accuracy * 100:.2f}%")

# Save model manually (if not using checkpoint)
model.save("brainmodel.h5")
print("💾 Model saved as brainmodel.h5")