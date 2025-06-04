import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from pathlib import Path
import sys
from model import build_model
import os

# Load model
model = build_model()

# Use relative path
weights_path = os.path.join("classification", "weights", "best_flood_model.h5")

# Load weights if available
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Weights loaded successfully.")
else:
    print(f"Model weights not found at: {weights_path}")
# If you have a different path structure, adjust accordingly

# Image parameters
img_size = (224, 224)

def predict_image(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prob = model.predict(img_array)[0][0]
    pred_class = 1 if prob >= 0.5 else 0

    print(f"\nImage: {img_path}")
    print(f"Prediction probability (flooded): {prob:.4f}")
    print(f"Predicted class: {pred_class} ({'Flooded' if pred_class == 1 else 'Not Flooded'})")

