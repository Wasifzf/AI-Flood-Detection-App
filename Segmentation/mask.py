# segmentation/mask.py

import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from segmentation.main import unet

# Load model
model = unet()

def resize_image(image):
    image = np.squeeze(image)
    image = tf.cast(image, dtype=tf.float32)
    image = image / 255.0
    image = tf.image.resize(image, (128, 128))
    return image

def generate_mask(pil_img):
    """Takes a PIL image, returns predicted mask as a PIL image (grayscale)"""
    img_rgb = np.array(pil_img.convert("RGB"))
    input_tensor = tf.expand_dims(resize_image(img_rgb), axis=0)

    predicted_mask = model.predict(input_tensor)[0, ..., 0]
    predicted_mask = tf.round(predicted_mask).numpy()  # Binary mask

    # Resize back to original image size
    mask_resized = cv2.resize(predicted_mask, pil_img.size, interpolation=cv2.INTER_NEAREST)
    mask_uint8 = (mask_resized * 255).astype(np.uint8)

    return Image.fromarray(mask_uint8)


# Process an entire video and save mask output
def process_video(video_path, output_path):
    """Takes input video path and outputs grayscale video of binary masks."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
        isColor=True  # must be True for 3-channel frames
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = resize_image(frame_rgb)
        input_tensor = tf.expand_dims(frame_tensor, axis=0)

        pred = model.predict(input_tensor)
        pred_mask = tf.round(pred[0, ..., 0]).numpy()

        mask_resized = cv2.resize(pred_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_uint8 = (mask_resized * 255).astype(np.uint8)

        # Convert grayscale single channel to 3-channel BGR before writing
        mask_3ch = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)

        out.write(mask_3ch)

    cap.release()
    out.release()
    print("✅ Black and white mask video saved at:", output_path)