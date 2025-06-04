import streamlit as st
import numpy as np
import os
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from classification.model import build_model
from segmentation.mask import generate_mask, process_video
import shutil
import tempfile
from io import BytesIO

import subprocess


def reencode_video_ffmpeg(input_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",  # overwrite output
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",  # ensures browser compatibility
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# Load classification model
@st.cache_resource
def load_flood_model():
    model = build_model()
    weights_path = os.path.join("classification", "weights", "best_flood_model.h5")
    model.load_weights(weights_path)
    return model


model = load_flood_model()

st.title("🌊 Flood Detection and Segmentation App")
st.write("Upload an image or video to classify and optionally segment flooded areas.")

# === IMAGE SECTION ===
st.header("🖼️ Image Upload & Segmentation")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # --- Classification ---
    resized_img = img.resize((224, 224))
    img_array = image.img_to_array(resized_img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prob = model.predict(img_array)[0][0]
    pred_class = 1 if prob >= 0.5 else 0

    st.subheader("Classification Result")
    st.write(f"**Confidence (Flooded):** {prob:.4f}")
    st.write(f"**Predicted Class:** {'Flooded 🌧️' if pred_class == 1 else 'Not Flooded ☀️'}")

    # --- Segmentation ---
    if pred_class == 1:
        st.subheader("Segmentation Mask")

        with st.spinner("Generating flood mask..."):
            mask = generate_mask(img)
            st.image(mask, caption="Flood Segmentation Mask", use_column_width=True)

# === VIDEO SECTION ===
st.header("🎥 Video Upload & Mask Generation")
video_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"], key="video_upload")

if video_file is not None:
    st.video(video_file)  # Show original video before processing

    # Use a temporary directory to store files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/input.mp4"
        raw_output_path = f"{tmpdir}/raw_masked_output.mp4"
        final_output_path = f"{tmpdir}/masked_output.mp4"

        # Save uploaded video
        with open(input_path, "wb") as f:
            f.write(video_file.read())

        # Process and re-encode only once per uploaded file
        if (not hasattr(st.session_state, "last_processed_video") or
                st.session_state.last_processed_video != video_file.name):
            with st.spinner("Processing video and generating flood masks..."):
                process_video(input_path, raw_output_path)
                reencode_video_ffmpeg(raw_output_path, final_output_path)

            st.session_state.last_processed_video = video_file.name
            with open(final_output_path, "rb") as f:
                st.session_state.processed_video_bytes = f.read()

        # Display processed video
        st.success("✅ Masked video generated successfully!")
        st.video(BytesIO(st.session_state.processed_video_bytes))

        # Download option
        st.download_button(
            label="⬇️ Download Masked Video",
            data=st.session_state.processed_video_bytes,
            file_name="flood_mask_output.mp4",
            mime="video/mp4"
        )

