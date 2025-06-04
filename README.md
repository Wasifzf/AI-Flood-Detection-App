# 🌊 AI Flood Detection App

This project uses deep learning for automated flood detection and segmentation from satellite or aerial imagery. It combines image classification and semantic segmentation to detect whether an image contains a flood and highlights flooded regions in images and videos.

## 📌 Features

- **Flood Classification**: Classifies images as `Flood` or `Non-Flood` with a confidence score.
- **Flood Segmentation**: Detects and segments flooded regions in input images and videos using a U-Net-based architecture.
- **Streamlit Web App**: Simple interface to upload and analyze images and videos with a download button.

---

## 📂 Project Structure

AI-Flood-Detection/

├── classification/

│ ├── main.py

│ ├── model.py # CNN for flood classification

│ └── weights/ # Pretrained classification weights downloadable 

├── segmentation/

│ ├── main.py

│ ├── model.py # U-Net model architecture

│ ├── mask.py # Flood mask prediction

│ └── weights/ # Pretrained segmentation weights, downloaded via drive link and creates directory 

├── app.py # Streamlit app

├── README.md # Project overview

├── .gitignore # Ignore large files

├── requirements.txt # Python dependencies


---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/AI-Flood-Detection.git
cd AI-Flood-Detection
pip install -r requirements.txt
```

---

## 📦 Model Weights

To avoid exceeding GitHub’s file size limits, segmentation (U-net) model weights are not included in the repository. Instead:

Classification model: Included and available to download.

Segmentation model: Automatically downloaded from Google Drive if missing.

You can modify the paths or use your own model weights.

---

## 🚀 Run the App

```
streamlit run app.py
```

Then open the provided local URL in your browser (usually http://localhost:8501).

---

## 📸 Sample Use

1. Upload an image.

2. The model predicts if it shows flooding.

3. If flood detected, a segmentation mask is generated to highlight flooded areas.

4. Upload a video, a segmentation mask is generated to highlight flooded areas.

5. Download processed video.

---

## 🧠 Models

Classifier: Convolutional Neural Network (CNN) trained on flood/non-flood images.

Segmenter: U-Net architecture trained for binary segmentation of flood regions.

---

## 🧾 Requirements

Install dependancies using:

```
pip install -r requirements.txt
```

---

## 🙏 Acknowledgements

Sentinel Hub & Open Aerial Imagery

Keras, TensorFlow, Streamlit

Google Drive & gdown for model weight hosting

---

## 📄 License

This project is licensed under the MIT License. You are free to use, modify, and distribute it with attribution.



