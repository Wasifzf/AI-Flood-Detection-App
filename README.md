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
