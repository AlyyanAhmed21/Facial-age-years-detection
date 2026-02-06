# Facial Demographics Analysis 👤

A robust Deep Learning application capable of detecting **Age, Gender, and Race** from images, videos, and live webcam feeds. This project utilizes a Multi-Task Learning approach with a shared EfficientNet backbone to perform simultaneous demographic classifications.

## 🚀 Features

*   **Multi-Task Learning:** Single model architecture predicting Age, Gender, and Race simultaneously.
*   **Real-Time Inference:** Optimized pipeline for live webcam analysis using lightweight face detection.
*   **Video Processing:** Full video file analysis with frame-by-frame annotation and export capabilities.
*   **MLOps Pipeline:** Modular code structure handling Data Ingestion, Preparation, and Model Training.
*   **Smart Face Detection:** Integrates MTCNN for accurate face localization before classification.

## 🛠️ Tech Stack

*   **Frameworks:** PyTorch, TensorFlow (for MTCNN), Transformers (Hugging Face)
*   **Base Models:** EfficientNet / EfficientFormer
*   **Interface:** Streamlit
*   **Data Processing:** Pandas, NumPy, OpenCV, Pillow
*   **Dataset:** FairFace (sourced via Hugging Face Hub)

## 📂 Project Structure

```text
├── config/              # Configuration files (YAML)
├── src/
│   └── cnnClassifier/
│       ├── components/  # Logic for Ingestion, Prep, and Training
│       ├── pipeline/    # Orchestration of components
│       └── entity/      # Data classes for configuration
├── app.py               # Streamlit Frontend application
├── main.py              # Training pipeline entry point
├── requirements.txt     # Dependencies
└── template.py          # Project scaffolding script
