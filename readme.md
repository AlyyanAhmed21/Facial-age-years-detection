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
```

## ⚙️ Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/AlyyanAhmed21/Facial-Age-Detection.git
    cd Facial-Age-Detection
    ```

2.  **Create a Virtual Environment**
    ```bash
    conda create -n face-env python=3.8 -y
    conda activate face-env
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

### 1. Run the Web Application
To start the interface for inference (Image/Video/Webcam):
```bash
streamlit run app.py
```

### 2. Train the Model
To re-run the training pipeline (Ingestion -> Preparation -> Training):
```bash
python main.py
```

## 🧠 Model Architecture

The model utilizes a **Shared Backbone** strategy:
1.  **Input:** Preprocessed Face Crop ($224 \times 224$).
2.  **Backbone:** EfficientNet (pretrained) acts as the feature extractor.
3.  **Heads:** Three separate Fully Connected layers branch off the pooled features:
    *   **Age Head:** Predicts age ranges.
    *   **Gender Head:** Predicts Male/Female.
    *   **Race Head:** Predicts ethnicity.
4.  **Loss Calculation:** A weighted sum loss function allows the model to learn all three tasks while prioritizing Age accuracy.

## 📊 Dataset
The project is trained on the **FairFace** dataset, which is designed to reduce racial bias in facial analysis algorithms. The pipeline automatically downloads and formats this data from the Hugging Face Hub.
