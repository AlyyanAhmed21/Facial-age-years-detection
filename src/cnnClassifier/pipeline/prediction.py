import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
import cv2
from mtcnn import MTCNN  # For high-quality
from pathlib import Path
import sys
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from safetensors.torch import load_file as load_safetensors

try:
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if src_path not in sys.path: sys.path.append(src_path)
    from components.multi_task_model_trainer import MultiTaskEfficientNet
    from utils.common import read_yaml
except ImportError:
    # Fallback for Hugging Face Spaces
    from src.cnnClassifier.components.multi_task_model_trainer import MultiTaskEfficientNet
    from src.cnnClassifier.utils.common import read_yaml

class PredictionPipeline:
    def __init__(self, repo_id: str = "ALYYAN/Facial-Age-Det"):
        self.device = "cpu"
        self.repo_id = repo_id
        
        print("--- Initializing Prediction Pipeline by downloading artifacts from Hub ---")
        
        # --- THE FIX: Download all artifacts from your HF Model Repo ---
        self.model_path = hf_hub_download(repo_id=self.repo_id, filename="checkpoint-26873/model.safetensors")
        self.params_path = hf_hub_download(repo_id=self.repo_id, filename="params.yaml")
        self.data_csv_path = hf_hub_download(repo_id=self.repo_id, filename="fairface_cleaned.csv")
        # --- END FIX ---
        
        self.base_model_name = "google/efficientnet-b2"
        self.params = read_yaml(Path(self.params_path))
        
        self.label_maps = self._load_label_maps()
        self.processor = AutoImageProcessor.from_pretrained(self.base_model_name)
        self.transforms = Compose([Resize((self.params.IMAGE_SIZE, self.params.IMAGE_SIZE)), ToTensor(), Normalize(mean=self.processor.image_mean, std=self.processor.image_std)])
        self.model = self._load_model()
        
        haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(haar_cascade_path)
        
        print(f"--- Pipeline Initialized Successfully on device: {self.device} ---")
    
    def _load_model(self):
        num_age, num_gender, num_race = len(self.label_maps['age_id2label']), len(self.label_maps['gender_id2label']), 7
        model = MultiTaskEfficientNet(self.base_model_name, num_age, num_gender, num_race)
        weight_file = self.model_path / 'model.safetensors'
        if not weight_file.exists(): weight_file = self.model_path / 'pytorch_model.bin'
        if not weight_file.exists(): raise FileNotFoundError(f"Weights not found in {self.model_path}")
        state_dict = load_safetensors(weight_file, device="cpu") if weight_file.suffix == ".safetensors" else torch.load(weight_file, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _draw_predictions(self, image, box, labels):
        x, y, w, h = [int(c) for c in box]
        font, font_scale, font_thickness = cv2.FONT_HERSHEY_DUPLEX, 0.6, 1
        text_color, bg_color = (255, 255, 255), (255, 75, 75)
        text_lines = [f"Gender: {labels['gender']}", f"Age: {labels['age']}"]
        max_width, line_height = 0, 25
        for line in text_lines:
            (w_text, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            if w_text > max_width: max_width = w_text
        total_height = len(text_lines) * line_height - 5
        cv2.rectangle(image, (x, y), (x + w, y + h), bg_color, 2)
        cv2.rectangle(image, (x-1, y - total_height), (x + max_width + 10, y), bg_color, -1)
        for i, line in enumerate(text_lines):
            y_text = y - total_height + (i * line_height) + 18
            cv2.putText(image, line, (x + 5, y_text), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    def predict_hq(self, image_array: np.ndarray) -> (np.ndarray, list):
        """High-quality prediction using MTCNN for images and videos."""
        annotated_image, predictions = image_array.copy(), []
        face_results = self.hq_face_detector.detect_faces(image_array)
        if not face_results: return annotated_image, predictions

        for face in face_results:
            if face['confidence'] < 0.95: continue
            x, y, w, h = face['box']
            face_img = image_array[max(0,y):min(image_array.shape[0],y+h), max(0,x):min(image_array.shape[1],x+w)]
            if face_img.size == 0: continue
            pil_face = Image.fromarray(face_img)
            pixel_values = self.transforms(pil_face).unsqueeze(0).to(self.device)
            with torch.no_grad(): outputs = self.model(pixel_values=pixel_values)
            pred_id_age = str(outputs['age_logits'].argmax(1).item())
            pred_id_gender = str(outputs['gender_logits'].argmax(1).item())
            age_label = self.label_maps['age_id2label'].get(pred_id_age, "N/A")
            gender_label = self.label_maps['gender_id2label'].get(pred_id_gender, "N/A")
            prediction_labels = {"age": age_label, "gender": gender_label}
            predictions.append({**prediction_labels, 'box': (x, y, w, h)})
            self._draw_predictions(annotated_image, (x, y, w, h), prediction_labels)
        return annotated_image, predictions

    def predict_lq(self, image_array: np.ndarray) -> (np.ndarray, list):
        """Lightweight prediction using Haar Cascade for live feed."""
        annotated_image, predictions = image_array.copy(), []
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        faces = self.lq_face_detector.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0: return annotated_image, predictions

        for (x, y, w, h) in faces:
            face_img = image_array[y:y+h, x:x+w]
            if face_img.size == 0: continue
            pil_face = Image.fromarray(face_img)
            pixel_values = self.transforms(pil_face).unsqueeze(0).to(self.device)
            with torch.no_grad(): outputs = self.model(pixel_values=pixel_values)
            pred_id_age = str(outputs['age_logits'].argmax(1).item())
            pred_id_gender = str(outputs['gender_logits'].argmax(1).item())
            age_label = self.label_maps['age_id2label'].get(pred_id_age, "N/A")
            gender_label = self.label_maps['gender_id2label'].get(pred_id_gender, "N/A")
            prediction_labels = {"age": age_label, "gender": gender_label}
            predictions.append({**prediction_labels, 'box': (x, y, w, h)})
            self._draw_predictions(annotated_image, (x, y, w, h), prediction_labels)
        return annotated_image, predictions