import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
import cv2
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
except ImportError as e:
    print(f"Could not import custom modules: {e}.")
    sys.exit(1)

class PredictionPipeline:
    def __init__(self, model_path: str = "artifacts/multi_task_model_trainer/checkpoint-26873"):
        self.device = "cpu"
        self.model_path = Path(model_path)
        params = read_yaml(Path("params.yaml"))
        
        self.label_maps = {
            'age_id2label': {'0': '0-2', '1': '3-9', '2': '10-19', '3': '20-29', '4': '30-39', '5': '40-49', '6': '50-59', '7': '60-69', '8': 'more than 70'},
            'gender_id2label': {'0': 'Male', '1': 'Female'}
        }

        # Use lightweight Haar Cascade
        haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(haar_cascade_path)
        
        print(f"--- Pipeline Initialized Successfully on device: {self.device} ---")
        self.processor = AutoImageProcessor.from_pretrained("google/efficientnet-b2")
        self.transforms = Compose([Resize((params.IMAGE_SIZE, params.IMAGE_SIZE)), ToTensor(), Normalize(mean=self.processor.image_mean, std=self.processor.image_std)])
        self.model = self._load_model()
        
        haar_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(haar_cascade_path)
        
        print(f"--- Pipeline Initialized Successfully on device: {self.device} ---")
    
    def _load_model(self):
        num_age, num_gender, num_race = len(self.label_maps['age_id2label']), len(self.label_maps['gender_id2label']), 7
        
        # Load the base architecture from the pre-downloaded cache
        model = MultiTaskEfficientNet("google/efficientnet-b2", num_age, num_gender, num_race)
        
        weight_file = self.model_path / 'model.safetensors'
        if not weight_file.exists(): weight_file = self.model_path / 'pytorch_model.bin'
        if not weight_file.exists(): raise FileNotFoundError(f"Weights not found in {self.model_path}")
        
        state_dict = load_safetensors(weight_file, device="cpu") if weight_file.suffix == ".safetensors" else torch.load(weight_file, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    # --- THE CORRECTED PREDICT METHOD IS HERE ---
    def predict(self, image_array: np.ndarray) -> (np.ndarray, list):
        annotated_image = image_array.copy()
        predictions = []
        
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray_image, 
            scaleFactor=1.2, 
            minNeighbors=8, 
            minSize=(60, 60)
        )
        
        if len(faces) == 0: 
            return annotated_image, predictions

        for (x, y, w, h) in faces:
            face_img = image_array[y:y+h, x:x+w]
            if face_img.size == 0: continue

            pil_face = Image.fromarray(face_img)
            pixel_values = self.transforms(pil_face).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(pixel_values=pixel_values)

            pred_id_age = str(outputs['age_logits'].argmax(1).item())
            pred_id_gender = str(outputs['gender_logits'].argmax(1).item())
            age_label = self.label_maps['age_id2label'].get(pred_id_age, "N/A")
            gender_label = self.label_maps['gender_id2label'].get(pred_id_gender, "N/A")

            predictions.append({"box": (x, y, w, h), "age": age_label, "gender": gender_label})

            # --- DRAWING LOGIC WITH THE FIX ---
            font, font_scale, font_thickness = cv2.FONT_HERSHEY_DUPLEX, 0.6, 1
            text_color, bg_color = (255, 255, 255), (255, 75, 75)
            text_lines = [f"Gender: {gender_label}", f"Age: {age_label}"]
            
            # 1. Calculate the maximum width required for any line of text
            max_width = 0
            for line in text_lines:
                (w_text, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                if w_text > max_width: 
                    max_width = w_text

            line_height = 25
            total_height = len(text_lines) * line_height - 5
            
            # 2. Draw the bounding box for the face
            cv2.rectangle(annotated_image, (x, y), (x + w, y + h), bg_color, 2)
            
            # 3. Use the calculated `max_width` to draw a background that is always big enough
            cv2.rectangle(annotated_image, (x - 1, y - total_height), (x + max_width + 10, y), bg_color, -1)
            
            # 4. Draw the text on top of the background
            for i, line in enumerate(text_lines):
                y_text = y - total_height + (i * line_height) + 18
                cv2.putText(annotated_image, line, (x + 5, y_text), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
            # --- END DRAWING LOGIC ---
            
        return annotated_image, predictions