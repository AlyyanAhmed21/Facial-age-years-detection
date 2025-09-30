import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor
import cv2
from mtcnn import MTCNN
from pathlib import Path
import sys
import os
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from safetensors.torch import load_file as load_safetensors
from collections import OrderedDict
from scipy.spatial import distance as dist

try:
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if src_path not in sys.path: sys.path.append(src_path)
    from components.multi_task_model_trainer import MultiTaskEfficientNet
    from utils.common import read_yaml
except ImportError as e:
    print(f"Could not import custom modules: {e}.")
    sys.exit(1)

class CentroidTracker:
    def __init__(self, max_disappeared=20):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

    def register(self, centroid, box):
        self.objects[self.next_object_id] = {'centroid': centroid, 'box': box, 'labels': {}, 'ema_preds': {}}
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, boxes):
        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([(x + w // 2, y + h // 2) for (x, y, w, h) in boxes])

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], boxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([v['centroid'] for v in self.objects.values()])
            D = dist.cdist(object_centroids, input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['box'] = boxes[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(D.shape[0])).difference(used_rows)
            unused_cols = set(range(D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col], boxes[col])
        return self.objects

class PredictionPipeline:
    def __init__(self, model_path: str = "artifacts/multi_task_model_trainer/checkpoint-26873"):
        self.device = "cpu" 
        self.model_path = Path(model_path)
        self.base_model_name = "google/efficientnet-b2"
        params = read_yaml(Path("params.yaml"))
        self.processor = AutoImageProcessor.from_pretrained(self.base_model_name)
        self.transforms = Compose([Resize((params.IMAGE_SIZE, params.IMAGE_SIZE)), ToTensor(), Normalize(mean=self.processor.image_mean, std=self.processor.image_std)])
        self.label_maps = self._load_label_maps()
        self.model = self._load_model()
        self.face_detector = MTCNN()
        self.tracker = CentroidTracker()
        print(f"--- Pipeline Initialized on device: {self.device} ---")

    def _load_label_maps(self):
        maps = {'age_id2label': {'0': '0-2', '1': '3-9', '2': '10-19', '3': '20-29', '4': '30-39', '5': '40-49', '6': '50-59', '7': '60-69', '8': 'more than 70'},
                'gender_id2label': {'0': 'Male', '1': 'Female'}}
        return maps
    
    def _load_model(self):
        num_age, num_gender, num_race = len(self.label_maps['age_id2label']), len(self.label_maps['gender_id2label']), 7
        model = MultiTaskEfficientNet(self.base_model_name, num_age, num_gender, num_race)
        weight_file = self.model_path / 'model.safetensors'
        if not weight_file.exists(): weight_file = self.model_path / 'pytorch_model.bin'
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
    
    def _predict_for_box(self, frame, box):
        x, y, w, h = [int(c) for c in box]
        face_img = frame[max(0,y):min(frame.shape[0],y+h), max(0,x):min(frame.shape[1],x+w)]
        if face_img.size == 0: return None
        pixel_values = self.transforms(Image.fromarray(face_img)).unsqueeze(0).to(self.device)
        with torch.no_grad(): outputs = self.model(pixel_values=pixel_values)
        return outputs
    
    def predict_image(self, image_array):
        annotated_image, predictions = image_array.copy(), []
        face_results = self.face_detector.detect_faces(image_array)
        if not face_results: return annotated_image, predictions
        for face in face_results:
            if face['confidence'] < 0.9: continue
            box = face['box']
            outputs = self._predict_for_box(annotated_image, box)
            if outputs:
                age_label = self.label_maps['age_id2label'][str(outputs['age_logits'].argmax(1).item())]
                gender_label = self.label_maps['gender_id2label'][str(outputs['gender_logits'].argmax(1).item())]
                prediction_labels = {"age": age_label, "gender": gender_label}
                predictions.append({**prediction_labels, 'box': box})
                self._draw_predictions(annotated_image, box, prediction_labels)
        return annotated_image, predictions

    def process_video_stream(self, frame_generator):
        self.tracker = CentroidTracker()
        for frame in frame_generator:
            face_results = self.face_detector.detect_faces(frame)
            boxes = [tuple(face['box']) for face in face_results if face['confidence'] > 0.9]
            tracked_objects = self.tracker.update(boxes)
            
            for obj_id, data in tracked_objects.items():
                # Predict only for new tracks or tracks that have just been re-found
                if 'labels' not in data or self.tracker.disappeared[obj_id] == 0:
                    outputs = self._predict_for_box(frame, data['box'])
                    if outputs:
                        alpha = 0.3
                        current_probs = {
                            'age': outputs['age_logits'].softmax(1).cpu().numpy()[0],
                            'gender': outputs['gender_logits'].softmax(1).cpu().numpy()[0]
                        }
                        # Apply EMA smoothing
                        if not data.get('ema_preds'): data['ema_preds'] = current_probs
                        else:
                            for task in ['age', 'gender']:
                                data['ema_preds'][task] = alpha * current_probs[task] + (1 - alpha) * data['ema_preds'][task]
                
                # Always update the label from the latest smoothed probabilities
                if data.get('ema_preds'):
                    age_label = self.label_maps['age_id2label'][str(np.argmax(data['ema_preds']['age']))]
                    gender_label = self.label_maps['gender_id2label'][str(np.argmax(data['ema_preds']['gender']))]
                    data['labels'] = {"age": age_label, "gender": gender_label}
            
            annotated_frame = frame.copy()
            for obj_id, data in tracked_objects.items():
                if 'labels' in data:
                    self._draw_predictions(annotated_frame, data['box'], data['labels'])
            yield annotated_frame
            
    def process_live_frame(self, frame):
        annotated_frame, _ = self.predict_image(frame)
        return annotated_frame