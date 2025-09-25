import streamlit as st
import cv2
import numpy as np
from PIL import Image
from transformers import pipeline
from mtcnn import MTCNN
from collections import defaultdict

st.set_page_config(layout="wide", page_title="Facial Age Detection")

st.title("Facial Age Detection")
st.write("Detect age groups from images, videos, or a live webcam feed.")
st.write("This application uses an EfficientFormer-L1 model fine-tuned on the Facial Age dataset.")

# --- Helper Functions and Classes ---

@st.cache_resource
def load_model():
    """Load the age detection model pipeline."""
    model_path = "artifacts/model_trainer/facial_age_detector_model"
    pipe = pipeline('image-classification', model=model_path, device=0) # Use 0 for GPU
    return pipe

@st.cache_resource
def load_face_detector():
    """Load the MTCNN face detector."""
    return MTCNN()

def iou(boxA, boxB):
    """Calculate Intersection over Union."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

class EMATracker:
    """Exponential Moving Average Tracker for smoothing predictions."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.tracked_objects = {} # {track_id: {box: [], ema_preds: {}}}

    def update(self, detections, id_counter):
        # Detections are a list of face boxes
        # Simple tracking by IOU
        
        # Match detections to existing tracks
        matches = {} # {track_id: det_idx}
        used_det_indices = set()
        
        # This is a simple greedy matching. For more robust tracking, consider Hungarian algorithm.
        for track_id, data in self.tracked_objects.items():
            best_iou = 0
            best_det_idx = -1
            for i, det_box in enumerate(detections):
                if i in used_det_indices: continue
                current_iou = iou(data['box'], det_box)
                if current_iou > best_iou and current_iou > 0.3: # IOU threshold
                    best_iou = current_iou
                    best_det_idx = i
            if best_det_idx != -1:
                matches[track_id] = best_det_idx
                used_det_indices.add(best_det_idx)

        # Update matched tracks
        for track_id, det_idx in matches.items():
            self.tracked_objects[track_id]['box'] = detections[det_idx]
            
        # Add new tracks
        for i, det_box in enumerate(detections):
            if i not in used_det_indices:
                self.tracked_objects[id_counter] = {'box': det_box, 'ema_preds': defaultdict(float)}
                id_counter += 1
        
        # Remove old tracks (optional, for long videos)
        
        return id_counter

    def apply_ema(self, track_id, new_preds):
        """Applies EMA to the predictions for a given track."""
        if track_id not in self.tracked_objects:
            return {}
        
        current_ema = self.tracked_objects[track_id]['ema_preds']
        
        # Initialize if new
        if not current_ema:
            for pred in new_preds:
                current_ema[pred['label']] = pred['score']
        else:
            # Update existing values
            for pred in new_preds:
                label = pred['label']
                current_ema[label] = (self.alpha * pred['score']) + ((1 - self.alpha) * current_ema[label])
        
        self.tracked_objects[track_id]['ema_preds'] = current_ema
        
        # Return the top prediction from EMA
        if not current_ema: return None
        top_label = max(current_ema, key=current_ema.get)
        return f"{top_label} ({current_ema[top_label]:.2f})"


# --- Load Models ---
try:
    age_pipe = load_model()
    face_detector = load_face_detector()
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure the model is trained and located at 'artifacts/model_trainer/facial_age_detector_model'.")
    st.stop()


# --- UI Sidebar ---
st.sidebar.header("Input Options")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Image", "Video", "Live Webcam"])

# --- Main App Logic ---

if app_mode == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        img_array = np.array(image)
        
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting faces and predicting age...")
        
        faces = face_detector.detect_faces(img_array)
        
        if not faces:
            st.warning("No faces detected in the image.")
        else:
            for face in faces:
                x, y, w, h = face['box']
                face_img = img_array[y:y+h, x:x+w]
                pil_face = Image.fromarray(face_img)
                
                # Predict age
                age_preds = age_pipe(pil_face)
                top_pred = age_preds[0]
                
                # Draw on image
                cv2.rectangle(img_array, (x, y), (x+w, y+h), (0, 255, 0), 2)
                label = f"Age: {top_pred['label']} ({top_pred['score']:.2f})"
                cv2.putText(img_array, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
            st.image(img_array, caption='Processed Image.', use_column_width=True)

elif app_mode == "Live Webcam":
    st.sidebar.info("Webcam feed will start automatically. Press 'Stop' to end.")
    run = st.sidebar.button('Start Webcam')
    stop = st.sidebar.button('Stop Webcam')
    FRAME_WINDOW = st.image([])
    
    cap = cv2.VideoCapture(0)
    tracker = EMATracker()
    track_id_counter = 0

    while run and not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image from webcam.")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detector.detect_faces(frame_rgb)
        
        detection_boxes = [f['box'] for f in faces]
        track_id_counter = tracker.update(detection_boxes, track_id_counter)
        
        for track_id, data in tracker.tracked_objects.items():
            x, y, w, h = data['box']
            if w > 20 and h > 20: # Filter small detections
                face_img = frame_rgb[y:y+h, x:x+w]
                pil_face = Image.fromarray(face_img)
                
                age_preds = age_pipe(pil_face)
                smoothed_label = tracker.apply_ema(track_id, age_preds)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                if smoothed_label:
                    cv2.putText(frame, smoothed_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    st.sidebar.info("Webcam stopped.")

# Add a placeholder for Video processing, which would be similar to Webcam but with a file uploader.
elif app_mode == "Video":
    st.sidebar.warning("Video processing is similar to the webcam feed but processes a file. This feature is not fully implemented in this demo but follows the same logic.")
    # You would use cv2.VideoCapture(video_path) and loop through frames.