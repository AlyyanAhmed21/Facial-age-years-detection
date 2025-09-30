import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import tempfile
import time
from streamlit_option_menu import option_menu

# --- Page Config ---
st.set_page_config(page_title="Facial Analysis", page_icon="👤", layout="wide", initial_sidebar_state="expanded")

# --- Path Setup & Model Loading ---
try:
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
    if src_path not in sys.path: sys.path.append(src_path)
    from cnnClassifier.pipeline.prediction import PredictionPipeline
except ImportError:
    st.error("FATAL: Prediction pipeline not found. Please check your project structure.")
    st.stop()

@st.cache_resource
def load_pipeline():
    return PredictionPipeline()

pipeline = load_pipeline()

# --- Session State for Webcam Control ---
if 'webcam_running' not in st.session_state: st.session_state.webcam_running = False
def start_webcam(): st.session_state.webcam_running = True
def stop_webcam(): st.session_state.webcam_running = False

# --- UI ---
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    app_mode = option_menu(None, ["Image", "Video", "Live Feed"], 
        icons=['image', 'film', 'camera-video'], menu_icon="cast", default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#FF4B4B", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#FF4B4B"},
        })

if not pipeline:
    st.error("AI Pipeline failed to load. Check the terminal logs for errors.")
else:
    st.title(f"👤 Facial Demographics Analysis")
    st.header(f"Mode: {app_mode}")
    st.divider()

    if app_mode == "Image":
        uploaded_file = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            col1, col2 = st.columns(2)
            with col1: st.image(image, caption='Original Image', use_column_width=True)
            with col2:
                with st.spinner('🔬 Analyzing...'):
                    annotated_image, predictions = pipeline.predict(np.array(image))
                st.image(annotated_image, caption='Processed Image', use_column_width=True)
                if predictions:
                    with st.expander("View Details", expanded=True):
                        for i, p in enumerate(predictions):
                            st.write(f"**Face {i+1}:** Gender: `{p['gender']}`, Age Group: `{p['age']}`")
                else: st.warning("No faces detected.")

    elif app_mode == "Video":
        uploaded_file = st.file_uploader("Upload a video for analysis", type=["mp4", "mov", "avi"])
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.info(f"Video has {frame_count} frames. Processing will be slow on this server.")
            if st.button("Process Video", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Initializing...")
                out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                out = cv2.VideoWriter(out_tfile.name, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (w, h))
                for i in range(frame_count):
                    ret, frame = cap.read()
                    if not ret: break
                    annotated_frame_rgb, _ = pipeline.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    out.write(cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR))
                    progress_bar.progress((i + 1) / frame_count, text=f"Processing Frame {i+1}/{frame_count}")
                cap.release(), out.release()
                st.success("Video processing complete!")
                st.video(out_tfile.name)
                with open(out_tfile.name, "rb") as f:
                    st.download_button("Download Processed Video", f, "output.mp4", "video/mp4", use_container_width=True)

    elif app_mode == "Live Feed":
        col1, col2 = st.columns(2)
        with col1: st.button("Start Feed", on_click=start_webcam, use_container_width=True, type="primary")
        with col2: st.button("Stop Feed", on_click=stop_webcam, use_container_width=True)
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            FRAME_WINDOW = st.image([])
            fps_display = st.empty()
        if st.session_state.webcam_running:
            cap = cv2.VideoCapture(0)
            while st.session_state.webcam_running:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                annotated_frame, _ = pipeline.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                FRAME_WINDOW.image(annotated_frame, channels="RGB")
                fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                fps_display.markdown(f"<p style='text-align: center;'><b>FPS: {fps:.2f}</b></p>", unsafe_allow_html=True)
            cap.release()
            cv2.destroyAllWindows()
            st.session_state.webcam_running = False
            st.rerun()