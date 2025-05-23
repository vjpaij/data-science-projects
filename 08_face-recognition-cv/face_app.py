import streamlit as st
import cv2
import numpy as np
from PIL import Image
from recognize import FaceRecognizer
import tempfile
import os

st.title("Real-Time Face Recognition System")

# Sidebar controls
st.sidebar.header("Settings")
detection_model = st.sidebar.selectbox("Detection Model", ["hog", "cnn"], index=0)
tolerance = st.sidebar.slider("Tolerance", 0.1, 1.0, 0.6, 0.05)
process_every = st.sidebar.slider("Process Every N Frames", 1, 10, 3)

# Initialize recognizer
@st.cache_resource
def get_recognizer():
    return FaceRecognizer(detection_model=detection_model, tolerance=tolerance, process_every=process_every)

recognizer = get_recognizer()

# Display options
option = st.radio("Select Input Type:", ("Webcam", "Upload Video", "Upload Image"))

if option == "Webcam":
    st.write("Starting webcam...")
    run_webcam = st.checkbox("Run Webcam")
    
    if run_webcam:
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        
        while run_webcam:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video")
                break
                
            # Convert to RGB for Streamlit
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb)
            
            # Process frame
            boxes, names = recognizer.recognize_faces(frame)
            
            # Draw boxes and names
            for (top, right, bottom, left), name in zip(boxes, names):
                cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(rgb, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            FRAME_WINDOW.image(rgb)
            
        cap.release()

elif option == "Upload Video":
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_path = tfile.name
        st.video(video_path)
        cap = cv2.VideoCapture(video_path)
        FRAME_WINDOW = st.image([])
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB for Streamlit
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(rgb)
            
            # Process frame
            boxes, names = recognizer.recognize_faces(frame)
            
            # Draw boxes and names
            for (top, right, bottom, left), name in zip(boxes, names):
                cv2.rectangle(rgb, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(rgb, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
            
            FRAME_WINDOW.image(rgb)
        cap.release()
        os.remove(video_path)
    else:
        st.warning("Please upload a video file.")
elif option == "Upload Image":
    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Convert to OpenCV format
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Process image
        boxes, names = recognizer.recognize_faces(image_cv)
        
        # Draw boxes and names
        for (top, right, bottom, left), name in zip(boxes, names):
            cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(image_cv, name, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        st.image(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)
    else:
        st.warning("Please upload an image file.")
