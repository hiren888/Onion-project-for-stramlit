import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
from roboflow import Roboflow
import gc

# --- NEW GRADING STANDARDS ---
# Based on your request: >65mm, 55-60mm, and <55mm
GRADE_STANDARDS = {
    "Grade A (>65mm)": (65.0, 1000.0),
    "Grade B (55-60mm)": (55.0, 65.0), # Adjusted to cover the gap up to 65
    "Grade C (<55mm)": (0.0, 55.0)
}

GRADE_COLORS = {
    "Grade A (>65mm)": (0, 255, 0),    # Green
    "Grade B (55-60mm)": (0, 165, 255), # Orange
    "Grade C (<55mm)": (0, 0, 255)      # Red
}

@st.cache_resource
def load_model():
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("onion-project").project("onion-tydja")
        model = project.version(9).model
        return model, None
    except Exception as e:
        return None, str(e)

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3: image = image.rotate(180, expand=True)
            elif orientation == 6: image = image.rotate(270, expand=True)
            elif orientation == 8: image = image.rotate(90, expand=True)
    except:
        pass
    return image

def determine_grade(diameter_mm):
    for grade, (min_d, max_d) in GRADE_STANDARDS.items():
        if min_d <= diameter_mm < max_d:
            return grade
    return "Unknown"

def process_onions(model, image_bgr, manual_ppm, conf_threshold, ref_size_mm):
    try:
        cv2.imwrite("temp_inference.jpg", image_bgr)
        response = model.predict("temp_inference.jpg", confidence=conf_threshold).json()
        predictions = response.get('predictions', [])

        # --- 1. DYNAMIC CALIBRATION ---
        calculated_ppm = None
        for p in predictions:
            # Flexible check for the Reference class
            c_name = p['class'].lower()
            if c_name == 'reference' or c_name == 'referance':
                pixel_size = max(p['width'], p['height'])
                calculated_ppm = pixel_size / ref_size_mm
                break 

        final_ppm = calculated_ppm if calculated_ppm else manual_ppm
        
        if calculated_ppm:
            st.success(f"🎯 Auto-Calibrated Scale: {final_ppm:.2f} px/mm")
        else:
            st.warning("⚠️ Reference object not detected. Using fallback scale.")

        # --- 2. PROCESSING ---
        processed_image = image_bgr.copy()
        onion_data = []

        for i, p in enumerate(predictions):
            x_c, y_c, w_px, h_px = p['x'], p['y'], p['width'], p['height']
            x1, y1 = int(x_c - w_px/2), int(y_c - h_px/2)
            x2, y2 = int(x_c + w_px/2), int(y_c + h_px/2)

            if p['class'].lower() in ['reference', 'referance']:
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(processed_image, "REF", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                continue

            diameter_mm = max(w_px, h_px) / final_ppm
            grade = determine_grade(diameter_mm)
            color = GRADE_COLORS.get(grade, (255, 255, 255))
            
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(processed
