import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
from roboflow import Roboflow
import gc

# --- CONFIGURATION ---
GRADE_STANDARDS = {
    "Small": (0, 50.8),
    "Medium": (50.8, 76.2),
    "Large": (76.2, 95.0),
    "Colossal": (95.0, 1000)
}

GRADE_COLORS = {
    "Small": (255, 200, 0),    
    "Medium": (0, 255, 150),   
    "Large": (0, 150, 255),    
    "Colossal": (255, 0, 255)
}

@st.cache_resource
def load_model():
    """Initializes the Roboflow API Client."""
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
    try:
        rf = Roboflow(api_key=api_key)
        # Update version to 4 after you train with the 'reference' class
        project = rf.workspace("onion-project").project("onion-tydja")
        model = project.version(4).model 
        return model, None
    except Exception as e:
        return None, str(e)

def correct_orientation(image):
    """Fixes mobile upload rotation issues."""
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
    return "Oversized"

def process_onions(model, image_bgr, manual_ppm, conf_threshold, ref_size_mm):
    """Detects onions and uses a reference object to calibrate size."""
    try:
        cv2.imwrite("temp_inference.jpg", image_bgr)
        response = model.predict("temp_inference.jpg", confidence=conf_threshold).json()
        predictions = response.get('predictions', [])

        # --- 1. DYNAMIC CALIBRATION ---
        calculated_ppm = None
        for p in predictions:
            if p['class'].lower() == 'reference':
                pixel_size = max(p['width'], p['height'])
                calculated_ppm = pixel_size / ref_size_mm
                break 

        final_ppm = calculated_ppm if calculated_ppm else manual_ppm
        
        if calculated_ppm:
            st.success(f"🎯 Auto-Calibrated Scale: {final_ppm:.2f} px/mm")
        else:
            st.info("ℹ️ No reference object detected. Using manual scale.")

        # --- 2. PROCESSING ---
        processed_image = image_bgr.copy()
        onion_data = []

        for i, p in enumerate(predictions):
            x_c, y_c, w_px, h_px = p['x'], p['y'], p['width'], p['height']
            x1, y1 = int(x_c - w_px/2), int(y_c - h_px/2)
            x2, y2 = int(x_c + w_px/2), int(y_c + h_px/2)

            if p['class'].lower() == 'reference':
                # Draw the reference object in Cyan
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(processed_image, "REF", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                continue

            # Calculate Diameter for Onions
            diameter_mm = max(w_px, h_px) / final_ppm
            grade = determine_grade(diameter_mm)
            color = GRADE_COLORS.get(grade, (255, 255, 255))
            
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(processed_image, f"{diameter_mm:.1f}mm", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            onion_data.append({
                "ID": i + 1,
                "Diameter (mm)": round(diameter_mm, 2),
                "Grade": grade,
                "Confidence": round(p['confidence'], 2)
            })
            
        return processed_image, onion_data
    except Exception as e:
        st.error(f"Inference Error: {e}")
        return image_bgr, []

def main():
    st.set_page_config(page_title="AgriGrade AI", layout="wide")
    st.title("🧅 Onion Grading System (Pro Mode)")
    
    st.sidebar.header("⚙️ Configuration")
    
    # NEW: Reference Object Size Setting
    ref_type = st.sidebar.selectbox("Reference Object", ["25mm Coin", "85mm Card", "Custom"])
    if ref_type == "25mm Coin": ref_size_mm = 25.0
    elif ref_type == "85mm Card": ref_size_mm = 85.6
    else: ref_size_mm = st.sidebar.number_input("Custom Size (mm)", 1.0, 500.0, 50.0)

    # Fallback manual scale if no reference is found
    ppm = st.sidebar.number_input("Fallback Manual Scale (px/mm)", 0.1, 100.0, 5.0)
    conf = st.sidebar.slider("AI Confidence %", 10, 100, 40)
    
    model, err = load_model()
    if err:
        st.error(f"API Error: {err}")
        st.stop()

    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        image_pil = correct_orientation(image_pil)
        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing produce..."):
            processed_img, data = process_onions(model, img_bgr, ppm, conf, ref_size_mm)
            
            col1, col2 = st.columns(2)
            col1.image(image_pil, caption="Original Photo", use_container_width=True)
            col2.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), 
                       caption=f"Results ({len(data)} detected)", use_container_width=True)

        if data:
            df = pd.DataFrame(data)
            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Count", len(df))
            m2.metric("Avg Diameter", f"{df['Diameter (mm)'].mean():.1f} mm")
            m3.metric("Primary Grade", df['Grade'].mode()[0])
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Report", csv, "onion_report.csv")

    gc.collect()

if __name__ == "__main__":
    main()
