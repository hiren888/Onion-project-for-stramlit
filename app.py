import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
import gc
from roboflow import Roboflow

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
        project = rf.workspace("onion-project").project("onion-project-slug")
        model = project.version(1).model # Ensure version matches your training
        return model, None
    except Exception as e:
        return None, str(e)

def determine_grade(diameter_mm):
    for grade, (min_d, max_d) in GRADE_STANDARDS.items():
        if min_d <= diameter_mm < max_d:
            return grade
    return "Oversized"

def process_onions(model, image_bgr, ppm, conf_threshold):
    """Detects and measures onions using the standard Hosted API."""
    try:
        # Save temp image for the API to read
        cv2.imwrite("temp.jpg", image_bgr)
        
        # Run inference
        response = model.predict("temp.jpg", confidence=conf_threshold).json()
        predictions = response.get('predictions', [])

        processed_image = image_bgr.copy()
        onion_data = []

        for i, p in enumerate(predictions):
            # Roboflow returns x, y (center), width, and height in pixels
            w_px, h_px = p['width'], p['height']
            x_c, y_c = p['x'], p['y']
            
            # Diameter Calculation (Standardized to Major Axis)
            diameter_mm = max(w_px, h_px) / ppm
            grade = determine_grade(diameter_mm)
            color = GRADE_COLORS.get(grade, (255, 255, 255))
            
            # Draw Bounding Box
            x1, y1 = int(x_c - w_px/2), int(y_c - h_px/2)
            x2, y2 = int(x_c + w_px/2), int(y_c + h_px/2)
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
    st.title("🧅 Onion Size Distribution App")
    
    # Sidebar
    st.sidebar.header("Settings")
    ppm = st.sidebar.number_input("Pixels per mm (Calibration)", 0.1, 50.0, 5.0)
    conf = st.sidebar.slider("AI Confidence", 0.1, 0.9, 0.4)
    
    model, err = load_model()
    if err:
        st.error(f"API Connection Failed: {err}")
        st.stop()

    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing Stock..."):
            processed_img, data = process_onions(model, img_bgr, ppm, conf*100)
            
            col1, col2 = st.columns(2)
            col1.image(img, caption="Original", use_container_width=True)
            col2.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Graded", use_container_width=True)

        if data:
            df = pd.DataFrame(data)
            st.success(f"Detected {len(df)} onions.")
            
            # Analytics
            st.subheader("📊 Distribution Data")
            st.dataframe(df, use_container_width=True)
            
            # Summary Metrics
            c1, c2 = st.columns(2)
            c1.metric("Average Diameter", f"{df['Diameter (mm)'].mean():.1f} mm")
            c2.metric("Top Grade", df['Grade'].mode()[0])
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV Report", csv, "onion_report.csv", "text/csv")

    gc.collect()

if __name__ == "__main__":
    main()
