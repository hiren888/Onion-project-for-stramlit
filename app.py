import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
import gc
from inference_sdk import InferenceHTTPClient

# --- CONFIGURATION ---
# Industry standard onion grades (mm)
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
    "Colossal": (255, 0, 255), 
    "Oversized": (255, 0, 0)   
}

@st.cache_resource
def load_model():
    """Initializes the Roboflow Workflow Client."""
    # It is safer to use st.secrets for GitHub. 
    # If testing locally without secrets.toml, replace with your string.
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
    
    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        return client, None
    except Exception as e:
        return None, str(e)

def correct_orientation(image):
    """Corrects image orientation based on EXIF data."""
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

def process_onions_workflow(client, image_bgr, ppm, conf_threshold):
    """Detects and measures onions using Roboflow Workflows."""
    try:
        # Run Workflow - Removed 'use_cache' to prevent SDK version errors
        result = client.run_workflow(
            workspace_name="onion-project",
            workflow_id="find-onions",
            images={"image": image_bgr}
        )
        
        # Handle list vs dict response structure
        workflow_output = result[0] if isinstance(result, list) else result
        
        # Access the 'outputs' dictionary safely
        output_data = workflow_output.get('outputs', {})
        
        # NOTE: If your workflow output block is not named 'predictions', 
        # change the key below to match your Roboflow Workflow naming.
        predictions = output_data.get('predictions', []) 

        processed_image = image_bgr.copy()
        onion_data = []
        
        for i, p in enumerate(predictions):
            if p.get('confidence', 0) < conf_threshold:
                continue
                
            w_px, h_px = p['width'], p['height']
            x_c, y_c = p['x'], p['y']
            
            # Calculate Physical Diameter (using Major Axis)
            diameter_mm = max(w_px, h_px) / ppm
            grade = determine_grade(diameter_mm)
            color = GRADE_COLORS.get(grade, (255, 255, 255))
            
            # Draw on Image
            x1, y1 = int(x_c - w_px/2), int(y_c - h_px/2)
            x2, y2 = int(x_c + w_px/2), int(y_c + h_px/2)
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(processed_image, f"{grade} {diameter_mm:.1f}mm", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            onion_data.append({
                "ID": i + 1,
                "Diameter (mm)": round(diameter_mm, 2),
                "Grade": grade,
                "Confidence": round(float(p['confidence']), 2)
            })
            
        return processed_image, onion_data
    except Exception as e:
        st.error(f"Workflow Processing Error: {e}")
        # Show raw result in case of error to help user debug names
        with st.expander("Show Raw API Response"):
            st.write(result)
        return image_bgr, []

def main():
    st.set_page_config(page_title="AgriGrade AI", layout="wide")
    st.title("🧅 AgriGrade AI: Onion Size Distribution")
    
    # Sidebar Configuration
    st.sidebar.title("⚙️ Calibration & AI")
    # Reference: 1080p photo at 1 meter often has a ppm around 3.5-5.0
    ppm = st.sidebar.number_input("Pixels per mm (Scale)", 0.1, 100.0, 5.0, 
                                  help="Calibrate this by dividing an object's pixel width by its real mm width.")
    conf_thresh = st.sidebar.slider("AI Confidence Threshold", 0.1, 0.9, 0.4)
    
    client, err = load_model()
    if err:
        st.error(f"Connection Error: {err}")
        st.stop()

    uploaded_file = st.file_uploader("Upload Onion Stock Photo", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        image_pil = correct_orientation(image_pil)
        img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_pil, caption="Original Image", use_container_width=True)
            
        with col2:
            with st.spinner("Analyzing onions via Roboflow Cloud..."):
                processed_img, data = process_onions_workflow(client, img_bgr, ppm, conf_thresh)
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), 
                         caption=f"Analysis Complete: {len(data)} detected", use_container_width=True)

        if data:
            df = pd.DataFrame(data)
            st.divider()
            
            # KPI Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Sample Count", len(df))
            m2.metric("Avg Diameter", f"{df['Diameter (mm)'].mean():.1f} mm")
            m3.metric("Dominant Grade", df['Grade'].mode()[0])
            
            # Data View
            st.subheader("📋 Detailed Measurements")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📩 Download Data (CSV)", csv, "onion_distribution.csv")
        else:
            st.warning("No onions detected. Check your 'Confidence' setting or your Workflow Output names.")

    gc.collect()

if __name__ == "__main__":
    main()
