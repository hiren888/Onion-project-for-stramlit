import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import gc
from inference_sdk import InferenceHTTPClient

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
    """Initializes the Roboflow Workflow Client."""
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
    try:
        client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        return client, None
    except Exception as e:
        return None, str(e)

def determine_grade(diameter_mm):
    for grade, (min_d, max_d) in GRADE_STANDARDS.items():
        if min_d <= diameter_mm < max_d:
            return grade
    return "Oversized"

def process_onions_workflow(client, image_bgr, ppm, conf_threshold):
    """Parses Workflow response by indexing the list first."""
    try:
        # 1. Call the Workflow
        # Replace these with your actual IDs from the 'Rapid' project URL
        result = client.run_workflow(
            workspace_name="onion-project",
            workflow_id="find-onions",
            images={"image": image_bgr}
        )

        # 2. Fix the 'List' vs 'Dict' structure
        # Workflows return a list: [{'outputs': {...}}]
        if isinstance(result, list):
            raw_output = result[0]
        else:
            raw_output = result

        # 3. Drill down to the predictions
        # We look for 'predictions' inside 'outputs'
        outputs = raw_output.get('outputs', {})
        
        # This checks the most common keys Workflows use for detection blocks
        predictions = outputs.get('predictions') or outputs.get('detections') or []

        processed_image = image_bgr.copy()
        onion_data = []

        for i, p in enumerate(predictions):
            conf = p.get('confidence', 0)
            if conf < conf_threshold:
                continue
            
            # Extract box dimensions
            w_px, h_px = p.get('width', 0), p.get('height', 0)
            x_c, y_c = p.get('x', 0), p.get('y', 0)
            
            # Diameter Math
            diameter_mm = max(w_px, h_px) / ppm
            grade = determine_grade(diameter_mm)
            color = GRADE_COLORS.get(grade, (255, 255, 255))
            
            # Draw on Image
            x1, y1 = int(x_c - w_px/2), int(y_c - h_px/2)
            x2, y2 = int(x_c + w_px/2), int(y_c + h_px/2)
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(processed_image, f"{diameter_mm:.1f}mm", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            onion_data.append({
                "ID": i + 1,
                "Diameter (mm)": round(diameter_mm, 2),
                "Grade": grade,
                "Confidence": round(float(conf), 2)
            })
            
        return processed_image, onion_data

    except Exception as e:
        st.error(f"Workflow Logic Error: {e}")
        # This will show you exactly what the API sent back so we can fix it
        with st.expander("See Raw System Response"):
            st.write(result)
        return image_bgr, []

def main():
    st.set_page_config(page_title="AgriGrade AI", layout="wide")
    st.title("🧅 Onion Grading System (Roboflow Rapid)")
    
    st.sidebar.header("Settings")
    ppm = st.sidebar.number_input("Scale (Pixels per mm)", 0.1, 100.0, 5.0)
    conf = st.sidebar.slider("AI Confidence", 0.1, 0.9, 0.4)
    
    client, err = load_model()
    if err:
        st.error(f"API Connection Error: {err}")
        st.stop()

    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        img_pil = Image.open(uploaded_file)
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing Stock..."):
            processed_img, data = process_onions_workflow(client, img_bgr, ppm, conf)
            
            col1, col2 = st.columns(2)
            col1.image(img_pil, caption="Original", use_container_width=True)
            col2.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption="Graded Results", use_container_width=True)

        if data:
            df = pd.DataFrame(data)
            st.success(f"Detected {len(df)} onions.")
            st.subheader("📊 Distribution Analysis")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Report (CSV)", csv, "onion_report.csv")
        else:
            st.info("No onions detected. Check 'See Raw System Response' expander for details.")

    gc.collect()

if __name__ == "__main__":
    main()
