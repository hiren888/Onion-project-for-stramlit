import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
from roboflow import Roboflow
import gc

# --- CONFIGURATION ---
GRADE_STANDARDS = {
    "Grade A (>65mm)": (65.0, 1000.0),
    "Grade B (55-65mm)": (55.0, 65.0), 
    "Grade C (<55mm)": (0.0, 55.0)
}

GRADE_COLORS = {
    "Grade A (>65mm)": (0, 255, 0),     # Green
    "Grade B (55-65mm)": (255, 165, 0), # Orange
    "Grade C (<55mm)": (0, 0, 255)      # Red
}

if 'master_data' not in st.session_state:
    st.session_state['master_data'] = pd.DataFrame(columns=["Batch ID", "Diameter (mm)", "Grade"])
if 'batch_counter' not in st.session_state:
    st.session_state['batch_counter'] = 0

@st.cache_resource
def load_model():
    # Attempt to load secret; default to empty if not found (will error later safely)
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "")
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("onion-project").project("onion-tydja")
        model = project.version(11).model
        return model, None
    except Exception as e:
        return None, str(e)

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation': break
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3: image = image.rotate(180, expand=True)
            elif orientation == 6: image = image.rotate(270, expand=True)
            elif orientation == 8: image = image.rotate(90, expand=True)
    except: pass
    return image

def determine_grade(diameter_mm):
    for grade, (min_d, max_d) in GRADE_STANDARDS.items():
        if min_d <= diameter_mm < max_d: return grade
    return "Unknown"

def process_onions(model, image_bgr, manual_ppm, conf_threshold, ref_size_mm, camera_height_cm, ignore_edges):
    try:
        img_h, img_w = image_bgr.shape[:2]
        edge_margin = 25 # Slightly increased margin for safety

        cv2.imwrite("temp_inference.jpg", image_bgr)
        # Low confidence to catch reference
        response = model.predict("temp_inference.jpg", confidence=10).json()
        raw_predictions = response.get('predictions', [])

        # --- CALIBRATION ---
        calculated_ppm = None
        reference_prediction = None
        target_names = ['reference', 'ref', 'coin', 'card', 'marker']
        
        for p in raw_predictions:
            if any(name in p['class'].lower() for name in target_names):
                pixel_size = max(p['width'], p['height'])
                calculated_ppm = pixel_size / ref_size_mm
                reference_prediction = p
                break 

        final_ppm = calculated_ppm if calculated_ppm else manual_ppm
        
        # --- PROCESSING ---
        processed_image = image_bgr.copy()
        current_batch_data = []

        # Draw Reference Box
        if reference_prediction:
            p = reference_prediction
            x1, y1 = int(p['x'] - p['width']/2), int(p['y'] - p['height']/2)
            x2, y2 = int(p['x'] + p['width']/2), int(p['y'] + p['height']/2)
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (255, 255, 0), 3)
            cv2.putText(processed_image, "REF", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        user_conf_decimal = conf_threshold / 100.0
        cam_h_mm = camera_height_cm * 10.0

        for p in raw_predictions:
            if p == reference_prediction: continue
            if p['confidence'] < user_conf_decimal: continue

            x_c, y_c, w_px, h_px = p['x'], p['y'], p['width'], p['height']
            x1, y1 = int(x_c - w_px/2), int(y_c - h_px/2)
            x2, y2 = int(x_c + w_px/2), int(y_c + h_px/2)

            # Edge Filter
            if ignore_edges:
                if x1 < edge_margin or y1 < edge_margin or x2 > (img_w - edge_margin) or y2 > (img_h - edge_margin):
                    continue 

            # --- GEOMETRIC SIZE CORRECTION ---
            # 1. Raw Size (Floor level)
            raw_diameter_mm = max(w_px, h_px) / final_ppm
            
            # 2. Correction
            # Estimate radius (height of the onion surface from floor)
            estimated_radius = raw_diameter_mm / 2.0
            
            # Factor = (Camera_Height - Onion_Height) / Camera_Height
            if cam_h_mm > 0:
                correction_factor = (cam_h_mm - estimated_radius) / cam_h_mm
            else:
                correction_factor = 1.0
            
            final_diameter_mm = raw_diameter_mm * correction_factor
            
            grade = determine_grade(final_diameter_mm)
            color = GRADE_COLORS.get(grade, (255, 255, 255))
            
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(processed_image, f"{final_diameter_mm:.1f}mm", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            current_batch_data.append({"Diameter (mm)": round(final_diameter_mm, 2), "Grade": grade})
            
        return processed_image, current_batch_data, final_ppm
    except Exception as e:
        st.error(f"Error: {e}")
        return image_bgr, [], manual_ppm

def main():
    st.set_page_config(page_title="AgriGrade Procurement", layout="wide")
    st.title("🧅 Onion Procurement System")
    
    # --- SIDEBAR ---
    st.sidebar.header("⚙️ Configuration")
    ref_type = st.sidebar.selectbox("Reference Object", ["25mm Coin", "85mm Card", "Custom"])
    ref_size_mm = 25.0 if ref_type == "25mm Coin" else 85.6 if ref_type == "85mm Card" else st.sidebar.number_input("Custom (mm)", 1.0, 500.0, 50.0)

    st.sidebar.divider()
    st.sidebar.subheader("📐 Capture Conditions")
    
    # NEW: Simplified Stance Selector
    stance = st.sidebar.radio("Photographer Stance", ["Standing (Standard)", "Sitting/Table", "Custom"], index=0)
    
    if stance == "Standing (Standard)":
        camera_height_cm = 120  # Safe average for standing
        st.sidebar.caption("Using default height: 120cm")
    elif stance == "Sitting/Table":
        camera_height_cm = 60   # Safe average for sitting at a table
        st.sidebar.caption("Using default height: 60cm")
    else:
        camera_height_cm = st.sidebar.number_input("Custom Height (cm)", 30, 200, 120)

    ignore_edges = st.sidebar.checkbox("Ignore Edge Onions", value=True, help="Removes partial onions at borders.")
    conf = st.sidebar.slider("AI Confidence %", 10, 100, 40)
    manual_ppm = st.sidebar.number_input("Fallback Scale", 0.1, 100.0, 5.0)

    if st.sidebar.button("🗑️ Clear Data"):
        st.session_state['master_data'] = pd.DataFrame(columns=["Batch ID", "Diameter (mm)", "Grade"])
        st.session_state['batch_counter'] = 0
        st.rerun()

    model, err = load_model()
    if err: 
        st.error(f"API Error: {err}. Please check your API Key in Secrets.")
        st.stop()

    # --- MAIN ---
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Detect & Verify")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            img = Image.open(uploaded_file)
            img = correct_orientation(img)
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            processed, data, ppm = process_onions(model, img_bgr, manual_ppm, conf, ref_size_mm, camera_height_cm, ignore_edges)
            
            st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), caption=f"Analyzed (Stance: {stance} | H: {camera_height_cm}cm)", use_container_width=True)
            
            if data and st.button("✅ Add to Report"):
                batch = pd.DataFrame(data)
                batch["Batch ID"] = st.session_state['batch_counter'] + 1
                st.session_state['master_data'] = pd.concat([st.session_state['master_data'], batch], ignore_index=True)
                st.session_state['batch_counter'] += 1
                st.success("Batch Added!"); st.rerun()

    with col2:
        st.subheader("2. Cumulative Report")
        df = st.session_state['master_data']
        if not df.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("Count", len(df))
            m2.metric("Batches", st.session_state['batch_counter'])
            m3.metric("Avg Size", f"{df['Diameter (mm)'].mean():.1f} mm")
            
            st.markdown("### 📊 Grade Breakdown")
            counts = df['Grade'].value_counts()
            for g in GRADE_STANDARDS: 
                if g not in counts: counts[g] = 0
            counts = counts.reindex(list(GRADE_STANDARDS.keys()))
            
            summary = [{"Grade": g, "Count": c, "%": f"{(c/len(df)*100):.1f}%"} for g, c in counts.items()]
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
            st.bar_chart(counts)
            
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "onion_report.csv", "text/csv", type="primary")

    gc.collect()

if __name__ == "__main__":
    main()
