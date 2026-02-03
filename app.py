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

# --- SESSION STATE INITIALIZATION ---
if 'master_data' not in st.session_state:
    st.session_state['master_data'] = pd.DataFrame(columns=["Batch ID", "Diameter (mm)", "Grade"])
if 'batch_counter' not in st.session_state:
    st.session_state['batch_counter'] = 0

@st.cache_resource
def load_model():
    api_key = st.secrets.get("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
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

        # Calibration
        calculated_ppm = None
        for p in predictions:
            if 'refer' in p['class'].lower():
                pixel_size = max(p['width'], p['height'])
                calculated_ppm = pixel_size / ref_size_mm
                break 
        final_ppm = calculated_ppm if calculated_ppm else manual_ppm
        
        processed_image = image_bgr.copy()
        current_batch_data = []

        for p in predictions:
            if 'refer' in p['class'].lower():
                continue

            x_c, y_c, w_px, h_px = p['x'], p['y'], p['width'], p['height']
            x1, y1 = int(x_c - w_px/2), int(y_c - h_px/2)
            x2, y2 = int(x_c + w_px/2), int(y_c + h_px/2)

            diameter_mm = max(w_px, h_px) / final_ppm
            grade = determine_grade(diameter_mm)
            color = GRADE_COLORS.get(grade, (255, 255, 255))
            
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 4)
            cv2.putText(processed_image, f"{diameter_mm:.1f}mm", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            current_batch_data.append({
                "Diameter (mm)": round(diameter_mm, 2),
                "Grade": grade
            })
            
        return processed_image, current_batch_data, final_ppm
    except Exception as e:
        return image_bgr, [], manual_ppm

def main():
    st.set_page_config(page_title="AgriGrade Procurement", layout="wide")
    st.title("🧅 Onion Procurement System")
    
    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("⚙️ Configuration")
    ref_type = st.sidebar.selectbox("Reference Object", ["25mm Coin", "85mm Card", "Custom"])
    if ref_type == "25mm Coin": ref_size_mm = 25.0
    elif ref_type == "85mm Card": ref_size_mm = 85.6
    else: ref_size_mm = st.sidebar.number_input("Custom Size (mm)", 1.0, 500.0, 50.0)

    manual_ppm = st.sidebar.number_input("Fallback Scale (px/mm)", 0.1, 100.0, 5.0)
    conf = st.sidebar.slider("AI Confidence %", 10, 100, 40)
    
    # Add Reset Button to Sidebar
    if st.sidebar.button("🗑️ Clear All Data"):
        st.session_state['master_data'] = pd.DataFrame(columns=["Batch ID", "Diameter (mm)", "Grade"])
        st.session_state['batch_counter'] = 0
        st.rerun()

    model, err = load_model()
    if err:
        st.error(f"API Error: {err}")
        st.stop()

    # --- MAIN INTERFACE ---
    col_upload, col_stats = st.columns([1, 1])
    
    with col_upload:
        st.subheader("1. Detect & Verify")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image_pil = Image.open(uploaded_file)
            image_pil = correct_orientation(image_pil)
            img_bgr = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            
            # Process Image
            processed_img, current_data, used_ppm = process_onions(model, img_bgr, manual_ppm, conf, ref_size_mm)
            
            st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), caption=f"Analyzed (Scale: {used_ppm:.2f})", use_container_width=True)
            
            # "Add to Batch" Logic
            if current_data:
                st.info(f"Detected {len(current_data)} onions in this image.")
                if st.button("✅ Add these to Report"):
                    # Create DF for current batch
                    batch_df = pd.DataFrame(current_data)
                    batch_df["Batch ID"] = st.session_state['batch_counter'] + 1
                    
                    # Append to Master DF
                    st.session_state['master_data'] = pd.concat([st.session_state['master_data'], batch_df], ignore_index=True)
                    st.session_state['batch_counter'] += 1
                    st.success(f"Added Batch #{st.session_state['batch_counter']} to report!")
                    st.rerun()

    with col_stats:
        st.subheader("2. Cumulative Report")
        master_df = st.session_state['master_data']
        
        if not master_df.empty:
            total_onions = len(master_df)
            
            # Top Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Onions", total_onions)
            m2.metric("Batches Processed", st.session_state['batch_counter'])
            m3.metric("Avg Diameter", f"{master_df['Diameter (mm)'].mean():.1f} mm")
            
            st.divider()
            
            # --- PERCENTAGE CALCULATION ---
            st.markdown("### 📊 Grade Breakdown")
            
            # Get counts for all grades (including 0 if missing)
            grade_counts = master_df['Grade'].value_counts()
            for g in GRADE_STANDARDS.keys():
                if g not in grade_counts: grade_counts[g] = 0
            
            # Sort grades
            grade_counts = grade_counts.reindex(list(GRADE_STANDARDS.keys()))
            
            # Create a nice summary table with Percentages
            summary_data = []
            for grade, count in grade_counts.items():
                percentage = (count / total_onions) * 100 if total_onions > 0 else 0
                summary_data.append({
                    "Grade": grade,
                    "Count": count,
                    "Percentage": f"{percentage:.1f}%"
                })
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
            
            # Visualization
            st.bar_chart(grade_counts)

            # --- DOWNLOAD BUTTON ---
            csv = master_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cumulative Report (CSV)",
                data=csv,
                file_name="onion_procurement_report.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.info("No data yet. Upload an image and click 'Add to Report' to build your stats.")

    gc.collect()

if __name__ == "__main__":
    main()
