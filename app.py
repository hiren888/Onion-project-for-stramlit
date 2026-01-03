import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd

# --- CONFIGURATION ---
# USDA Onion Grading Standards (Diameter in mm)
# Source: USDA Agricultural Marketing Service 
GRADE_STANDARDS = {
    "Small": (0, 50.8),      # < 2 inches
    "Medium": (50.8, 76.2),  # 2 to 3 inches
    "Large": (76.2, 95.0),   # 3 to 3.75 inches
    "Colossal": (95.0, 1000) # > 3.75 inches
}

# ArUco Configuration
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
# Default Marker Size (User can override in UI)
DEFAULT_MARKER_SIZE_MM = 50.0 

@st.cache_resource
def load_model():
    """
    Loads the YOLOv8 segmentation model.
    The 'yolov8n-seg.pt' is the Nano model, optimized for CPU speed.
    """
    return YOLO('yolov8n-seg.pt')

def detect_aruco_and_get_ppm(image_bgr, marker_size_mm):
    """
    Detects ArUco marker to calculate Pixels-Per-Metric (PPM) ratio.
    Uses sub-pixel corner refinement for high-precision metrology.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
    parameters = cv2.aruco.DetectorParameters()
    # Critical for accuracy: Sub-pixel refinement
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    
    # Updated for OpenCV 4.8+: Use ArucoDetector class
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(image_bgr)
    
    if ids is None:
        return None, None, image_bgr
        
    # Visualization: Draw the detected marker
    cv2.aruco.drawDetectedMarkers(image_bgr, corners, ids)
    
    # Use the first detected marker to calculate scale
    # Corners structure: [top-left, top-right, bottom-right, bottom-left]
    # We take the first marker found (index 0)
    c = corners
    
    # Calculate Euclidean distance of the top edge and left edge in pixels
    width_px = np.linalg.norm(c - c[1])
    height_px = np.linalg.norm(c - c[2])
    
    # Average the sides to account for slight perspective tilt
    avg_size_px = (width_px + height_px) / 2.0
    
    # Calculate PPM (Pixels / mm)
    ppm = avg_size_px / marker_size_mm
    return ppm, ids, image_bgr

def determine_grade(diameter_mm):
    """Classifies diameter against USDA standards."""
    for grade, (min_d, max_d) in GRADE_STANDARDS.items():
        if min_d <= diameter_mm < max_d:
            return grade
    return "Oversized"

def process_onions_yolo(model, image_bgr, ppm, conf_threshold):
    """
    Uses YOLOv8-seg to detect onions and measure them using the PPM ratio.
    """
    # 1. Run Inference
    # Note: 'classes' argument can filter specific COCO classes (e.g., 46=banana, 49=orange)
    # Ideally, use a custom model trained specifically on Onions.
    # Here we accept all detections for demonstration or assume custom weights.
    results = model(image_bgr, conf=conf_threshold) 
    
    processed_image = image_bgr.copy()
    onion_data =  # Initialize empty list to avoid SyntaxError
    
    # Access the first result object
    # model() returns a list of Results objects, we take the first one
    if not results:
        # Return empty list if no results found to match unpacking in main()
        return processed_image,

    result = results
    
    if result.masks is None:
        # Return empty list if no masks found to match unpacking in main()
        return processed_image,

    # 2. Iterate over detected instances
    for i, mask_data in enumerate(result.masks.data):
        # result.masks.xy gives coordinates of the mask contour
        polygon = result.masks.xy[i].astype(np.int32)
        
        # Calculate Area using Contour
        area_px = cv2.contourArea(polygon)
        
        # 3. Biometric Sizing: Equivalent Circle Diameter (ECD)
        # This is robust against the "flaky skin" irregular shapes
        diameter_px = 2 * np.sqrt(area_px / np.pi)
        
        # Convert to Millimeters
        diameter_mm = diameter_px / ppm
        grade = determine_grade(diameter_mm)
        
        # 4. Visualization
        # Draw the accurate polygon contour
        # cv2.polylines expects a list of points
        cv2.polylines(processed_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Find center for labeling
        M = cv2.moments(polygon)
        if M["m00"]!= 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            # Fallback if moment is zero (rare)
            if len(polygon) > 0:
                cX, cY = polygon, polygon[1]
            else:
                continue # Skip if polygon is invalid
            
        # Label with Grade and Size
        label = f"{grade}\n{diameter_mm:.1f}mm"
        # Draw background for text readability
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(processed_image, (cX - 10, cY - 20), (cX + w, cY), (0,0,0), -1)
        cv2.putText(processed_image, f"{diameter_mm:.1f}mm", (cX, cY - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        onion_data.append({
            "ID": i+1,
            "Diameter (mm)": round(diameter_mm, 2),
            "Grade": grade,
            "Confidence": float(result.boxes.conf[i])
        })
        
    return processed_image, onion_data

# --- MAIN APP LOGIC ---
def main():
    st.set_page_config(page_title="AgriGrade AI: Onion Analytics", layout="wide")
    
    st.title("🧅 AgriGrade AI: Precision Onion Grading")
    st.markdown("""
    **System Status:** Active | **Engine:** YOLOv8-seg + ArUco Metrology
    
    This system replaces legacy watershed segmentation with Deep Learning Instance Segmentation 
    to resolve texture noise issues. It requires a **4x4 ArUco Marker** for absolute sizing.
    """)
    
    # Sidebar Configuration
    st.sidebar.header("Calibration Settings")
    marker_size = st.sidebar.number_input("ArUco Marker Size (mm)", value=DEFAULT_MARKER_SIZE_MM)
    conf_threshold = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.25, 
                                       help="Lower value detects more objects but may increase false positives.")
    
    # File Uploader
    uploaded_file = st.file_uploader("Upload Grading Batch (Top-Down View)", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        # Load and Decode Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        
        # Layout: Two Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Input & Calibration")
            # Display original (convert BGR to RGB for Streamlit)
            st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption="Raw Input", use_container_width=True)
            
            # Perform Calibration
            with st.status("Calibrating Geometry...", expanded=True) as status:
                ppm, ids, debug_img = detect_aruco_and_get_ppm(img_bgr.copy(), marker_size)
                
                if ppm:
                    st.write(f"✅ **Marker Detected:** ID {ids.flatten()}")
                    st.write(f"📏 **Scale Factor:** {ppm:.2f} pixels/mm")
                    status.update(label="Calibration Complete", state="complete")
                else:
                    st.error("❌ No ArUco marker detected.")
                    st.warning("System will default to pixel measurements (uncalibrated).")
                    ppm = 1.0 # Fallback to prevent crash
                    status.update(label="Calibration Failed", state="error")
        
        with col2:
            st.subheader("2. AI Segmentation & Grading")
            
            if ppm == 1.0:
                st.warning("⚠️ Displaying results in Pixels (Uncalibrated)")
            
            with st.spinner("Running YOLOv8 Inference..."):
                model = load_model()
                processed_img, data = process_onions_yolo(model, img_bgr.copy(), ppm, conf_threshold)
                
                # Display Result
                st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), 
                         caption="Segmented & Graded Output", use_container_width=True)
        
        # Data Reporting Section
        st.divider()
        st.subheader("3. Batch Analytics")
        
        if data:
            df = pd.DataFrame(data)
            
            # Summary Metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Count", len(df))
            m2.metric("Average Diameter", f"{df.mean():.1f} mm")
            m3.metric("Min Diameter", f"{df.min():.1f} mm")
            m4.metric("Max Diameter", f"{df.max():.1f} mm")
            
            # Grade Distribution Table
            grade_counts = df['Grade'].value_counts().reset_index()
            grade_counts.columns = ['Grade', 'Count']
            
            c1, c2 = st.columns(2)
            with c1:
                st.write("#### Grade Distribution")
                st.dataframe(grade_counts, hide_index=True, use_container_width=True)
            with c2:
                st.write("#### Detailed Log")
                st.dataframe(df, use_container_width=True)
                
            # CSV Download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Batch Report", csv, "onion_grading.csv", "text/csv")
            
        else:
            st.info("No onions detected. Adjust confidence threshold or check image lighting.")

if __name__ == "__main__":
    main()
