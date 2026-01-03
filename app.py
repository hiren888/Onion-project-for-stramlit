import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
import io
import gc

# --- OPTIONAL DEPENDENCIES ---
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# --- CONFIGURATION ---
GRADE_STANDARDS = {
    "Small": (0, 50.8),
    "Medium": (50.8, 76.2),
    "Large": (76.2, 95.0),
    "Colossal": (95.0, 1000)
}

GRADE_COLORS = {
    "Small": (255, 200, 0),    # Yellow-ish
    "Medium": (0, 255, 150),   # Green-ish
    "Large": (0, 150, 255),    # Blue-ish
    "Colossal": (255, 0, 255), # Magenta
    "Oversized": (255, 0, 0)   # Red
}

DEFAULT_MARKER_SIZE_MM = 50.0

def check_dependencies():
    """Check and display dependency status."""
    missing = []
    if not YOLO_AVAILABLE:
        missing.append("ultralytics")
    if not PLOTLY_AVAILABLE:
        missing.append("plotly")
    return missing

@st.cache_resource
def load_model():
    """Loads YOLOv8 segmentation model."""
    if not YOLO_AVAILABLE:
        return None, "ultralytics package not installed"
    
    try:
        # Downloads the model if not present locally
        model = YOLO('yolov8n-seg.pt')
        return model, None
    except Exception as e:
        return None, str(e)

def correct_orientation(image):
    """Corrects image orientation based on EXIF data (fixes mobile upload rotation)."""
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif = image._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def detect_aruco_and_get_ppm(image_bgr, marker_size_mm, dict_type):
    """
    Enhanced ArUco detection with multiple strategies.
    Returns: (ppm, marker_ids, annotated_image, corners, debug_info)
    """
    debug_info = {"tried_methods": [], "preprocessing": [], "success": False}
    
    def try_detection(img, method_name, preprocess_func=None):
        try:
            if preprocess_func:
                img_processed = preprocess_func(img)
            else:
                img_processed = img
            
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
            parameters = cv2.aruco.DetectorParameters()
            
            # Robust parameters
            parameters.adaptiveThreshWinSizeMin = 3
            parameters.adaptiveThreshWinSizeMax = 23
            parameters.adaptiveThreshWinSizeStep = 10
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, rejected = detector.detectMarkers(img_processed)
            
            debug_info["tried_methods"].append({
                "method": method_name,
                "found": ids is not None,
                "count": len(ids) if ids is not None else 0
            })
            
            return corners, ids, rejected
            
        except Exception as e:
            debug_info["tried_methods"].append({"method": method_name, "error": str(e)})
            return None, None, None
    
    # 1. Original
    corners, ids, _ = try_detection(image_bgr, "Original")
    
    # 2. Grayscale
    if ids is None:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = try_detection(gray, "Grayscale", lambda x: x)
    
    # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    if ids is None:
        def enhance_contrast(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            return clahe.apply(gray)
        corners, ids, _ = try_detection(image_bgr, "CLAHE", enhance_contrast)
    
    # 4. Adaptive Thresholding
    if ids is None:
        def adaptive_thresh(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        corners, ids, _ = try_detection(image_bgr, "Adaptive Thresh", adaptive_thresh)

    # Calculation
    if ids is not None and len(corners) > 0:
        annotated = image_bgr.copy()
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        
        # Use the first found marker for calibration
        c = corners[0][0]
        width_px = np.linalg.norm(c[0] - c[1])
        height_px = np.linalg.norm(c[0] - c[3])
        avg_size_px = (width_px + height_px) / 2.0
        
        ppm = avg_size_px / marker_size_mm
        
        debug_info["success"] = True
        debug_info["ppm"] = ppm
        return ppm, ids, annotated, corners, debug_info

    return None, None, image_bgr, None, debug_info

def determine_grade(diameter_mm):
    for grade, (min_d, max_d) in GRADE_STANDARDS.items():
        if min_d <= diameter_mm < max_d:
            return grade
    return "Oversized"

def process_onions_yolo(model, image_bgr, ppm, conf_threshold, iou_threshold, measure_mode="Equivalent Diameter"):
    """Detects and measures onions."""
    try:
        results = model(image_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
        
        if not results or results[0].masks is None:
            return image_bgr, []
        
        result = results[0]
        processed_image = image_bgr.copy()
        onion_data = []
        
        for i, mask_data in enumerate(result.masks.data):
            # Convert mask to polygon
            polygon = result.masks.xy[i].astype(np.int32)
            if len(polygon) == 0: continue
            
            # Metrics
            area_px = cv2.contourArea(polygon)
            if area_px < 100: continue # Filter tiny noise
            
            # Fit Ellipse
            if len(polygon) >= 5:
                try:
                    ellipse = cv2.fitEllipse(polygon)
                    (_, axes, _) = ellipse
                    major_px = max(axes)
                    minor_px = min(axes)
                    eccentricity = np.sqrt(1 - (minor_px**2 / major_px**2))
                except:
                    major_px = minor_px = np.sqrt(4 * area_px / np.pi)
                    eccentricity = 0.0
            else:
                 major_px = minor_px = np.sqrt(4 * area_px / np.pi)
                 eccentricity = 0.0

            # Convert to mm
            major_mm = major_px / ppm
            minor_mm = minor_px / ppm
            area_mm2 = area_px / (ppm ** 2)
            
            # Determine diameter based on selected mode
            if measure_mode == "Major Axis":
                diameter_mm = major_mm
            elif measure_mode == "Minor Axis":
                diameter_mm = minor_mm
            else: # Equivalent Diameter
                diameter_mm = 2 * np.sqrt(area_mm2 / np.pi)

            grade = determine_grade(diameter_mm)
            color = GRADE_COLORS.get(grade, (255, 255, 255))
            
            # Draw
            cv2.polylines(processed_image, [polygon], True, color, 2)
            
            # Centroid
            M = cv2.moments(polygon)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                cX, cY = polygon[0][0], polygon[0][1]
            
            # Label
            label_text = f"{grade}"
            cv2.putText(processed_image, label_text, (cX - 20, cY), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(processed_image, label_text, (cX - 20, cY), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            onion_data.append({
                "ID": i + 1,
                "Diameter (mm)": round(diameter_mm, 2),
                "Area (mm²)": round(area_mm2, 2),
                "Grade": grade,
                "Eccentricity": round(eccentricity, 3),
                "Confidence": round(float(result.boxes.conf[i]), 2)
            })
            
        return processed_image, onion_data

    except Exception as e:
        st.error(f"Processing Error: {e}")
        return image_bgr, []

def main():
    st.set_page_config(page_title="AgriGrade AI", layout="wide")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Calibration Settings
    calib_mode = st.sidebar.radio("Calibration Mode", ["Auto (ArUco)", "Manual Scale"])
    
    ppm_manual = 0.0
    aruco_dict_type = cv2.aruco.DICT_4X4_50 # Default
    marker_size = DEFAULT_MARKER_SIZE_MM

    if calib_mode == "Auto (ArUco)":
        marker_size = st.sidebar.number_input("Marker Size (mm)", 10.0, 200.0, 50.0)
        dict_name = st.sidebar.selectbox("Dict Type", ["4x4_50", "4x4_100", "5x5_50"])
        if dict_name == "4x4_100": aruco_dict_type = cv2.aruco.DICT_4X4_100
        elif dict_name == "5x5_50": aruco_dict_type = cv2.aruco.DICT_5X5_50
    else:
        ppm_manual = st.sidebar.number_input("Manual Scale (Pixels per mm)", 0.1, 100.0, 5.0)

    st.sidebar.divider()
    
    # AI Settings
    conf_thresh = st.sidebar.slider("AI Confidence", 0.1, 0.9, 0.25)
    measure_mode = st.sidebar.selectbox("Measurement Basis", 
                                      ["Equivalent Diameter", "Major Axis", "Minor Axis"],
                                      help="Equivalent: Based on area. Major/Minor: Based on ellipse fit.")
    
    # Main Content
    st.title("🧅 AgriGrade AI: Onion Grading")
    
    missing = check_dependencies()
    if missing:
        st.error(f"Missing libraries: {', '.join(missing)}")
        st.stop()

    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        # 1. Load Image using PIL to handle rotation correctly
        image_pil = Image.open(uploaded_file)
        image_pil = correct_orientation(image_pil)
        image_np = np.array(image_pil)
        
        # Convert RGB (PIL) to BGR (OpenCV)
        img_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        col1, col2 = st.columns(2)
        
        # 2. Calibration
        ppm = 0.0
        with col1:
            st.image(image_pil, caption="Original Image", use_container_width=True)
            
            if calib_mode == "Manual Scale":
                ppm = ppm_manual
                st.info(f"Using Manual Scale: {ppm:.2f} px/mm")
            else:
                with st.spinner("Calibrating..."):
                    ppm_calc, ids, debug_img, _, info = detect_aruco_and_get_ppm(
                        img_bgr.copy(), marker_size, aruco_dict_type
                    )
                    
                    if ppm_calc:
                        ppm = ppm_calc
                        st.success(f"Calibration Successful! ({ppm:.2f} px/mm)")
                        with st.expander("Debug Info"):
                            st.write(info)
                    else:
                        st.error("Calibration Failed: Marker not found.")
                        st.warning("Switch to 'Manual Scale' in sidebar if marker is unreadable.")
                        with st.expander("Troubleshooting"):
                            st.write(info["tried_methods"])
        
        # 3. Processing
        if ppm > 0:
            model, err = load_model()
            if model:
                with col2:
                    with st.spinner("Analyzing onions..."):
                        processed_img, data = process_onions_yolo(
                            model, img_bgr, ppm, conf_thresh, 0.45, measure_mode
                        )
                        
                        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), 
                                caption=f"Processed ({len(data)} onions)", 
                                use_container_width=True)
                
                # 4. Results
                if data:
                    df = pd.DataFrame(data)
                    st.divider()
                    st.subheader("📊 Grading Report")
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Count", len(df))
                    m2.metric("Avg Diameter", f"{df['Diameter (mm)'].mean():.1f} mm")
                    m3.metric("Top Grade", df['Grade'].mode()[0] if not df.empty else "N/A")
                    m4.metric("Avg Quality (Conf)", f"{df['Confidence'].mean():.0%}")

                    # Charts
                    if PLOTLY_AVAILABLE:
                        tab1, tab2 = st.tabs(["Distributions", "Data"])
                        with tab1:
                            fig = px.histogram(df, x="Diameter (mm)", color="Grade", nbins=20,
                                             color_discrete_map={k: f"rgb{v}" for k,v in GRADE_COLORS.items()})
                            st.plotly_chart(fig, use_container_width=True)
                        with tab2:
                            st.dataframe(df, use_container_width=True)
                    else:
                        st.dataframe(df, use_container_width=True)
                    
                    # CSV Download
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Data (CSV)", csv, "onion_data.csv", "text/csv")
            else:
                st.error(f"Could not load AI model: {err}")
        
        # Cleanup
        gc.collect()

if __name__ == "__main__":
    main()
