import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

# Try importing optional dependencies
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
    "Small": (255, 200, 0),
    "Medium": (0, 255, 150),
    "Large": (0, 150, 255),
    "Colossal": (255, 0, 255),
    "Oversized": (255, 0, 0)
}

ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
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
        model = YOLO('yolov8n-seg.pt')
        return model, None
    except Exception as e:
        return None, str(e)

def detect_aruco_and_get_ppm(image_bgr, marker_size_mm):
    """Detects ArUco marker and calculates PPM ratio."""
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(image_bgr)
        
        if ids is None or len(corners) == 0:
            return None, None, image_bgr, None
        
        annotated = image_bgr.copy()
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        
        c = corners[0][0]
        width_px = np.linalg.norm(c[0] - c[1])
        height_px = np.linalg.norm(c[0] - c[3])
        avg_size_px = (width_px + height_px) / 2.0
        ppm = avg_size_px / marker_size_mm
        
        return ppm, ids, annotated, corners
        
    except Exception as e:
        st.error(f"ArUco detection error: {e}")
        return None, None, image_bgr, None

def determine_grade(diameter_mm):
    """Classifies diameter against USDA standards."""
    for grade, (min_d, max_d) in GRADE_STANDARDS.items():
        if min_d <= diameter_mm < max_d:
            return grade
    return "Oversized"

def calculate_ellipse_metrics(contour):
    """Calculates shape metrics."""
    area = cv2.contourArea(contour)
    ecd = 2 * np.sqrt(area / np.pi)
    
    if len(contour) >= 5:
        try:
            ellipse = cv2.fitEllipse(contour)
            (_, axes, _) = ellipse
            major_axis = max(axes)
            minor_axis = min(axes)
            eccentricity = np.sqrt(1 - (minor_axis**2 / major_axis**2))
            return ecd, major_axis, minor_axis, eccentricity
        except:
            return ecd, ecd, ecd, 0.0
    
    return ecd, ecd, ecd, 0.0

def process_onions_yolo(model, image_bgr, ppm, conf_threshold, show_advanced=False):
    """Detects and measures onions using YOLOv8."""
    try:
        results = model(image_bgr, conf=conf_threshold, verbose=False)
        
        if not results or results[0].masks is None:
            return image_bgr, []
        
        result = results[0]
        processed_image = image_bgr.copy()
        onion_data = []
        
        for i, mask_data in enumerate(result.masks.data):
            polygon = result.masks.xy[i].astype(np.int32)
            
            if len(polygon) == 0:
                continue
            
            area_px = cv2.contourArea(polygon)
            ecd_px, major_px, minor_px, eccent = calculate_ellipse_metrics(polygon)
            
            diameter_mm = ecd_px / ppm
            major_mm = major_px / ppm
            minor_mm = minor_px / ppm
            area_mm2 = area_px / (ppm ** 2)
            
            grade = determine_grade(diameter_mm)
            color = GRADE_COLORS.get(grade, (255, 255, 255))
            
            cv2.polylines(processed_image, [polygon], True, color, 3)
            
            M = cv2.moments(polygon)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = polygon[0][0], polygon[0][1]
            
            if show_advanced:
                label = f"{grade}\n{diameter_mm:.1f}mm\ne={eccent:.2f}"
            else:
                label = f"{grade}\n{diameter_mm:.1f}mm"
            
            y_offset = 0
            for line in label.split('\n'):
                (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(processed_image, 
                            (cX - 5, cY - 25 - y_offset), 
                            (cX + w + 5, cY - y_offset), 
                            (0, 0, 0), -1)
                cv2.putText(processed_image, line, 
                          (cX, cY - 10 - y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += h + 5
            
            cv2.circle(processed_image, (cX, cY), 5, color, -1)
            
            onion_data.append({
                "ID": i + 1,
                "Diameter (mm)": round(diameter_mm, 2),
                "Area (mm²)": round(area_mm2, 2),
                "Major Axis (mm)": round(major_mm, 2),
                "Minor Axis (mm)": round(minor_mm, 2),
                "Eccentricity": round(eccent, 3),
                "Grade": grade,
                "Confidence": round(float(result.boxes.conf[i]), 3)
            })
        
        return processed_image, onion_data
        
    except Exception as e:
        st.error(f"Processing error: {e}")
        return image_bgr, []

def create_visualizations(df):
    """Creates interactive Plotly visualizations."""
    if not PLOTLY_AVAILABLE:
        return None, None, None
    
    grade_counts = df['Grade'].value_counts()
    fig_pie = px.pie(
        values=grade_counts.values,
        names=grade_counts.index,
        title="Grade Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig_hist = px.histogram(
        df,
        x="Diameter (mm)",
        color="Grade",
        nbins=20,
        title="Diameter Distribution by Grade",
        labels={"Diameter (mm)": "Diameter (mm)", "count": "Frequency"}
    )
    
    fig_scatter = px.scatter(
        df,
        x="Diameter (mm)",
        y="Eccentricity",
        color="Grade",
        size="Confidence",
        hover_data=["ID", "Area (mm²)"],
        title="Shape Analysis: Diameter vs Roundness"
    )
    
    return fig_pie, fig_hist, fig_scatter

def show_dependency_warning(missing_deps):
    """Display installation instructions for missing dependencies."""
    st.error("⚠️ **Missing Required Dependencies**")
    
    st.markdown(f"""
    The following packages are not installed: **{', '.join(missing_deps)}**
    
    ### 📦 Installation Instructions
    
    **Option 1: Using pip**
    ```bash
    pip install ultralytics plotly opencv-python
    ```
    
    **Option 2: Using requirements.txt**
    
    Create a `requirements.txt` file with:
    ```
    streamlit
    opencv-python-headless
    numpy
    pandas
    pillow
    ultralytics
    plotly
    ```
    
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
    
    **For Streamlit Cloud:**
    
    1. Create a `requirements.txt` file in your repository root
    2. Add the packages listed above
    3. Commit and push to GitHub
    4. Streamlit Cloud will automatically install them
    
    ### 🔧 Package Details
    
    - **ultralytics**: YOLOv8 object detection and segmentation
    - **plotly**: Interactive data visualizations
    - **opencv-python**: Computer vision operations
    """)
    
    with st.expander("📄 View Complete requirements.txt"):
        st.code("""streamlit>=1.28.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
ultralytics>=8.0.0
plotly>=5.17.0""", language="text")

def main():
    st.set_page_config(
        page_title="AgriGrade AI: Onion Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check dependencies
    missing_deps = check_dependencies()
    
    if missing_deps:
        st.title("🧅 AgriGrade AI: Precision Onion Grading System")
        show_dependency_warning(missing_deps)
        return
    
    # Header
    st.title("🧅 AgriGrade AI: Precision Onion Grading System")
    st.markdown("""
    **AI-Powered Onion Sizing & Quality Control** | **Engine:** YOLOv8-seg + ArUco Metrology  
    *Automated USDA-compliant grading using computer vision and deep learning*
    """)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ System Configuration")
    
    marker_size = st.sidebar.number_input(
        "ArUco Marker Size (mm)",
        min_value=10.0,
        max_value=200.0,
        value=DEFAULT_MARKER_SIZE_MM,
        step=1.0,
        help="Physical size of your ArUco calibration marker"
    )
    
    conf_threshold = st.sidebar.slider(
        "AI Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.25,
        step=0.05,
        help="Lower = more detections (may include false positives)"
    )
    
    show_advanced = st.sidebar.checkbox(
        "Show Advanced Metrics",
        value=False,
        help="Display eccentricity and ellipse measurements"
    )
    
    st.sidebar.divider()
    st.sidebar.markdown("""
    ### 📋 USDA Grade Standards
    - **Small:** < 50.8mm (2")
    - **Medium:** 50.8-76.2mm (2-3")
    - **Large:** 76.2-95mm (3-3.75")
    - **Colossal:** > 95mm (3.75"+)
    """)
    
    # Main content
    uploaded_file = st.file_uploader(
        "📁 Upload Batch Image (Top-Down View Required)",
        type=['jpg', 'png', 'jpeg', 'bmp'],
        help="Image should contain visible ArUco marker for calibration"
    )
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            st.error("Failed to decode image. Please upload a valid image file.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Input & Calibration")
            st.image(
                cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Original Image ({img_bgr.shape[1]}x{img_bgr.shape[0]})",
                use_container_width=True
            )
            
            with st.status("🎯 Performing Calibration...", expanded=True) as status:
                ppm, ids, debug_img, corners = detect_aruco_and_get_ppm(
                    img_bgr.copy(),
                    marker_size
                )
                
                if ppm:
                    st.success(f"✅ Marker Detected: ID {ids.flatten().tolist()}")
                    st.metric("Scale Factor", f"{ppm:.3f} px/mm")
                    st.metric("Image Resolution", f"{1000/ppm:.2f} mm per 1000px")
                    status.update(label="✅ Calibration Complete", state="complete")
                else:
                    st.error("❌ No ArUco marker detected")
                    st.warning("⚠️ Measurements will be in PIXELS (uncalibrated)")
                    ppm = 1.0
                    status.update(label="❌ Calibration Failed", state="error")
        
        with col2:
            st.subheader("🤖 AI Segmentation & Grading")
            
            model, error = load_model()
            if error:
                st.error(f"Model loading failed: {error}")
                return
            
            with st.spinner("🔍 Running YOLOv8 Inference..."):
                processed_img, data = process_onions_yolo(
                    model,
                    img_bgr.copy(),
                    ppm,
                    conf_threshold,
                    show_advanced
                )
            
            st.image(
                cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
                caption=f"Segmented Output ({len(data)} onions detected)",
                use_container_width=True
            )
        
        if data:
            st.divider()
            st.subheader("📊 Batch Analytics Dashboard")
            
            df = pd.DataFrame(data)
            
            metric_cols = st.columns(5)
            metric_cols[0].metric("🧅 Total Count", len(df))
            metric_cols[1].metric("📏 Avg Diameter", f"{df['Diameter (mm)'].mean():.1f} mm")
            metric_cols[2].metric("📐 Min Diameter", f"{df['Diameter (mm)'].min():.1f} mm")
            metric_cols[3].metric("📐 Max Diameter", f"{df['Diameter (mm)'].max():.1f} mm")
            metric_cols[4].metric("🎯 Avg Confidence", f"{df['Confidence'].mean():.2%}")
            
            st.divider()
            
            tab1, tab2, tab3 = st.tabs(["📈 Charts", "📋 Data Table", "📥 Export"])
            
            with tab1:
                if PLOTLY_AVAILABLE:
                    fig_pie, fig_hist, fig_scatter = create_visualizations(df)
                    
                    chart_col1, chart_col2 = st.columns(2)
                    with chart_col1:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with chart_col2:
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.warning("Install plotly for interactive charts: `pip install plotly`")
                    
                    # Fallback: Simple bar chart
                    st.bar_chart(df['Grade'].value_counts())
            
            with tab2:
                st.write("#### 📊 Grade Distribution Summary")
                grade_summary = df.groupby('Grade').agg({
                    'ID': 'count',
                    'Diameter (mm)': ['mean', 'min', 'max'],
                    'Confidence': 'mean'
                }).round(2)
                grade_summary.columns = ['Count', 'Avg Diameter', 'Min Diameter', 'Max Diameter', 'Avg Confidence']
                st.dataframe(grade_summary, use_container_width=True)
                
                st.write("#### 📝 Detailed Measurements")
                st.dataframe(df, use_container_width=True, hide_index=True)
            
            with tab3:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download CSV Report",
                    csv,
                    f"onion_grading_{timestamp}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                summary_text = f"""
AGRIGRADE AI - BATCH REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*50}

SUMMARY STATISTICS
- Total Onions Detected: {len(df)}
- Average Diameter: {df['Diameter (mm)'].mean():.2f} mm
- Standard Deviation: {df['Diameter (mm)'].std():.2f} mm
- Calibration: {ppm:.3f} pixels/mm

GRADE BREAKDOWN
{df['Grade'].value_counts().to_string()}

QUALITY METRICS
- Average AI Confidence: {df['Confidence'].mean():.2%}
- Shape Uniformity (Avg Eccentricity): {df['Eccentricity'].mean():.3f}
                """
                
                st.download_button(
                    "📄 Download Text Summary",
                    summary_text,
                    f"onion_summary_{timestamp}.txt",
                    "text/plain",
                    use_container_width=True
                )
        else:
            st.info("🔍 No onions detected. Try adjusting the confidence threshold or check image quality.")
    
    else:
        st.info("👆 Upload an image to begin automated grading")
        
        with st.expander("📖 Quick Start Guide"):
            st.markdown("""
            1. **Prepare Your Setup:**
               - Print a 4x4 ArUco marker (ID 0-49)
               - Measure marker size accurately
               - Place marker in frame with onions
            
            2. **Capture Image:**
               - Use top-down view (perpendicular to onions)
               - Ensure good, even lighting
               - Avoid shadows and reflections
            
            3. **Upload & Analyze:**
               - Upload your image
               - System auto-calibrates using marker
               - AI detects and grades each onion
            
            4. **Review & Export:**
               - Check measurements and grades
               - Download CSV for record-keeping
            """)

if __name__ == "__main__":
    main()
