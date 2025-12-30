import streamlit as st
import cv2
import numpy as np
import pandas as pd

def get_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    clean_contours = sorted(clean_contours, key=cv2.contourArea, reverse=True)
    return clean_contours

def get_min_axis_width(contour):
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    return min(width, height)

def get_circularity(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0: return 0
    return 4 * np.pi * (area / (perimeter * perimeter))

def main():
    st.set_page_config(page_title="Onion AI: Master Grader", layout="wide")
    st.title("🧅 Onion AI: Master Grader")
    st.markdown("### Final Polish Mode")

    # --- Sidebar ---
    st.sidebar.header("1. Detection & Filters")
    min_size_mm = st.sidebar.number_input("Min Onion Size (mm)", value=45.0)
    min_circularity = st.sidebar.slider("Roundness Filter", 0.1, 1.0, 0.6, help="1.0 is a perfect circle. 0.6 allows slightly oval onions. Increase to remove long/messy objects.")
    
    st.sidebar.header("2. Color Tuning")
    with st.sidebar.expander("Show Color Settings"):
        st.write("**Green Cap**")
        c_h_min = st.slider("Green Hue Min", 0, 179, 35)
        c_h_max = st.slider("Green Hue Max", 0, 179, 95) 
        st.write("**Red Onion**")
        o_h_min = st.slider("Red Hue Min", 0, 179, 0)
        o_h_max = st.slider("Red Hue Max", 0, 179, 179)
        o_v_min = st.slider("Red Value Min", 0, 255, 50)

    st.sidebar.header("3. Calibration")
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)
    calib_factor = st.sidebar.slider("Fine Tune %", 90, 110, 100, help="If measurements are consistently off, adjust this.") / 100.0
    
    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        display_img = img.copy()

        # --- MASKS ---
        # Green Mask
        mask_cap = cv2.inRange(hsv, np.array([c_h_min, 60, 50]), np.array([c_h_max, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask_cap = cv2.morphologyEx(mask_cap, cv2.MORPH_OPEN, kernel)
        
        # Red Mask
        mask_onion = cv2.inRange(hsv, np.array([o_h_min, 50, o_v_min]), np.array([o_h_max, 255, 255]))
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel)
        mask_onion = cv2.dilate(mask_onion, kernel, iterations=1)

        # --- PROCESS ---
        cnts_cap = get_contours(mask_cap, 200)
        scale = 0
        ref_found = False

        # 1. Find Reference
        if cnts_cap:
            ref_contour = cnts_cap[0] # Largest Green
            ref_px = get_min_axis_width(ref_contour)
            
            if ref_px > 0:
                scale = (ref_px / ref_width_mm) * calib_factor
                ref_found = True
                
                # Draw Blue Box
                box = cv2.boxPoints(cv2.minAreaRect(ref_contour))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (255, 0, 0), 3)
                cv2.putText(display_img, f"REF {ref_width_mm}mm", (box[0][0], box[0][1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # 2. Find Onions
        onion_data = []
        if ref_found:
            cnts_onion = get_contours(mask_onion, 200)
            
            for cnt in cnts_onion:
                # --- Filter: Circularity ---
                circ = get_circularity(cnt)
                if circ < min_circularity:
                    continue # Skip objects that aren't round enough

                w_px = get_min_axis_width(cnt)
                w_mm = w_px / scale
                
                # --- Filter: Size ---
                if w_mm < min_size_mm:
                    continue 

                # Determine Grade
                if w_mm >= 65: grade = "L"
                elif w_mm >= 55: grade = "M"
                else: grade = "S"
                
                onion_data.append({"Size": w_mm, "Grade": grade})
                
                # Draw Box
                box = cv2.boxPoints(cv2.minAreaRect(cnt))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                
                # Draw Label (Size + Grade)
                label = f"{int(w_mm)}mm ({grade})"
                x, y = box[1][0], box[1][1]
                
                # Smart label placement (ensure it stays on screen)
                x = max(10, min(x, display_img.shape[1] - 100))
                y = max(30, min(y, display_img.shape[0] - 10))
                
                # Label Background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(display_img, (int(x), int(y) - h - 5), (int(x) + w, int(y) + 5), (0,0,0), -1)
                cv2.putText(display_img, label, (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- DISPLAY RESULTS ---
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.subheader("📸 Processed Image")
            st.image(display_img, channels="BGR", use_column_width=True)
            
        with col2:
            st.subheader("📊 Grading Report")
            if ref_found and onion_data:
                df = pd.DataFrame(onion_data)
                
                # Metrics
                total = len(df)
                large = len(df[df['Grade'] == 'L'])
                medium = len(df[df['Grade'] == 'M'])
                small = len(df[df['Grade'] == 'S'])
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Large (>65)", large, f"{large/total:.0%}")
                c2.metric("Medium (55-64)", medium, f"{medium/total:.0%}")
                c3.metric("Small (<55)", small, f"{small/total:.0%}")
                
                # Data Table
                st.divider()
                st.write("Detailed Data:")
                st.dataframe(df.style.format({"Size": "{:.1f} mm"}))
                
                # CSV Download
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Report (CSV)", csv, "onion_report.csv", "text/csv")
            
            elif not ref_found:
                st.error("Reference cap not found.")
            else:
                st.warning("No onions detected.")

if __name__ == "__main__":
    main()
