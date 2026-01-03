import streamlit as st
try:
    import cv2
    import numpy as np
    import pandas as pd
except ImportError as e:
    st.error(f"CRITICAL ERROR: Library failed to load. {e}")
    st.stop()

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
    st.set_page_config(page_title="Onion AI: Brown/Irregular", layout="wide")
    st.title("🧅 Onion AI: Robust Grader")
    st.caption("Tuned for: Red-to-Brown colors & Irregular/Flaky skin.")

    # --- Sidebar ---
    st.sidebar.header("1. Operation Mode")
    mode = st.sidebar.radio("Detection Mode", ["🤖 Auto-Detect (Red/Brown)", "🎛️ Manual Tuning"], index=0)

    st.sidebar.header("2. Physical Settings")
    min_size_mm = st.sidebar.number_input("Min Onion Size (mm)", value=35.0)
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)
    
    if mode == "🤖 Auto-Detect (Red/Brown)":
        st.sidebar.info("Auto Mode Active: Scanning for Red, Purple, and Earth/Brown tones.")
        # LOWERED DEFAULT: 0.35 allows very irregular/messy shapes
        min_circularity = st.sidebar.slider("Roundness Sensitivity", 0.0, 1.0, 0.35, help="Kept low (0.35) to allow irregular skin shapes.")
        # INCREASED HOLE FILLER:
        morph_strength = st.sidebar.slider("Skin Repair Strength", 1, 10, 4, help="Higher = Fills bigger gaps in flaky skin.")
        calib_factor = 0.97 
        
    else:
        st.sidebar.subheader("Manual Color Tuning")
        min_circularity = st.sidebar.slider("Roundness Filter", 0.0, 1.0, 0.44)
        morph_strength = st.sidebar.slider("Morph Strength", 1, 10, 2)
        
        with st.sidebar.expander("Green Settings"):
            c_h_min = st.slider("Green Hue Min", 0, 179, 35)
            c_h_max = st.slider("Green Hue Max", 0, 179, 85)
            
        with st.sidebar.expander("Red/Brown Settings"):
            o_h_min = st.slider("Hue Min", 0, 179, 0)
            o_h_max = st.slider("Hue Max", 0, 179, 179)
            o_s_min = st.slider("Sat Min", 0, 255, 0)
            o_v_min = st.slider("Val Min", 0, 255, 69)
            
        calib_factor = st.sidebar.slider("Fine Tune %", 80, 120, 97) / 100.0

    debug_mode = st.sidebar.checkbox("Show Debug Masks", value=False)

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        display_img = img.copy()

        # --- MASKS LOGIC ---
        # The 'Skin Repair' Kernel
        kernel_size = (morph_strength * 2) + 1 # Ensure odd number (3, 5, 7...)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if mode == "🤖 Auto-Detect (Red/Brown)":
            # 1. GREEN CAP (Standard)
            mask_cap = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            
            # 2. RED/BROWN LOGIC
            # Range A: Standard Red (0-15)
            mask_red = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([15, 255, 255]))
            
            # Range B: BROWN / ORANGE (15-35) <--- NEW: Captures brown skin
            # Note: We keep Saturation/Value lower (30) to catch dry/dirty skins
            mask_brown = cv2.inRange(hsv, np.array([15, 30, 30]), np.array([35, 255, 255]))
            
            # Range C: Deep Purple/Red (160-180)
            mask_purple = cv2.inRange(hsv, np.array([160, 40, 40]), np.array([180, 255, 255]))
            
            # Combine all
            mask_onion = mask_red | mask_brown | mask_purple
            
            # 3. IRREGULAR SKIN REPAIR (Morphology)
            # "Close" fills holes inside the object (flaky skin)
            mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel)
            # "Dilate" expands the edges slightly to connect broken pieces
            mask_onion = cv2.dilate(mask_onion, kernel, iterations=1)
            
        else:
            # MANUAL MODE
            mask_cap = cv2.inRange(hsv, np.array([c_h_min, 50, 50]), np.array([c_h_max, 255, 255]))
            mask_onion = cv2.inRange(hsv, np.array([o_h_min, o_s_min, o_v_min]), np.array([o_h_max, 255, 255]))
            mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel)
            mask_onion = cv2.dilate(mask_onion, kernel, iterations=1)

        # --- PROCESSING ---
        cnts_cap = get_contours(mask_cap, 200)
        scale = 0
        ref_found = False

        if cnts_cap:
            ref_contour = cnts_cap[0]
            ref_px = get_min_axis_width(ref_contour)
            if ref_px > 0:
                scale = (ref_px / ref_width_mm) * calib_factor
                ref_found = True
                box = cv2.boxPoints(cv2.minAreaRect(ref_contour))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (255, 0, 0), 3)
                cv2.putText(display_img, "REF", (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        onion_data = []
        if ref_found:
            cnts_onion = get_contours(mask_onion, 200)
            for cnt in cnts_onion:
                circ = get_circularity(cnt)
                if circ < min_circularity: continue 
                w_px = get_min_axis_width(cnt)
                w_mm = w_px / scale
                if w_mm < min_size_mm: continue 

                if w_mm >= 65: grade = "L"
                elif w_mm >= 55: grade = "M"
                else: grade = "S"
                
                onion_data.append({"Size": w_mm, "Grade": grade})
                box = cv2.boxPoints(cv2.minAreaRect(cnt))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                
                label = f"{int(w_mm)}"
                x, y = box[1][0], box[1][1]
                x = max(10, min(x, display_img.shape[1] - 80))
                y = max(30, min(y, display_img.shape[0] - 10))
                cv2.putText(display_img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.image(display_img, channels="BGR", use_column_width=True)
            if debug_mode: st.image(mask_onion, caption="Debug Mask (Holes Filled)", channels='GRAY')
            
        with col2:
            st.subheader("📊 Report")
            if ref_found and onion_data:
                df = pd.DataFrame(onion_data)
                total = len(df)
                large = len(df[df['Grade'] == 'L'])
                medium = len(df[df['Grade'] == 'M'])
                small = len(df[df['Grade'] == 'S'])
                
                st.metric("Total Onions", total)
                c1, c2, c3 = st.columns(3)
                c1.metric("L (>65)", large, f"{large/total:.0%}")
                c2.metric("M (55-64)", medium, f"{medium/total:.0%}")
                c3.metric("S (<55)", small, f"{small/total:.0%}")
                st.dataframe(df.style.format({"Size": "{:.1f}"}), height=200)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "onions.csv", "text/csv")
            elif not ref_found:
                st.error("Reference Cap NOT found.")
            else:
                st.warning("No onions detected.")

if __name__ == "__main__":
    main()
