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
    st.set_page_config(page_title="Onion AI: Custom Grader", layout="wide")
    st.title("🧅 Onion AI: Custom Grader")
    st.caption("Defaults set to your optimized calibration.")

    # --- Sidebar ---
    st.sidebar.header("1. Detection & Filters")
    
    # Tooltip added: Explains roundness
    min_circularity = st.sidebar.slider(
        "Roundness Filter", 0.0, 1.0, 0.44, 
        help="Controls how 'perfectly round' an object must be to be counted.\n\n1.0 = Perfect Circle.\n0.44 = Your custom setting for oval onions.\nLower this if valid onions are being ignored."
    )
    
    # Tooltip added: Explains size cutoff
    min_size_mm = st.sidebar.number_input(
        "Min Onion Size (mm)", value=30.0,
        help="Any object smaller than this size will be completely ignored (treated as dirt/noise)."
    )
    
    st.sidebar.header("2. Color Tuning")
    with st.sidebar.expander("Show Color Settings", expanded=True):
        st.write("**Green Cap**")
        
        # Tooltip added: Explains Green Range
        c_h_min = st.slider(
            "Green Hue Min", 0, 179, 45,
            help="Start of the Green color range. Decrease if the cap is not detected."
        )
        c_h_max = st.slider(
            "Green Hue Max", 0, 179, 73,
            help="End of the Green color range. Increase if the cap is not detected."
        )
        
        st.markdown("---")
        st.write("**Red Onion**")
        
        o_h_min = st.slider("Red Hue Min", 0, 179, 0)
        o_h_max = st.slider("Red Hue Max", 0, 179, 179)
        
        # Tooltip added: Explains Saturation (Pale onions)
        o_s_min = st.slider(
            "Red Saturation Min", 0, 255, 0,
            help="Minimum color intensity.\n\nKeep this at 0 to detect onions with pale, dry, or white skins."
        ) 
        
        # Tooltip added: Explains Value (Black mat)
        o_v_min = st.slider(
            "Red Brightness (Val) Min", 0, 255, 69,
            help="Minimum brightness.\n\nSet to ~69 to ignore the black mat.\nLower it if dark purple onions are disappearing."
        )

    st.sidebar.header("3. Calibration")
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)
    
    # Tooltip added: Explains Fine Tuning
    calib_factor = st.sidebar.slider(
        "Fine Tune %", 80, 120, 97,
        help="Adjusts the final mm calculation.\n\n97% = Reducing measured size by 3% (Your custom calibration).\nUse this if the app consistently measures slightly too big or too small."
    ) / 100.0
    
    debug_mode = st.sidebar.checkbox("Show Debug Masks", value=False, help="View the black & white computer vision layers to troubleshoot detection.")

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        display_img = img.copy()

        # --- MASKS ---
        # 1. Green Mask (Using your 45-73 range)
        mask_cap = cv2.inRange(hsv, np.array([c_h_min, 50, 50]), np.array([c_h_max, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask_cap = cv2.morphologyEx(mask_cap, cv2.MORPH_OPEN, kernel)
        
        # 2. Red Mask (Using your Brightness 69 cutoff)
        mask_onion = cv2.inRange(hsv, np.array([o_h_min, o_s_min, o_v_min]), np.array([o_h_max, 255, 255]))
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel)
        mask_onion = cv2.dilate(mask_onion, kernel, iterations=1)

        # --- PROCESS ---
        cnts_cap = get_contours(mask_cap, 200)
        scale = 0
        ref_found = False

        # Find Reference
        if cnts_cap:
            ref_contour = cnts_cap[0] 
            ref_px = get_min_axis_width(ref_contour)
            if ref_px > 0:
                scale = (ref_px / ref_width_mm) * calib_factor
                ref_found = True
                
                box = cv2.boxPoints(cv2.minAreaRect(ref_contour))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (255, 0, 0), 3)
                # Label Ref
                cv2.putText(display_img, "REF", (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Find Onions
        onion_data = []
        if ref_found:
            cnts_onion = get_contours(mask_onion, 200)
            
            for cnt in cnts_onion:
                # Filter: Circularity
                circ = get_circularity(cnt)
                if circ < min_circularity:
                    continue 

                w_px = get_min_axis_width(cnt)
                w_mm = w_px / scale
                
                # Filter: Size
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
                
                # Label (Large & Clear)
                label = f"{int(w_mm)}"
                x, y = box[1][0], box[1][1]
                # Keep label inside screen
                x = max(10, min(x, display_img.shape[1] - 80))
                y = max(30, min(y, display_img.shape[0] - 10))
                
                cv2.putText(display_img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # --- DISPLAY ---
        col1, col2 = st.columns([1.5, 1])
        
        with col1:
            st.image(display_img, channels="BGR", caption=f"Calibrated at {int(calib_factor*100)}%", use_column_width=True)
            if debug_mode:
                st.caption("Debug: Red Mask")
                st.image(mask_onion, channels='GRAY')
            
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
                
                # CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "onions.csv", "text/csv")

            elif not ref_found:
                st.error("Reference Cap NOT found.")
                st.write(f"Searching for Green Hue: {c_h_min} to {c_h_max}")
            else:
                st.warning("No onions detected.")

if __name__ == "__main__":
    main()
