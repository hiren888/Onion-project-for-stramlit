import streamlit as st

# --- SAFETY BLOCK ---
try:
    import cv2
    import numpy as np
    import pandas as pd
except ImportError as e:
    st.error(f"CRITICAL ERROR: Library failed to load. {e}")
    st.stop()

def apply_clahe(img_bgr):
    """Boosts contrast to separate onions from dull background."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def get_contours_watershed(mask, min_area, max_area=None):
    """
    Advanced logic to separate touching objects using Distance Transform.
    """
    # 1. Clean noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 2. Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 3. Finding sure foreground area (Distance Transform)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Threshold to get the peaks (centers of onions)
    # We take 40% of max distance as the cutoff for a "peak"
    ret, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    
    sure_fg = np.uint8(sure_fg)
    
    # 4. Find Contours from the separate peaks
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            # Optional: Check geometric max size to avoid detecting the whole tray
            if max_area and area > max_area:
                continue
            valid_contours.append(cnt)
            
    # Sort largest to smallest
    valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
    return valid_contours

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
    st.set_page_config(page_title="Onion AI: Tray Master", layout="wide")
    st.title("🧅 Onion AI: Tray Master")
    st.caption("Optimized for touching onions and rusty backgrounds.")

    # --- Sidebar ---
    st.sidebar.header("1. Scenario Settings")
    crowded_mode = st.sidebar.checkbox("Crowded / Touching Onions", value=True, help="Enable this to separate onions that are touching each other.")
    boost_contrast = st.sidebar.checkbox("Boost Contrast (Rusty Tray Fix)", value=True, help="Makes onions pop out against dull backgrounds.")

    st.sidebar.header("2. Physical Settings")
    min_size_mm = st.sidebar.number_input("Min Onion Size (mm)", value=35.0)
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)
    
    st.sidebar.header("3. Color Tuning (Manual)")
    # IMPORTANT: Default Saturation Min raised to 40 to hide the dull rusty tray
    o_s_min = st.sidebar.slider("Saturation Min (Background Remover)", 0, 255, 40, help="Increase this to 50-80 to remove the rusty tray.")
    o_v_min = st.sidebar.slider("Brightness Min", 0, 255, 40)
    o_h_min = st.sidebar.slider("Hue Min", 0, 179, 0)
    o_h_max = st.sidebar.slider("Hue Max", 0, 179, 179)
    
    with st.sidebar.expander("Advanced Green Cap Settings"):
        c_h_min = st.slider("Green Hue Min", 0, 179, 35)
        c_h_max = st.slider("Green Hue Max", 0, 179, 85)

    debug_mode = st.sidebar.checkbox("Show Debug View", value=True)

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # 1. PRE-PROCESS: Contrast Boost
        if boost_contrast:
            img_processed = apply_clahe(img)
        else:
            img_processed = img.copy()
            
        hsv = cv2.cvtColor(img_processed, cv2.COLOR_BGR2HSV)
        display_img = img.copy()

        # 2. MASKS
        # Green Mask
        mask_cap = cv2.inRange(hsv, np.array([c_h_min, 50, 50]), np.array([c_h_max, 255, 255]))
        
        # Red/Onion Mask
        mask_onion = cv2.inRange(hsv, np.array([o_h_min, o_s_min, o_v_min]), np.array([o_h_max, 255, 255]))
        
        # 3. FIND CONTOURS (Logic Switch)
        
        # A. Reference Cap (Always distinct)
        cnts_cap = cv2.findContours(mask_cap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cnts_cap = [c for c in cnts_cap if cv2.contourArea(c) > 200]
        cnts_cap = sorted(cnts_cap, key=cv2.contourArea, reverse=True)
        
        scale = 0
        ref_found = False

        if cnts_cap:
            ref_contour = cnts_cap[0]
            ref_px = get_min_axis_width(ref_contour)
            if ref_px > 0:
                scale = (ref_px / ref_width_mm)
                ref_found = True
                box = cv2.boxPoints(cv2.minAreaRect(ref_contour))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (255, 0, 0), 3)
                cv2.putText(display_img, "REF", (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # B. Onions (Touching logic)
        onion_data = []
        if ref_found:
            if crowded_mode:
                # Use Watershed/Distance Transform logic
                # Max area limit prevents detecting the whole tray frame as an onion
                cnts_onion = get_contours_watershed(mask_onion, min_area=300, max_area=img.shape[0]*img.shape[1]*0.8)
            else:
                # Use Standard logic
                cnts_onion = cv2.findContours(mask_onion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                cnts_onion = [c for c in cnts_onion if cv2.contourArea(c) > 300]
            
            for cnt in cnts_onion:
                # We relax circularity in Crowded Mode because Separation isn't always perfect circles
                min_circ = 0.2 if crowded_mode else 0.4
                circ = get_circularity(cnt)
                
                # Draw centers for debugging
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else: cX, cY = 0, 0
                
                if circ < min_circ: 
                    # Visual feedback for ignored objects (Yellow dots)
                    if debug_mode: cv2.circle(display_img, (cX, cY), 5, (0, 255, 255), -1)
                    continue 

                w_px = get_min_axis_width(cnt)
                w_mm = w_px / scale
                
                if w_mm < min_size_mm: continue 

                if w_mm >= 65: grade = "L"
                elif w_mm >= 55: grade = "M"
                else: grade = "S"
                
                onion_data.append({"Size": w_mm, "Grade": grade})
                
                # Draw Box
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                
                # Smart Labeling (Only show MM to avoid clutter)
                cv2.putText(display_img, f"{int(w_mm)}", (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- DISPLAY ---
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.image(display_img, channels="BGR", use_column_width=True, caption="Green=Onions, Blue=Ref, YellowDot=Ignored Noise")
            if debug_mode: 
                st.image(mask_onion, caption="Computer Vision Mask (White = Detected)", channels='GRAY')
            
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
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download CSV", csv, "onions.csv", "text/csv")
            elif not ref_found:
                st.error("Reference Cap NOT found.")
                st.write("Ensure the Green Cap is visible and not covered.")
            else:
                st.warning("No onions detected.")
                st.write("Tip: If you see the tray in the 'Mask' image, increase 'Saturation Min'.")

if __name__ == "__main__":
    main()
