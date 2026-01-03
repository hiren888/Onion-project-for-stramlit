import streamlit as st

try:
    import cv2
    import numpy as np
    import pandas as pd
except ImportError as e:
    st.error(f"CRITICAL ERROR: Library failed to load. {e}")
    st.stop()

def get_min_axis_width(contour):
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    return min(width, height)

def get_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = [c for c in contours if cv2.contourArea(c) > min_area]
    return sorted(clean, key=cv2.contourArea, reverse=True)

def watershed_separation(mask, sensitivity=0.4, erosion_iter=0):
    """
    Separates touching objects using Distance Transform + Watershed.
    """
    # 1. CLEANUP (Clump Breaking)
    # Erode the mask to snap thin connections (onion skins)
    kernel = np.ones((3,3), np.uint8)
    if erosion_iter > 0:
        mask = cv2.erode(mask, kernel, iterations=erosion_iter)
    
    # Clean noise (Opening)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 2. FIND SURE BACKGROUND
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 3. FIND SURE FOREGROUND (Peaks)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Normalize for visualization
    dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Threshold to find peaks
    ret, sure_fg = cv2.threshold(dist_transform, sensitivity * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 4. WATERSHED
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_bgr, markers)
    
    # 5. EXTRACT CONTOURS
    final_contours = []
    centers = [] # To store the peaks for debugging
    
    # Get centers from sure_fg
    peak_contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for pc in peak_contours:
        M = cv2.moments(pc)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))

    for label in np.unique(markers):
        if label <= 1: continue
        
        # Create mask for this single object
        obj_mask = np.zeros(mask.shape, dtype=np.uint8)
        obj_mask[markers == label] = 255
        
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # Take the convex hull to smooth out jagged watershed edges
            hull = cv2.convexHull(cnts[0])
            final_contours.append(hull)
            
    return final_contours, centers, dist_vis

def main():
    st.set_page_config(page_title="Onion AI: Clump Breaker", layout="wide")
    st.title("🧅 Onion AI: Clump Breaker")
    st.caption("Use the 'Blue Dots' to tune separation.")

    # --- Sidebar ---
    st.sidebar.header("1. Separation Tuning")
    
    # SLIDER 1: PEAK SENSITIVITY
    peak_sens = st.sidebar.slider("1. Peak Sensitivity", 0.1, 0.9, 0.4, step=0.05, 
                                help="Control the Blue Dots.\nToo many dots on one onion? Slide LEFT.\nOne dot for two onions? Slide RIGHT.")
    
    # SLIDER 2: CLUMP BREAKER
    clump_iter = st.sidebar.slider("2. Clump Breaker (Erosion)", 0, 5, 2, 
                                 help="Cuts the thin skin bridges between onions.\nIncrease if onions are stuck together.")

    st.sidebar.header("2. Detection (Rust/Color)")
    sat_min = st.sidebar.slider("Min Vibrancy (Sat)", 0, 255, 60, help="Increase to remove rusty tray background.")
    val_min = st.sidebar.slider("Min Brightness", 0, 255, 50)
    
    st.sidebar.header("3. Physical Settings")
    min_size_mm = st.sidebar.number_input("Min Onion Size (mm)", value=35.0)
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)

    debug_mode = st.sidebar.checkbox("Show Debug Layers", value=True)

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        display_img = img.copy()

        # --- MASKS ---
        # Cap
        mask_cap = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
        
        # Onion (Vibrancy)
        mask_onion = cv2.inRange(hsv, np.array([0, sat_min, val_min]), np.array([179, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel, iterations=2) # Fill holes first

        # --- SEPARATION ---
        cnts_onion, centers, dist_map = watershed_separation(mask_onion, peak_sens, clump_iter)

        # --- MEASURE ---
        cnts_cap = get_contours(mask_cap, 200)
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

        onion_data = []
        if ref_found:
            for cnt in cnts_onion:
                if cv2.contourArea(cnt) < 500: continue
                
                # Compensate for Clump Breaker shrinking
                w_px = get_min_axis_width(cnt) + (clump_iter * 2) 
                w_mm = w_px / scale
                
                if w_mm < min_size_mm: continue 

                if w_mm >= 65: grade = "L"
                elif w_mm >= 55: grade = "M"
                else: grade = "S"
                
                onion_data.append({"Size": w_mm, "Grade": grade})
                
                # Draw
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                
                # Label
                cv2.putText(display_img, f"{int(w_mm)}", (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw "Centers" (Blue Dots) for tuning
            if debug_mode:
                for (cx, cy) in centers:
                    cv2.circle(display_img, (cx, cy), 5, (255, 0, 0), -1) # Blue Dot
                    cv2.circle(display_img, (cx, cy), 2, (255, 255, 255), -1) # White center

        # --- DISPLAY ---
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(display_img, channels="BGR", use_column_width=True, caption="Blue Dots = Onion Centers")
            if debug_mode:
                st.image(dist_map, caption="Separation Map (Brighter = More likely to be center)", clamp=True)

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
            else:
                st.warning("No onions detected.")

if __name__ == "__main__":
    main()
