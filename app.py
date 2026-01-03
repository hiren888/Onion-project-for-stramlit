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

def watershed_separation(mask, min_distance_sensitivity=0.5):
    """
    Uses the Watershed algorithm to separate touching objects.
    min_distance_sensitivity: 0.0 to 1.0. Higher = splits more aggressively.
    """
    # 1. Clean small noise
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 2. Identify the "Sure Background" (dilate the object)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 3. Identify the "Sure Foreground" (Distance Transform)
    # This finds the "peaks" or centers of the onions
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Threshold to get the peaks. 
    # sensitivity controls how "tall" a peak must be to count as a separate onion.
    ret, sure_fg = cv2.threshold(dist_transform, min_distance_sensitivity * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 4. Identify the "Unknown Region" (The border where they touch)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # 5. Create Markers
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 # Add 1 so background is 1, not 0
    markers[unknown == 255] = 0 # Mark the unknown region as 0
    
    # 6. Run Watershed
    # We need a 3-channel image for watershed, so we convert mask to BGR
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_bgr, markers)
    
    # 7. Extract Contours from Markers
    final_contours = []
    # Loop through unique markers (skipping 0=boundary and 1=background)
    for label in np.unique(markers):
        if label <= 1: continue
        
        # Create a mask for just this object
        obj_mask = np.zeros(mask.shape, dtype=np.uint8)
        obj_mask[markers == label] = 255
        
        # Find contour of this separated object
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            final_contours.append(cnts[0])
            
    return final_contours, markers

def main():
    st.set_page_config(page_title="Onion AI: Watershed", layout="wide")
    st.title("🧅 Onion AI: Precision Separation")
    st.caption("Using Watershed Algorithm to split touching onions without shrinking them.")

    # --- Sidebar ---
    st.sidebar.header("1. Separation Engine")
    use_watershed = st.sidebar.checkbox("Enable Watershed Separation", value=True)
    # NEW SLIDER: The most important control for touching onions
    peak_sensitivity = st.sidebar.slider("Separation Sensitivity", 0.1, 0.9, 0.4, step=0.05, 
                                       help="Lower (0.2) = Merges onions. Higher (0.6) = Splits onions aggressively.")
    
    st.sidebar.header("2. Rust/Vibrancy Filters")
    sat_min = st.sidebar.slider("Min Vibrancy (Sat)", 0, 255, 65, help="Increase to remove rusty tray.")
    val_min = st.sidebar.slider("Min Brightness", 0, 255, 50)
    
    st.sidebar.header("3. Physical Settings")
    min_size_mm = st.sidebar.number_input("Min Onion Size (mm)", value=35.0)
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)

    debug_mode = st.sidebar.checkbox("Show Debug Maps", value=True)

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        display_img = img.copy()

        # --- MASKS ---
        # 1. Cap
        mask_cap = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
        
        # 2. Onion (Vibrancy Logic)
        mask_onion = cv2.inRange(hsv, np.array([0, sat_min, val_min]), np.array([179, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_OPEN, kernel)
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel)

        # --- SEPARATION LOGIC ---
        markers_vis = None
        
        if use_watershed:
            # Run the advanced separation
            cnts_onion, markers = watershed_separation(mask_onion, peak_sensitivity)
            
            # Create a colorful debug image for the markers
            if debug_mode:
                markers_vis = np.zeros_like(img)
                # Color code the markers
                for label in np.unique(markers):
                    if label <= 1: continue
                    color = np.random.randint(0, 255, size=3).tolist()
                    markers_vis[markers == label] = color
        else:
            # Fallback to standard
            cnts_onion = get_contours(mask_onion, 200)

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
                # Filter noise
                if cv2.contourArea(cnt) < 500: continue
                
                w_px = get_min_axis_width(cnt)
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
                cv2.putText(display_img, f"{int(w_mm)}", (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- DISPLAY ---
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.image(display_img, channels="BGR", use_column_width=True)
            if debug_mode:
                if markers_vis is not None:
                    st.image(markers_vis, caption="Watershed Segments (Each color is a separate onion)", use_column_width=True)
                else:
                    st.image(mask_onion, caption="Binary Mask", channels='GRAY')

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
