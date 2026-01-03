import streamlit as st

# --- SAFETY BLOCK ---
try:
    import cv2
    import numpy as np
    import pandas as pd
except ImportError as e:
    st.error(f"CRITICAL ERROR: Library failed to load. {e}")
    st.stop()

def get_min_axis_width(contour):
    """Calculates the width (short side) of the object."""
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    return min(width, height)

def get_contours(mask, min_area):
    """Standard contour finding."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort largest to smallest
    clean = sorted([c for c in contours if cv2.contourArea(c) > min_area], 
                   key=cv2.contourArea, reverse=True)
    return clean

def watershed_robust(mask, sensitivity=0.4, blur_strength=5):
    """
    Advanced Watershed:
    1. Blurs the distance map to merge 'skin noise' into one single peak.
    2. Finds peaks.
    3. Runs watershed.
    """
    # 1. Clean the mask (remove loose skin bridges)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 2. Sure Background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # 3. Distance Transform (The Height Map)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # --- CRITICAL FIX: BLUR THE PEAKS ---
    # This smooths out flaky skin so we don't get 5 dots for 1 onion
    if blur_strength > 0:
        # Ensure kernel size is odd
        k_size = (blur_strength * 2) + 1 
        dist_transform = cv2.GaussianBlur(dist_transform, (k_size, k_size), 0)
    
    # Normalize for visualization
    dist_vis = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 4. Find Peaks (Sure Foreground)
    ret, sure_fg = cv2.threshold(dist_transform, sensitivity * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # 5. Watershed Markers
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mask_bgr, markers)
    
    # 6. Extract Contours
    final_contours = []
    
    # Get the centers for debugging
    centers = []
    peak_cnts, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for pc in peak_cnts:
        M = cv2.moments(pc)
        if M["m00"] != 0:
            centers.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))

    # Process Watershed Results
    for label in np.unique(markers):
        if label <= 1: continue # Skip background and boundary
        
        # Create a mask for this single object
        obj_mask = np.zeros(mask.shape, dtype=np.uint8)
        obj_mask[markers == label] = 255
        
        cnts, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # Convex Hull smooths the jagged watershed edges
            hull = cv2.convexHull(cnts[0])
            final_contours.append(hull)
            
    return final_contours, centers, dist_vis

def filter_overlapping_boxes(onion_data):
    """
    Removes 'Spiderwebs' (boxes inside other boxes).
    """
    if not onion_data: return []
    
    # Sort by size (largest first)
    sorted_data = sorted(onion_data, key=lambda x: x['Size'], reverse=True)
    kept_onions = []
    
    for i, onion in enumerate(sorted_data):
        box = onion['Box']
        is_inside = False
        
        center_x, center_y = np.mean(box, axis=0)
        
        # Check if this onion's center is inside any LARGER onion's box
        for larger_onion in kept_onions:
            l_box = larger_onion['Box']
            # PointPolygonTest returns > 0 if point is inside
            if cv2.pointPolygonTest(l_box, (center_x, center_y), False) > 0:
                is_inside = True
                break
        
        if not is_inside:
            kept_onions.append(onion)
            
    return kept_onions

def main():
    st.set_page_config(page_title="Onion AI: Ultimate", layout="wide")
    st.title("🧅 Onion AI: Ultimate Grader")
    st.caption("Deep Logic: Vibrancy Detection + Gaussian Peak Smoothing + Overlap Removal")

    # --- Sidebar ---
    st.sidebar.header("1. Detection (The 'What')")
    # Vibrancy is key for Rusty Trays
    sat_min = st.sidebar.slider("Minimum Vibrancy (Sat)", 0, 255, 65, help="Increase to remove rusty tray. Decrease if onions disappear.")
    
    st.sidebar.header("2. Separation (The 'How')")
    peak_sens = st.sidebar.slider("Peak Sensitivity", 0.1, 0.9, 0.4, help="Controls Blue Dots. Higher = Splits clumps. Lower = Merges clumps.")
    blur_peaks = st.sidebar.slider("Texture Smoothing", 0, 10, 4, help="CRITICAL: Increase this to fix 'Spiderwebs' (multiple boxes on one onion).")
    
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

        # --- STEP 1: MASKS ---
        # Green Cap (Standard Range)
        mask_cap = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        
        # Onion (Vibrancy Logic - Ignores Hue, focuses on Saturation)
        # We assume onions are 'Colorful' (Sat > X) and 'Not Dark' (Val > 40)
        # We cap Hue at 179 to catch Red/Purple/Orange
        mask_onion = cv2.inRange(hsv, np.array([0, sat_min, 40]), np.array([179, 255, 255]))
        
        # Clean noise
        kernel = np.ones((3, 3), np.uint8)
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel, iterations=4) # Strong close to solidify skin

        # --- STEP 2: WATERSHED ---
        raw_contours, centers, dist_map = watershed_robust(mask_onion, sensitivity=peak_sens, blur_strength=blur_peaks)

        # --- STEP 3: REFERENCE ---
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

        # --- STEP 4: MEASURE & FILTER ---
        pre_filter_data = []
        if ref_found:
            for cnt in raw_contours:
                if cv2.contourArea(cnt) < 500: continue
                
                w_px = get_min_axis_width(cnt)
                w_mm = w_px / scale
                
                if w_mm < min_size_mm: continue 

                # Store data for overlap filtering
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = box.astype(int)
                
                pre_filter_data.append({
                    "Size": w_mm,
                    "Box": box,
                    "Contour": cnt
                })

            # --- STEP 5: REMOVE OVERLAPPING BOXES ---
            final_onions = filter_overlapping_boxes(pre_filter_data)
            
            # Draw Final Results
            for onion in final_onions:
                w_mm = onion['Size']
                box = onion['Box']
                
                if w_mm >= 65: grade = "L"
                elif w_mm >= 55: grade = "M"
                else: grade = "S"
                
                # Draw Green Box
                cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                
                # Label
                label = f"{int(w_mm)}"
                # Ensure label is on screen
                x, y = box[1][0], box[1][1]
                x = max(20, min(x, display_img.shape[1]-50))
                y = max(20, min(y, display_img.shape[0]-20))
                
                cv2.putText(display_img, label, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- DISPLAY ---
        col1, col2 = st.columns([2, 1])
        with col1:
            st.image(display_img, channels="BGR", use_column_width=True, caption="Final Grading")
            
            if debug_mode:
                st.write("### 🔍 Debug Views")
                c_d1, c_d2 = st.columns(2)
                c_d1.image(mask_onion, caption="1. Vibrancy Mask (White=Onion)", channels='GRAY')
                # Visualize the smoothed peaks
                c_d2.image(dist_map, caption="2. Smoothed Peaks (White=Centers)", clamp=True)
                st.caption("Tip: If the 'Smoothed Peaks' image looks grainy, increase 'Texture Smoothing'.")

        with col2:
            st.subheader("📊 Report")
            if ref_found and final_onions:
                df = pd.DataFrame([{"Size": o['Size']} for o in final_onions])
                # Add grades for the CSV
                df['Grade'] = df['Size'].apply(lambda x: 'L' if x>=65 else ('M' if x>=55 else 'S'))
                
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
