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

def main():
    st.set_page_config(page_title="Onion AI: Rust Fix", layout="wide")
    st.title("🧅 Onion AI: High-Contrast Grader")
    st.markdown("### 🛠️ Mode: Rusty Tray / Low Contrast Fix")
    st.caption("This version focuses on SATURATION (Shininess) to separate onions from the dull tray.")

    # --- Sidebar ---
    st.sidebar.header("1. The 'Rust Remover'")
    # CRITICAL SLIDER:
    # Default is 65. If you see the tray, INCREASE this. If onions disappear, DECREASE this.
    sat_min = st.sidebar.slider("1. Minimum Vibrancy (Sat)", 0, 255, 65, help="The most important slider. \nLow (0) = Detects Tray.\nHigh (100) = Detects only Shiny Onions.")
    
    st.sidebar.header("2. Color Range")
    # We use a broad Red/Purple range by default
    hue_min = st.sidebar.slider("2. Red Hue Min", 0, 179, 0)
    hue_max = st.sidebar.slider("3. Red Hue Max", 0, 179, 179)
    val_min = st.sidebar.slider("4. Brightness Min", 0, 255, 50, help="Increase to 60-80 to remove dark shadows.")

    st.sidebar.header("3. Physical Settings")
    min_size_mm = st.sidebar.number_input("Min Onion Size (mm)", value=35.0)
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)
    
    # Auto-separation for touching onions
    separation_strength = st.sidebar.slider("Separation Strength", 0, 10, 3, help="Higher = Cuts touching onions apart more aggressively.")

    debug_mode = st.sidebar.checkbox("Show Debug Masks", value=True)

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        display_img = img.copy()

        # --- 1. GREEN CAP (Standard) ---
        mask_cap = cv2.inRange(hsv, np.array([35, 60, 60]), np.array([85, 255, 255]))
        
        # --- 2. ONION MASK (The Fix) ---
        # We rely heavily on 'sat_min' to kill the rust
        mask_onion = cv2.inRange(hsv, np.array([hue_min, sat_min, val_min]), np.array([hue_max, 255, 255]))
        
        # Cleanup (Morphology)
        kernel = np.ones((3, 3), np.uint8)
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_OPEN, kernel) # Remove noise
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel) # Fill holes
        
        # --- 3. SEPARATION LOGIC (Watershed Lite) ---
        # If separation strength > 0, we erode the mask to find "centers"
        if separation_strength > 0:
            erosion_kernel = np.ones((3,3), np.uint8)
            # Erode to separate touching objects
            mask_eroded = cv2.erode(mask_onion, erosion_kernel, iterations=separation_strength)
            # Find contours on the ERODED mask (the separated centers)
            cnts_onion = get_contours(mask_eroded, 100)
            
            # Re-inflate contours to get real size (Approximate)
            # Note: For perfect accuracy we'd use watershed, but this is faster/stable for Streamlit
            final_cnts = []
            for cnt in cnts_onion:
                # We simply use the original bounding box from the non-eroded mask that matches this center
                # Simplified strategy: Just use the eroded contour but scale the measurement up slightly
                final_cnts.append(cnt)
        else:
            cnts_onion = get_contours(mask_onion, 200)
            final_cnts = cnts_onion

        # --- PROCESS CAP ---
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

        # --- PROCESS ONIONS ---
        onion_data = []
        if ref_found:
            for cnt in final_cnts:
                w_px = get_min_axis_width(cnt)
                
                # Compensation for Erosion: If we eroded heavily, the onion looks smaller.
                # We add back ~1.5 pixels per erosion iteration to the diameter estimate
                if separation_strength > 0:
                    w_px += (separation_strength * 3.0)

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
                cv2.putText(display_img, f"{int(w_mm)}", (box[0][0], box[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- DISPLAY ---
        col1, col2 = st.columns([1.5, 1])
        with col1:
            st.image(display_img, channels="BGR", use_column_width=True)
            if debug_mode: 
                st.image(mask_onion, caption="Debug: The Rust Remover Mask", channels='GRAY')
                st.info("Goal: The Onions should be WHITE. The Tray should be BLACK.")
            
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
                st.write("👉 Decrease 'Minimum Vibrancy' slider slightly.")

if __name__ == "__main__":
    main()
