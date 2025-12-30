import streamlit as st
import cv2
import numpy as np

def get_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return clean_contours

def get_min_axis_width(contour):
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    return min(width, height)

def main():
    st.set_page_config(page_title="Onion AI: Dual Color", layout="wide")
    st.title("🧅 Onion AI: Dual Color Grader")
    st.markdown("This version detects **Red Onions** and **Green Caps** separately to ensure nothing is missed.")

    # --- Sidebar ---
    
    # 1. Onion Settings (Red/Purple)
    st.sidebar.header("🔴 1. Onion Color (Red/Purple)")
    # Note: Red wraps around 0 and 180 in HSV, so we use a wide low range usually or two ranges. 
    # Simplified here: 0-25 + 150-180 is ideal, but 0-180 with high Saturation works for black backgrounds.
    o_h_min = st.sidebar.slider("Onion Hue Min", 0, 179, 0)
    o_h_max = st.sidebar.slider("Onion Hue Max", 0, 179, 179)
    o_s_min = st.sidebar.slider("Onion Sat Min", 0, 255, 60) # Default to skip gray mat
    o_v_min = st.sidebar.slider("Onion Val Min", 0, 255, 60) # Default to skip dark shadows

    # 2. Reference Cap Settings (Green)
    st.sidebar.header("🟢 2. Cap Color (Green)")
    c_h_min = st.sidebar.slider("Cap Hue Min", 0, 179, 35)  # Green usually starts ~35
    c_h_max = st.sidebar.slider("Cap Hue Max", 0, 179, 85)  # Green usually ends ~85
    c_s_min = st.sidebar.slider("Cap Sat Min", 0, 255, 50)
    c_v_min = st.sidebar.slider("Cap Val Min", 0, 255, 50)

    st.sidebar.header("⚙️ 3. General Settings")
    morph_size = st.sidebar.slider("Hole Filler", 1, 20, 8)
    min_area = st.sidebar.slider("Min Area Filter", 100, 5000, 200)
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)
    
    # Selection Strategy
    ref_strategy = st.sidebar.radio("Which is the Reference?", 
                                    ["Smallest Object", "Largest Object"], 
                                    index=0) # Default: Smallest

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # --- MASK 1: ONIONS (Generic Color) ---
        # We use a broad range for onions but strict saturation to avoid the mat
        lower_onion = np.array([o_h_min, o_s_min, o_v_min])
        upper_onion = np.array([o_h_max, 255, 255])
        mask_onion = cv2.inRange(hsv, lower_onion, upper_onion)

        # --- MASK 2: CAP (Green Specific) ---
        lower_cap = np.array([c_h_min, c_s_min, c_v_min])
        upper_cap = np.array([c_h_max, 255, 255])
        mask_cap = cv2.inRange(hsv, lower_cap, upper_cap)

        # Combine Masks (Onion OR Cap)
        mask_combined = cv2.bitwise_or(mask_onion, mask_cap)

        # Cleanup
        kernel = np.ones((morph_size, morph_size), np.uint8)
        mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

        contours = get_contours(mask_clean, min_area)
        
        # --- Visualization ---
        display_img = img.copy()
        debug_img = cv2.cvtColor(mask_clean, cv2.COLOR_GRAY2BGR) # For visual comparison
        onion_data = []

        if len(contours) > 1:
            # Sort contours
            if ref_strategy == "Smallest Object":
                contours = sorted(contours, key=cv2.contourArea, reverse=False)
            else:
                contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            ref_contour = contours[0]
            ref_px = get_min_axis_width(ref_contour)
            
            # Draw REFERENCE (Blue)
            ref_box = cv2.boxPoints(cv2.minAreaRect(ref_contour))
            ref_box = ref_box.astype(int)
            cv2.drawContours(display_img, [ref_box], 0, (255, 0, 0), 4)
            cv2.putText(display_img, "REF", (ref_box[0][0], ref_box[0][1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            if ref_px > 0:
                scale = ref_px / ref_width_mm
                
                # Measure others
                for cnt in contours[1:]:
                    w_px = get_min_axis_width(cnt)
                    w_mm = w_px / scale
                    onion_data.append(w_mm)
                    
                    # Draw ONION (Green)
                    box = cv2.boxPoints(cv2.minAreaRect(cnt))
                    box = box.astype(int)
                    cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                    
                    # Label
                    label = f"{int(w_mm)}mm"
                    cv2.putText(display_img, label, (box[0][0], box[0][1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # --- Display ---
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Combined Mask")
            st.write("Ensure both **Onions** and **Cap** are white.")
            st.image(debug_img, channels="BGR", use_column_width=True)
        
        with col2:
            st.subheader("2. Final Grader")
            st.image(display_img, channels="BGR", use_column_width=True)
            
            if onion_data:
                st.success(f"Detected: {len(onion_data)} Onions")
                
                # Statistics
                c55 = sum(1 for x in onion_data if x >= 55)
                c65 = sum(1 for x in onion_data if x >= 65)
                
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Average Size", f"{sum(onion_data)/len(onion_data):.1f} mm")
                kpi2.metric("Grade >55mm", f"{c55} ({c55/len(onion_data):.0%})")
                kpi3.metric("Grade >65mm", f"{c65} ({c65/len(onion_data):.0%})")
            else:
                st.warning("Reference found, but no other onions detected.")

if __name__ == "__main__":
    main()
