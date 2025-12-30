import streamlit as st
import cv2
import numpy as np
from PIL import Image

def get_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return clean_contours

def get_min_axis_width(contour):
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    return min(width, height)

def main():
    st.set_page_config(page_title="Onion AI: Color Tuner", layout="wide")
    st.title("🧅 Onion AI: Color Tuning Dashboard")
    st.write("Use the sliders to isolate the onions from the background.")

    # --- Sidebar: Color Settings (HSV) ---
    st.sidebar.header("🎨 Color Settings")
    
    # Defaults set to capture Red/Purple onions comfortably
    h_min = st.sidebar.slider("Hue Min", 0, 179, 0)
    h_max = st.sidebar.slider("Hue Max", 0, 179, 179)
    s_min = st.sidebar.slider("Saturation Min", 0, 255, 0)
    s_max = st.sidebar.slider("Saturation Max", 0, 255, 255)
    v_min = st.sidebar.slider("Value (Brightness) Min", 0, 255, 0)
    v_max = st.sidebar.slider("Value (Brightness) Max", 0, 255, 255)

    st.sidebar.header("🛠️ Cleanup Settings")
    # This helps fill the "hollow" centers seen in your image
    morph_size = st.sidebar.slider("Hole Filler (Kernel)", 1, 20, 5)
    min_area = st.sidebar.slider("Min Area Filter", 100, 5000, 500)
    
    ref_width_mm = st.sidebar.number_input("Reference Cap Size (mm)", value=30.0)

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Convert to CV2
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 1. Create Mask based on Sliders
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 2. Hole Filling (Morphology) - Fixes the "hollow onion" issue
        kernel = np.ones((morph_size, morph_size), np.uint8)
        # Dilate then Erode (Closing) fills small holes
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Extra dilation to solidify edges
        mask_clean = cv2.dilate(mask_clean, kernel, iterations=1)

        contours = get_contours(mask_clean, min_area)
        
        # --- Visualization ---
        display_img = img.copy()
        
        # Draw all contours found
        onion_data = []
        
        # Find Reference (Largest Area)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            ref_contour = contours[0]
            ref_px = get_min_axis_width(ref_contour)
            
            if ref_px > 0:
                scale = ref_px / ref_width_mm
                
                for cnt in contours[1:]:
                    w_px = get_min_axis_width(cnt)
                    w_mm = w_px / scale
                    onion_data.append(w_mm)
                    
                    # Draw
                    box = cv2.boxPoints(cv2.minAreaRect(cnt))
                    box = box.astype(int)
                    cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                    cv2.putText(display_img, f"{int(w_mm)}mm", (box[0][0], box[0][1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- Layout Results ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. Tune this Mask")
            st.write("Adjust sliders until onions are **Solid White** and background is **Black**.")
            st.image(mask_clean, channels='GRAY', use_column_width=True)
            
        with col2:
            st.subheader("2. Final Result")
            st.image(display_img, channels="BGR", use_column_width=True)
            
            if onion_data:
                total = len(onion_data)
                count_55 = sum(1 for x in onion_data if x >= 55)
                count_65 = sum(1 for x in onion_data if x >= 65)
                
                st.info(f"Detected: {total} Onions (Ref Cap excluded)")
                st.write(f"**≥ 55mm:** {count_55} ({count_55/total:.1%})")
                st.write(f"**≥ 65mm:** {count_65} ({count_65/total:.1%})")

if __name__ == "__main__":
    main()
