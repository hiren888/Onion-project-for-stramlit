import streamlit as st
import cv2
import numpy as np
from PIL import Image

def get_contours(img_processed, min_area):
    # Find contours
    contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter by area
    clean_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return clean_contours

def get_min_axis_width(contour):
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    return min(width, height)

def process_image(image_file, sprout_kernel, min_area, block_size, c_value, invert_mode):
    # 1. Read Image
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 2. Grayscale & Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    
    # 3. Thresholding (The critical part for detection)
    # Ensure block_size is odd and > 1
    if block_size % 2 == 0: block_size += 1
    if block_size < 3: block_size = 3
    
    thresh_type = cv2.THRESH_BINARY_INV if invert_mode else cv2.THRESH_BINARY
    
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   thresh_type, block_size, c_value)

    # 4. Morphological Ops (Clean up noise/sprouts)
    kernel = np.ones((sprout_kernel, sprout_kernel), np.uint8)
    img_dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    # 5. Get Contours
    contours = get_contours(img_dilated, min_area)
    
    return img, img_dilated, contours

def main():
    st.set_page_config(page_title="Onion AI: Size Probability", layout="wide")
    st.title("🧅 Onion AI: Size Dashboard")

    # --- Sidebar ---
    st.sidebar.header("⚙️ Tuning")
    
    st.sidebar.subheader("1. Detection Sensitivity")
    block_size = st.sidebar.slider("Block Size (Coarseness)", 3, 99, 21, step=2, help="Increase this if onions have texture inside them.")
    c_value = st.sidebar.slider("Constant (Subtract)", 0, 50, 2, help="Fine tune the border cut-off.")
    invert = st.sidebar.checkbox("Invert Colors (Dark/Light)", value=True, help="Toggle this if onions are not showing up.")

    st.sidebar.subheader("2. Filtering")
    sprout_eraser = st.sidebar.slider("Sprout Eraser", 1, 20, 5)
    min_area = st.sidebar.slider("Min Area Size", 50, 5000, 500)
    
    st.sidebar.subheader("3. Reference")
    ref_width_mm = st.sidebar.number_input("Ref Cap Size (mm)", value=30.0)

    debug_mode = st.sidebar.checkbox("Show Debug View", value=True)

    # --- Main ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Process
        original_img, binary_map, contours = process_image(uploaded_file, sprout_eraser, min_area, block_size, c_value, invert)
        
        display_img = original_img.copy()
        
        # --- DEBUG VIEW ---
        if debug_mode:
            st.warning("👀 **Debug Mode:** The image below must show white onion shapes on black background. If it's all black or all white, adjust the 'Sensitivity' sliders on the left.")
            st.image(binary_map, caption="Computer Vision Mask", clamp=True, channels='GRAY')
            st.write(f"**Contours Found:** {len(contours)}")

        # --- LOGIC ---
        if len(contours) > 0:
            # Sort by area (Largest is likely the reference cap if it's placed close to camera or is big)
            # NOTE: For better UX, we assume largest contour is Reference Cap
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            ref_contour = contours[0]
            ref_pixels = get_min_axis_width(ref_contour)
            
            if ref_pixels > 0:
                scale = ref_pixels / ref_width_mm
                onion_sizes = []
                
                for cnt in contours[1:]: # Skip the first one (Reference)
                    w_px = get_min_axis_width(cnt)
                    w_mm = w_px / scale
                    onion_sizes.append(w_mm)
                    
                    # Draw
                    box = cv2.boxPoints(cv2.minAreaRect(cnt))
                    box = box.astype(int)
                    cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                    cv2.putText(display_img, f"{int(w_mm)}", (box[0][0], box[0][1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Show Final Image
                st.image(display_img, channels="BGR", caption="Detected Onions", use_column_width=True)
                
                # Stats
                if onion_sizes:
                    total = len(onion_sizes)
                    c55 = sum(1 for x in onion_sizes if x >= 55)
                    c65 = sum(1 for x in onion_sizes if x >= 65)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Count", total)
                    col2.metric("Grade >55mm", f"{c55} ({c55/total:.0%})")
                    col3.metric("Grade >65mm", f"{c65} ({c65/total:.0%})")
            else:
                st.error("Reference object invalid.")
        else:
            st.error("No objects detected. Try adjusting 'Block Size' or 'Invert Colors'.")

if __name__ == "__main__":
    main()
