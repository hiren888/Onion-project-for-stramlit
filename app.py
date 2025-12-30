import streamlit as st
import cv2
import numpy as np
from PIL import Image

def get_contours(img_processed, min_area):
    """
    Finds contours in the processed binary image.
    Filters out small noise based on min_area.
    """
    contours, _ = cv2.findContours(img_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    return clean_contours

def get_min_axis_width(contour):
    """
    Calculates the 'Min Axis' (Width) of the rotated rectangle.
    This effectively ignores the length of the onion (sprouts).
    """
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    # The smaller dimension is the width (diameter)
    return min(width, height)

def process_image(image_file, sprout_eraser_kernel, min_area_threshold):
    # Convert PIL image to OpenCV format
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1)
    
    # Adaptive Thresholding for segmentation
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations (Sprout Eraser)
    kernel = np.ones((sprout_eraser_kernel, sprout_eraser_kernel), np.uint8)
    img_dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours = get_contours(img_dilated, min_area_threshold)
    
    return img, contours

def main():
    st.set_page_config(page_title="Onion AI: Size Probability Dashboard", layout="wide")
    
    st.title("🧅 Onion AI: Size Probability Dashboard")
    st.write("Upload a photo of your onion lot with the **Green Cap** reference to get grading probabilities.")

    # --- Sidebar Settings ---
    st.sidebar.header("⚙️ Detection Settings")
    
    ref_width_mm = st.sidebar.number_input("Reference Cap Size (mm)", value=30.0)
    sprout_eraser = st.sidebar.slider("Sprout Eraser Size (Kernel)", 1, 15, 3)
    min_area = st.sidebar.slider("Min Contour Area", 100, 5000, 1000)

    # --- Upload Section ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        # Process Image
        original_img, contours = process_image(uploaded_file, sprout_eraser, min_area)
        
        # Draw on image for visualization
        display_img = original_img.copy()
        
        # 1. Find Reference Object (Green Cap) logic would go here
        # For this version, we assume the LARGEST contour is the reference
        # OR the user clicks/selects. To keep it automated as per history:
        # We will assume the Reference is the contour with the highest circularity or specific color.
        # *Simplification for this script:* We take the first contour as reference to scale.
        
        if len(contours) > 0:
            # Sort contours by area to find significant objects
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Assume Reference is the largest green-ish object (simplified here as largest area for code stability)
            ref_contour = contours[0] 
            ref_pixel_width = get_min_axis_width(ref_contour)
            
            if ref_pixel_width > 0:
                pixels_per_mm = ref_pixel_width / ref_width_mm
                
                onion_sizes = []
                
                # Iterate through remaining contours (Onions)
                for cnt in contours[1:]:
                    w_pixels = get_min_axis_width(cnt)
                    w_mm = w_pixels / pixels_per_mm
                    onion_sizes.append(w_mm)
                    
                    # Draw bounding box
                    box = cv2.boxPoints(cv2.minAreaRect(cnt))
                    box = np.int0(box)
                    cv2.drawContours(display_img, [box], 0, (0, 255, 0), 2)
                    cv2.putText(display_img, f"{int(w_mm)}mm", (box[0][0], box[0][1]), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # --- Display Results ---
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.image(display_img, channels="BGR", caption="Processed Lot", use_column_width=True)
                
                with col2:
                    st.subheader("📊 Probability Report")
                    
                    if len(onion_sizes) > 0:
                        count_55 = sum(1 for x in onion_sizes if x >= 55)
                        count_65 = sum(1 for x in onion_sizes if x >= 65)
                        total = len(onion_sizes)
                        
                        st.metric("Total Onions Detected", total)
                        
                        st.markdown("### Grade Probability")
                        st.progress(count_55 / total)
                        st.write(f"**Probability $\ge$ 55mm:** {count_55/total:.1%} ({count_55} onions)")
                        
                        st.progress(count_65 / total)
                        st.write(f"**Probability $\ge$ 65mm:** {count_65/total:.1%} ({count_65} onions)")
                        
                        st.markdown("---")
                        st.write("*Note: Measurements use Min-Axis logic to ignore sprouts.*")
                    else:
                        st.warning("Reference found, but no onions detected.")
            else:
                st.error("Could not determine reference width.")
        else:
            st.error("No contours found. Try adjusting the Min Contour Area.")

if __name__ == "__main__":
    main()
