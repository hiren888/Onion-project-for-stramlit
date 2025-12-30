import streamlit as st
import cv2
import numpy as np

def get_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Filter out small noise
    clean_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    # Sort largely to small (helps in picking the main reference object)
    clean_contours = sorted(clean_contours, key=cv2.contourArea, reverse=True)
    return clean_contours

def get_min_axis_width(contour):
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    return min(width, height)

def main():
    st.set_page_config(page_title="Onion AI: Final Pro", layout="wide")
    st.title("🧅 Onion AI: Professional Grader")
    st.markdown("### detection Mode: Strict Color Separation")
    st.write("The app now scans for **Green** (Reference) and **Red** (Onions) completely separately.")

    # --- Sidebar ---
    st.sidebar.header("1. Green Reference Cap")
    # Tweak these if the cap isn't detected
    c_h_min = st.sidebar.slider("Green Hue Min", 0, 179, 35)
    c_h_max = st.sidebar.slider("Green Hue Max", 0, 179, 95) 
    c_s_min = st.sidebar.slider("Green Sat Min", 0, 255, 60)
    
    st.sidebar.header("2. Red Onion Settings")
    o_h_min = st.sidebar.slider("Red Hue Min", 0, 179, 0)
    o_h_max = st.sidebar.slider("Red Hue Max", 0, 179, 179)
    o_v_min = st.sidebar.slider("Red Brightness Min", 0, 255, 50) # Lower this if onions are dark

    st.sidebar.header("3. Display & Size")
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)
    font_scale = st.sidebar.slider("Text Size", 0.5, 3.0, 1.3, step=0.1)
    min_area = st.sidebar.slider("Min Area Filter", 50, 5000, 200)

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        display_img = img.copy()

        # ==========================================
        # STEP 1: FIND THE REFERENCE (GREEN ONLY)
        # ==========================================
        lower_cap = np.array([c_h_min, c_s_min, 50]) # Fixed value min 50
        upper_cap = np.array([c_h_max, 255, 255])
        mask_cap = cv2.inRange(hsv, lower_cap, upper_cap)
        
        # Clean noise
        kernel = np.ones((5, 5), np.uint8)
        mask_cap = cv2.morphologyEx(mask_cap, cv2.MORPH_OPEN, kernel)
        
        cnts_cap = get_contours(mask_cap, min_area)
        
        scale = 0
        ref_found = False

        if cnts_cap:
            # We assume the LARGEST green object is the cap (ignores tiny green specs)
            ref_contour = cnts_cap[0] 
            ref_px = get_min_axis_width(ref_contour)
            
            if ref_px > 0:
                scale = ref_px / ref_width_mm
                ref_found = True
                
                # Draw Blue Box for Reference
                box = cv2.boxPoints(cv2.minAreaRect(ref_contour))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (255, 0, 0), 3)
                
                # Label "REF"
                cv2.putText(display_img, "REF", (box[0][0], box[0][1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 3)

        # ==========================================
        # STEP 2: FIND THE ONIONS (RED ONLY)
        # ==========================================
        lower_onion = np.array([o_h_min, 50, o_v_min]) # Fixed sat min 50
        upper_onion = np.array([o_h_max, 255, 255])
        mask_onion = cv2.inRange(hsv, lower_onion, upper_onion)
        
        # Fill holes
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel)
        mask_onion = cv2.dilate(mask_onion, kernel, iterations=1)
        
        cnts_onion = get_contours(mask_onion, min_area)
        onion_data = []

        # ==========================================
        # STEP 3: MEASURE & DRAW
        # ==========================================
        if ref_found:
            for cnt in cnts_onion:
                # Check if this contour overlaps with the reference (avoid detecting the cap as an onion)
                # Simple check: distance between centers or just trust the color masks are different
                
                w_px = get_min_axis_width(cnt)
                w_mm = w_px / scale
                onion_data.append(w_mm)
                
                # Draw Green Box
                box = cv2.boxPoints(cv2.minAreaRect(cnt))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (0, 255, 0), 3)
                
                # LARGE TEXT LABEL
                label = f"{int(w_mm)}mm"
                # Position for text
                x, y = box[1][0], box[1][1] 
                
                # Draw black background for text so it's readable
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
                cv2.rectangle(display_img, (int(x), int(y) - h - 10), (int(x) + w, int(y) + 5), (0,0,0), -1)
                
                # Draw white text
                cv2.putText(display_img, label, (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
        
        # --- Display Section ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("1. System Vision")
            st.caption("Top: Red Mask (Onions) | Bottom: Green Mask (Ref)")
            # Show both masks to help user debug
            st.image(mask_onion, caption="Red Mask", channels='GRAY', use_column_width=True)
            st.image(mask_cap, caption="Green Mask", channels='GRAY', use_column_width=True)
            
        with col2:
            st.subheader("2. Final Result")
            st.image(display_img, channels="BGR", use_column_width=True)
            
            if ref_found:
                if onion_data:
                    st.success(f"Detected {len(onion_data)} Onions")
                    c55 = sum(1 for x in onion_data if x >= 55)
                    c65 = sum(1 for x in onion_data if x >= 65)
                    st.metric("Grade A (>55mm)", f"{c55} ({c55/len(onion_data):.0%})")
                    st.metric("Super (>65mm)", f"{c65} ({c65/len(onion_data):.0%})")
                else:
                    st.warning("Reference found, but no onions detected. Adjust 'Red Onion Settings'.")
            else:
                st.error("❌ Reference Cap NOT found. Adjust 'Green Reference' sliders.")

if __name__ == "__main__":
    main()
