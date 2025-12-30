import streamlit as st
import cv2
import numpy as np

def get_contours(mask, min_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    # Sort largest to smallest
    clean_contours = sorted(clean_contours, key=cv2.contourArea, reverse=True)
    return clean_contours

def get_min_axis_width(contour):
    rect = cv2.minAreaRect(contour)
    (width, height) = rect[1]
    return min(width, height)

def main():
    st.set_page_config(page_title="Onion AI: Final Grader", layout="wide")
    st.title("🧅 Onion AI: Production Grader")
    st.markdown("### Detection Mode: 3-Tier Grading + 45mm Cutoff")

    # --- Sidebar ---
    st.sidebar.header("1. Green Reference Cap")
    c_h_min = st.sidebar.slider("Green Hue Min", 0, 179, 35)
    c_h_max = st.sidebar.slider("Green Hue Max", 0, 179, 95) 
    c_s_min = st.sidebar.slider("Green Sat Min", 0, 255, 60)
    
    st.sidebar.header("2. Red Onion Settings")
    o_h_min = st.sidebar.slider("Red Hue Min", 0, 179, 0)
    o_h_max = st.sidebar.slider("Red Hue Max", 0, 179, 179)
    o_v_min = st.sidebar.slider("Red Brightness Min", 0, 255, 50)

    st.sidebar.header("3. Display & Filtering")
    ref_width_mm = st.sidebar.number_input("Real Cap Size (mm)", value=30.0)
    min_size_mm = st.sidebar.number_input("Min Onion Size (mm)", value=45.0, help="Ignore anything smaller than this.")
    font_scale = st.sidebar.slider("Text Size", 0.5, 3.0, 1.5, step=0.1)
    min_area = st.sidebar.slider("Min Area Filter (Pixels)", 50, 5000, 200)

    # --- Upload ---
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        display_img = img.copy()

        # ==========================================
        # STEP 1: FIND REFERENCE (GREEN)
        # ==========================================
        lower_cap = np.array([c_h_min, c_s_min, 50]) 
        upper_cap = np.array([c_h_max, 255, 255])
        mask_cap = cv2.inRange(hsv, lower_cap, upper_cap)
        
        kernel = np.ones((5, 5), np.uint8)
        mask_cap = cv2.morphologyEx(mask_cap, cv2.MORPH_OPEN, kernel)
        
        cnts_cap = get_contours(mask_cap, min_area)
        
        scale = 0
        ref_found = False

        if cnts_cap:
            # Assume largest green object is Reference
            ref_contour = cnts_cap[0] 
            ref_px = get_min_axis_width(ref_contour)
            
            if ref_px > 0:
                scale = ref_px / ref_width_mm
                ref_found = True
                
                # Draw Blue Box (Reference)
                box = cv2.boxPoints(cv2.minAreaRect(ref_contour))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (255, 0, 0), 3)
                cv2.putText(display_img, "REF", (box[0][0], box[0][1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 3)

        # ==========================================
        # STEP 2: FIND ONIONS (RED)
        # ==========================================
        lower_onion = np.array([o_h_min, 50, o_v_min])
        upper_onion = np.array([o_h_max, 255, 255])
        mask_onion = cv2.inRange(hsv, lower_onion, upper_onion)
        
        mask_onion = cv2.morphologyEx(mask_onion, cv2.MORPH_CLOSE, kernel)
        mask_onion = cv2.dilate(mask_onion, kernel, iterations=1)
        
        cnts_onion = get_contours(mask_onion, min_area)
        onion_data = []

        # ==========================================
        # STEP 3: MEASURE, FILTER & DRAW
        # ==========================================
        if ref_found:
            for cnt in cnts_onion:
                w_px = get_min_axis_width(cnt)
                w_mm = w_px / scale
                
                # --- FILTER: IGNORE SMALL OBJECTS ---
                if w_mm < min_size_mm:
                    continue  # Skip to next contour
                
                onion_data.append(w_mm)
                
                # Draw Green Box
                box = cv2.boxPoints(cv2.minAreaRect(cnt))
                box = box.astype(int)
                cv2.drawContours(display_img, [box], 0, (0, 255, 0), 3)
                
                # Draw Text
                label = f"{int(w_mm)}mm"
                x, y = box[1][0], box[1][1] 
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
                cv2.rectangle(display_img, (int(x), int(y) - h - 10), (int(x) + w, int(y) + 5), (0,0,0), -1)
                cv2.putText(display_img, label, (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 3)
        
        # --- REPORTING ---
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(display_img, channels="BGR", caption="Processed Output", use_column_width=True)
            
        with col2:
            st.subheader("📊 Grading Report")
            
            if ref_found and onion_data:
                total_onions = len(onion_data)
                
                # 3-Tier Logic
                grade_large = sum(1 for x in onion_data if x >= 65)
                grade_medium = sum(1 for x in onion_data if 55 <= x < 65)
                grade_small = sum(1 for x in onion_data if x < 55) # implicitly >= 45 due to filter
                
                st.metric("Total Onions (≥45mm)", total_onions)
                st.divider()
                
                # Display in columns
                kpi1, kpi2, kpi3 = st.columns(3)
                
                kpi1.metric("Large (>65mm)", 
                            f"{grade_large}", 
                            delta=f"{grade_large/total_onions:.0%}" if total_onions else "0%")
                            
                kpi2.metric("Medium (55-64mm)", 
                            f"{grade_medium}", 
                            delta=f"{grade_medium/total_onions:.0%}" if total_onions else "0%")
                            
                kpi3.metric("Small (<55mm)", 
                            f"{grade_small}", 
                            delta=f"{grade_small/total_onions:.0%}" if total_onions else "0%")
                            
            elif not ref_found:
                st.error("Reference Cap not found. Adjust Green settings.")
            else:
                st.warning(f"No onions found larger than {min_size_mm}mm.")

if __name__ == "__main__":
    main()
