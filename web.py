import streamlit as st
import numpy as np 
from PIL import Image
import sys
from streamlit_cropper import st_cropper
import pandas as pd
import cv2

st.set_page_config(page_title="Crochet Grid Cropper", layout='wide')
if 'step' not in st.session_state:
    st.session_state.step = 1

st.title("🧶 Crochet Grid Cropper")
box_color = st.sidebar.color_picker(label="Box Color", value='#0000FF')
#Upload pic
uploaded = st.file_uploader("Upload your crochet-grid image", type=["png", "jpg", "jpeg"])
if not uploaded:
    st.info("Loukaki pelase add a picture to get started")
    st.stop()
img = Image.open(uploaded)

if st.session_state.step == 1:
    st.markdown("**Step 1: Crop out the background**\n\nDrag the corners to select the region containing your grid.")
    st.session_state.bg = st_cropper(
        img, 
        realtime_update=True, 
        box_color=box_color, 
        aspect_ratio=None, 
        return_type="image",
        stroke_width = 1
    )
    if st.button("Next"):
        st.session_state.step = 2

if st.session_state.step == 2:
    st.markdown("**Step 2: Choose number of rows and columns**")
    st.session_state.cols = st.number_input("Number of columns in your grid", min_value=1, value=250, step=1)
    st.session_state.rows = st.number_input("Number of rows in your grid",    min_value=1, value=250, step=1)
    if st.button("Next"):
        st.session_state.step = 3
    
if st.session_state.step == 3:
    st.markdown("**Step 3: Pick a single grid cell**")
    
    st.session_state.img_grid = np.array(st.session_state.bg)
    gray = cv2.cvtColor(st.session_state.img_grid,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, binary = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)


    # Let the user tell us the grid dimensions:
    cols = st.session_state.cols
    rows = st.session_state.rows

    h, w = st.session_state.img_grid.shape[:2]
    cell_h, cell_w = h // rows, w // cols

    classification = []
    for y in range(0, h, cell_h):
        row = []
        for x in range(0, w, cell_w):
            tile = binary[y:y+cell_h, x:x+cell_w]
            # Count how manu pixels are white (255) or black (0)
            white_count = cv2.countNonZero(tile)
            total_pixels = cell_w * cell_h
            label = "black" if white_count > int(0.6*total_pixels) else "white"
            row.append(label)
            
            color = (255, 255, 255) if label == "white" else (0, 0, 0)
            overlay = st.session_state.img_grid.copy()
            cv2.rectangle(overlay, (x, y), (x + cell_w, y + cell_h), color, thickness=2)
        classification.append(row)


    reconstructed = np.ones_like(st.session_state.img_grid) * 255  # white background
    for y in range(rows):
        for x in range(cols):
            label = classification[y][x]
            color = (0, 0, 0) if label == "black" else (255, 255, 255)
            top_left = (x * cell_w, y * cell_h)
            bottom_right = ((x + 1) * cell_w, (y + 1) * cell_h)
            cv2.rectangle(reconstructed, top_left, bottom_right, color, thickness=-1)



    reconstructed_pil = Image.fromarray(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
    original_pil = st.session_state.img_grid

    col1, col2 = st.columns(2)
    with col1:
        st.image(original_pil, caption="Original Grid", use_container_width=True)
    with col2:
        
        st.image(reconstructed_pil, caption="Reconstructed Grid", use_container_width=True)


    # # Optional: download
    # buf = None
    # if st.button("Download this cell"):
    #     import io
    #     buf = io.BytesIO()
    #     cell_img.save(buf, format="PNG")
    #     st.download_button("⬇️ Download PNG", buf.getvalue(), file_name=f"cell_R{sel_row}_C{sel_col}.png")




