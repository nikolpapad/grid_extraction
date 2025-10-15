# --- Imports (same) ---
import streamlit as st
import numpy as np
from PIL import Image
import sys
from streamlit_cropper import st_cropper
import pandas as pd
import cv2
import os
st.set_page_config(page_title="Crochet Grid Cropper", layout='wide')
# Initialize 
if "step" not in st.session_state:
    st.session_state.step = 0
if "base_img" not in st.session_state:
    st.session_state.base_img = None

# DEBUG_CROP = 'pics/_last_crop.png'

# if 'bg' not in st.session_state:
#     if os.path.exists(DEBUG_CROP):
#         with st.sidebar:
#             use_saved_crop = st.checkbox('Use last crop', value=True, help='Automatically load the previously saved crop')

#         if use_saved_crop:
#             st.session_state.bg = Image.open(DEBUG_CROP).convert('RGB')
#             st.session_state.step = max(st.session_state.step, 2)  #at least at step 2
# with st.sidebar:
#     if os.path.exists(DEBUG_CROP):
#         if st.button('Clear saved prop'):
#             try:
#                 os.remove(DEBUG_CROP)
#             except Exception as e:
#                 st.warning(f"Couldn't remove saved crop: {e}")
#             st.session_state.pop('bg', None)
#             st.session_state.step = 1
#             st.rerun()

# ###########################################################################################################
st.title("🧶 Crochet Grid Cropper")

#Upload
uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    st.session_state.base_img = Image.open(uploaded).convert("RGB")
    # st.image(st.session_state.base_img, caption="Uploaded photo", use_column_width=False)
if st.button("Next ▶", key="nextstep0"):
    st.session_state.step = 1
    st.query_params["step"] = "1"

#Asking user to upload an image (TODO)
SAMPLE_IMG_PATH = "pics/80.png"
# if not os.path.exists(SAMPLE_IMG_PATH):
#     st.error(f"Base image not found at {SAMPLE_IMG_PATH}")
#     st.stop()
# base_img = Image.open(SAMPLE_IMG_PATH).convert("RGB")

# --- Step 1: Crop ---
if st.session_state.step == 1:
    st.markdown("**Step 1: Crop out the background**\n\nDrag the corners to select the region containing your grid.")
    cropped = st_cropper(
        st.session_state.base_img,
        realtime_update=True,
        box_color=st.color_picker(label="Pick Color", value="#0000FF"),
        aspect_ratio=None,
        return_type="image",
        stroke_width=0.6,
    )
    st.session_state.bg = cropped

    cola, colb = st.columns([1,2])
    with cola:
        if st.button("Next ▶", key="nextstep1"):
            # if st.session_state.bg is not None:
            #     os.makedirs(os.path.dirname(DEBUG_CROP), exist_ok=True)
            #     st.session_state.bg.save(DEBUG_CROP)
            st.session_state.step = 2
            st.query_params["step"] = "2"
    with colb:
        if st.button("Start Over ⟲", key="startover1"):
            # Clear any saved crop and restart
            # if os.path.exists(DEBUG_CROP):
            #     try: os.remove(DEBUG_CROP)
            #     except Exception as e: st.warning(f"Couldn't remove saved crop: {e}")
            st.session_state.pop("bg", None)
            st.session_state.step = 1
            st.query_params["step"] = "1"
            st.rerun()

# --- Step 2: Set grid size ---
if st.session_state.step == 2:
    
    realtime_update=True,
    st.markdown("**Step 2: Choose number of rows and columns**")
    st.session_state.cols = st.number_input("Number of columns in your grid", min_value=1, value=None, step=1)
    st.session_state.rows = st.number_input("Number of rows in your grid", min_value=1, value=None, step=1)

    if st.session_state.cols is not None and st.session_state.rows is not None:
        cola, colb = st.columns([1,2])
        with cola:
            if st.button("Next ▶", key="nextstep2"):
                st.session_state.step = 3
                st.query_params["step"] = "3"
        with colb:
                if st.button("Start Over ⟲", key="startover2"):
                    st.session_state.step = 2
                    st.query_params["step"] = "2"

# --- Step 3: Classify each grid cell as black or white (robust) ---
if st.session_state.step == 3:

    st.markdown("**Step 3: Classify each grid cell as black or white**")

    # Guards
    if "bg" not in st.session_state or st.session_state.bg is None:
        st.warning("No cropped image found. Go to Step 1 or load a saved crop.")
        st.session_state.step = 0
    if "rows" not in st.session_state or "cols" not in st.session_state:
        st.info("Set rows/cols in Step 2 first.")
        st.session_state.step = 2

    img_np = np.array(st.session_state.bg)  # RGB
    h, w = img_np.shape[:2]
    rows = int(st.session_state.rows)
    cols = int(st.session_state.cols)
    # Debug prints (NEW)
    st.markdown(
        f"**Debug:** image size = `{w}×{h}` px, rows = `{rows}`, cols = `{cols}`"
    )

    # --- 1) Robust binarization (Otsu; toggle invert if your motif is inverted) ---
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 2) Exact tiling: distribute all pixels with integer edges ---
    # This avoids the "thin strip" problem from integer division.
    ys = np.linspace(0, h, rows + 1, dtype=int)  # len rows+1
    xs = np.linspace(0, w, cols + 1, dtype=int)

    # --- 3) Margin inside each cell to avoid counting grid lines ---
    # Try 8–12% of the cell size. Adjust if needed.
    inset_ratio = 0.15

    classification = []
    for y in range(rows):
        row_labels = []
        for x in range(cols):
            y0, y1 = ys[y], ys[y + 1]
            x0, x1 = xs[x], xs[x + 1]

            # Inset the ROI to avoid the grid lines on the cell borders
            ih = max(0, int((y1 - y0) * inset_ratio))
            iw = max(0, int((x1 - x0) * inset_ratio))
            yi0, yi1 = y0 + ih, y1 - ih
            xi0, xi1 = x0 + iw, x1 - iw

            # Make sure inset doesn't collapse the cell (very small cells)
            if yi1 <= yi0 or xi1 <= xi0:
                yi0, yi1 = y0, y1
                xi0, xi1 = x0, x1

            cell = binary[yi0:yi1, xi0:xi1]

            total_pixels = cell.size
            white_pixels = cv2.countNonZero(cell)
            black_pixels = total_pixels - white_pixels
            label = "white" if white_pixels > black_pixels else "black"
            row_labels.append(label)
        classification.append(row_labels)

    # --- 4) Reconstruct using the same exact edges (no cv2.rectangle needed) ---
    recon = np.full_like(img_np, 255)  # white canvas
    for y in range(rows):
        for x in range(cols):
            y0, y1 = ys[y], ys[y + 1]
            x0, x1 = xs[x], xs[x + 1]
            color = 0 if classification[y][x] == "black" else 255
            # Fill slice; broadcast to 3 channels (RGB)
            recon[y0:y1, x0:x1] = (color, color, color)

    # 4a) 
    binary_debug = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)  # convert to 3-channel for colored lines

    # Draw horizontal and vertical lines
    for y in ys:
        cv2.line(binary_debug, (0, y), (w, y), (0, 0, 255), 1)  # red lines
    for x in xs:
        cv2.line(binary_debug, (x, 0), (x, h), (0, 0, 255), 1)

    st.image(binary_debug, caption="Binary with grid overlay", use_container_width=True)


    # --- 5) Debug visuals: show binary and an overlay of misclassified cells (optional) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(st.session_state.bg, caption="Cropped Grid", use_container_width=True)
    with col2:
        st.image(binary, caption="Binary (Otsu; try invert if needed)", clamp=True, use_container_width=True)
    with col3:
        st.image(Image.fromarray(recon), caption="Reconstructed (Exact tiling + margin)", clamp=True, use_container_width=True)

    # Start over (keep URL step in sync)
    if st.button("Start Over ⟲", key="startover_step3"):
        st.session_state.step = 1
        st.query_params["step"] = "1"
