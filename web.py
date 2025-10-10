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

# Initialize step
if "step" not in st.session_state:
    st.session_state.step = 1

# # Read step from URL if present (so refresh keeps your place)
# params = st.query_params
# if "step" in params and str(params["step"]).isdigit():
#     st.session_state.step = int(params["step"])

st.title("🧶 Crochet Grid Cropper")

# --- Sidebar controls (NEW: debug helpers) ---
with st.sidebar:
    box_color = st.color_picker(label="Box Color", value="#0000FF")
    st.markdown("### 🛠️ Debug")
    # save_cells = st.checkbox("Save individual cell images", value=False)
    invert = st.checkbox("Invert binary (if needed)", value=False)
    jump = st.selectbox("Jump to step", [1, 2, 3], index=st.session_state.step - 1)
    if jump != st.session_state.step:
        st.session_state.step = jump
        st.query_params["step"] = str(st.session_state.step)

# Your original sample image (keep it)
SAMPLE_IMG_PATH = "pics/wow.png" # TODO ask from user

# Ensure 
if not os.path.exists(SAMPLE_IMG_PATH):
    st.error(f"Base image not found at {SAMPLE_IMG_PATH}")
    st.stop()

base_img = Image.open(SAMPLE_IMG_PATH).convert("RGB")

# # If a debug crop exists and we don't already have bg in memory, preload it and skip to Step 2
# if os.path.exists(DEBUG_CROP) and "bg" not in st.session_state:
#     st.session_state.bg = Image.open(DEBUG_CROP).convert("RGB")
#     st.session_state.step = max(st.session_state.step, 2)
#     st.query_params["step"] = str(st.session_state.step)

# --- Step 1: Crop ---
if st.session_state.step == 1:
    st.markdown("**Step 1: Crop out the background**\n\nDrag the corners to select the region containing your grid.")
    st.session_state.bg = st_cropper(
        base_img,
        realtime_update=True,
        box_color=box_color,
        aspect_ratio=None,
        return_type="image",
        stroke_width=0.8,
    )

    cola, colb = st.columns([1,2])
    with cola:
        if st.button("Next ▶", key="nextstep1"):
            st.session_state.step = 2
            st.query_params["step"] = "2"
    with colb:
        if st.button("Start Over ⟲", key="startover1"):
            st.session_state.step = 1
            st.query_params["step"] = "1"

# --- Step 2: Set grid size ---
if st.session_state.step == 2:
    st.markdown("**Step 2: Choose number of rows and columns**")
    st.session_state.cols = st.number_input("Number of columns in your grid", min_value=1, value=70, step=1)
    st.session_state.rows = st.number_input("Number of rows in your grid", min_value=1, value=70, step=1)

    # Debug info (NEW)
    st.markdown(
        f"**Debug:** rows = `{int(st.session_state.rows)}`, cols = `{int(st.session_state.cols)}`"
    )

    colA, colB = st.columns(2)
    with colA:
        if st.button("Next ▶", key="nextstep2"):
            st.session_state.step = 3
            st.query_params["step"] = "3"
    # with colB:
    #     if st.button("Start Over ⟲", key="startover2"):
    #         st.session_state.step = 1
    #         st.query_params["step"] = "1"

# --- Step 3: Classify each grid cell as black or white (robust) ---

    st.markdown("**Step 3: Classify each grid cell as black or white**")

    # Guards
    if "bg" not in st.session_state or st.session_state.bg is None:
        st.warning("No cropped image found. Go to Step 1 or load a saved crop.")
        st.stop()
    if "rows" not in st.session_state or "cols" not in st.session_state:
        st.info("Set rows/cols in Step 2 first.")
        st.stop()

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

    # # If you added an "invert" checkbox in the sidebar:
    # invert = st.session_state.get("invert", False)                    # To krataw gia pio meta to invert
    # if invert:
    #     binary = cv2.bitwise_not(binary)

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

    # Draw horizontal lines
    for y in ys:
        cv2.line(binary_debug, (0, y), (w, y), (0, 0, 255), 1)  # red lines

    # Draw vertical lines
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
