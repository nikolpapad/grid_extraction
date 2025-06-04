import streamlit as st
from PIL import Image
import numpy as np
import base64
from streamlit_drawable_canvas import st_canvas


def classify_colors(colors):
    unique = []
    for c in colors:
        if c not in unique:
            unique.append(c)
        if len(unique) == 2:
            break
    if len(unique) < 2:
        st.error("Couldn't detect two distinct colors. Please upload a clearer image.")
    return unique


def get_label(color, color_map):
    distances = [np.linalg.norm(np.array(color) - np.array(ref)) for ref in color_map]
    return "A" if distances[0] < distances[1] else "B"


def run_length_encode(row):
    if not row:
        return ""
    result = []
    current = row[0]
    count = 1
    for c in row[1:]:
        if c == current:
            count += 1
        else:
            result.append(f"{count}{current}")
            current = c
            count = 1
    result.append(f"{count}{current}")
    return " ".join(result)


st.set_page_config(page_title="Crochet Pattern Reader")
st.title("🧶 Crochet Pattern Reader")

uploaded_file = st.file_uploader("Upload a 2-color crochet pattern (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_np = np.array(img)

    st.subheader("📐 Draw Grid Area on Image")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        background_image=img_np,
        update_streamlit=True,
        height=img.height,
        width=img.width,
        drawing_mode="rect",
        key="canvas"
    )

    coords_valid = False
    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        obj = canvas_result.json_data["objects"][-1]
        x = int(obj["left"])
        y = int(obj["top"])
        width = int(obj["width"])
        height = int(obj["height"])
        coords_valid = True
        st.success(f"Selected area: x={x}, y={y}, width={width}, height={height}")

    cols = st.number_input("Number of Columns", min_value=1, value=10)
    rows = st.number_input("Number of Rows", min_value=1, value=10)

    if st.button("Generate Instructions") and coords_valid:
        img_rgb = img.convert("RGB")
        img_np = np.array(img_rgb)
        cell_w = int(width / cols)
        cell_h = int(height / rows)

        grid_colors = []
        for row in range(rows):
            row_colors = []
            for col in range(cols):
                cx = int(x + col * cell_w + cell_w / 2)
                cy = int(y + row * cell_h + cell_h / 2)
                if cy < img_np.shape[0] and cx < img_np.shape[1]:
                    row_colors.append(tuple(img_np[cy, cx]))
            grid_colors.append(row_colors)

        flat = [color for row in grid_colors for color in row]
        color_refs = classify_colors(flat)

        instructions = []
        for row in reversed(grid_colors):
            label_row = [get_label(color, color_refs) for color in reversed(row)]
            instructions.append(run_length_encode(label_row))

        st.subheader("🧵 Crochet Row Instructions")
        output = ""
        for i, line in enumerate(instructions, 1):
            row_text = f"Row {i}: {line}"
            st.text(row_text)
            output += row_text + "\n"

        # Download link
        b64 = base64.b64encode(output.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="crochet_instructions.txt">📥 Download Instructions</a>'
        st.markdown(href, unsafe_allow_html=True)
