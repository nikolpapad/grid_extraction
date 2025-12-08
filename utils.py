import fitz  # PyMuPDF
import os
from PIL import Image
import io
import math
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import cv2

def resize(img, scale = 0.8):
    new_size = (int(img.width*scale), int(img.height*scale))
    resized = img.resize(new_size, Image.LANCZOS) # high quality downsampling flter
    return resized

#  Take pdf five back images
def extractFromPDF(pdf_path, out_dir = None):
    if out_dir is None:
        out_dir = os.path.splitext(pdf_path)[0] + "_pages"
    os.makedirs(out_dir, exist_ok=True)

    with fitz.open(pdf_path) as doc:
        total_pages = len(doc)
        for i, page in tqdm(enumerate(doc, start=1), total = total_pages , desc="Extracting pages into images:"):
            out = os.path.join(out_dir, f"page_{i}.png")
            
            if os.path.exists(out):
                tqdm.write(f"Skipped existing: {out}")
                continue

            pix = page.get_pixmap(dpi=400)     # high quality
            img_bytes = pix.tobytes("png")

            img = Image.open(io.BytesIO(img_bytes))
            img.save(out, format='PNG')
            

# # pdf_path = r"C:\Users\nikol\OneDrive\Έγγραφα\Crochet_Books\C-TT006.pdf"
# # out_dir = r"C:\Users\nikol\OneDrive\Έγγραφα\Crochet_Books\deutero_imgs"
# extractFromPDF(pdf_path, out_dir)

def extend_line(height, width, x1, y1, x2, y2, SCALE=10):
    """
    Extend the segment (x1,y1)-(x2,y2) in both directions and clip to image.
    Always returns 4 ints: (x_start, y_start, x_end, y_end).
    height, width: image dimensions.
    """
    # Image dims
    h, w = height, width
    distance = SCALE * max(w, h)

    # Work with Python ints
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        # degenerate line; just clamp a single point
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        return x1, y1, x1, y1

    # Normalize direction
    ux = dx / length
    uy = dy / length

    # Extend in both directions
    p3_x = int(round(x1 - ux * distance))
    p3_y = int(round(y1 - uy * distance))
    p4_x = int(round(x2 + ux * distance))
    p4_y = int(round(y2 + uy * distance))

    # Clip to image boundaries
    def clip_point(x, y):
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        return x, y

    p3_x, p3_y = clip_point(p3_x, p3_y)
    p4_x, p4_y = clip_point(p4_x, p4_y)

    return p3_x, p3_y, p4_x, p4_y

def classify_cell(mean_color):
    """
    Classify the mean color into basic color categories.
    mean_color: tuple of (B, G, R) values.
    Returns a string representing the color category.
    Threshols:
    - <50 -> black
    - >200 and low spread -> white
    - high R, low G,B -> red
    """
    b, g, r = map(float, mean_color)
    gray_weighted_average= 0.299 * r + 0.587 * g + 0.114 * b
    spread = max(r,g,b) - min(r,g,b)

    th_black = 20
    th_white = 150
    th_gray_spread = 15 # max spread for grayish colors
    th_red = 40 # min difference R - max(G,B) to be red
    th_dark = 100 

    if gray_weighted_average < th_black: # Very dark doesn't matter the spread
        return (np.array([0,0,0], dtype=np.uint8)), "black"
    elif gray_weighted_average > th_white and spread < th_gray_spread: # very bright and low spread
        return (np.array([255,255,255], dtype=np.uint8)), "white"
    elif (r - max(g,b)) > th_red and gray_weighted_average < th_dark:
        return (np.array([0,0,153], dtype=np.uint8)), "dark_red"
    elif not "black" or "red":
        return (np.array([255,255,255], dtype=np.uint8)), "white"  # light gray treated as white
    
def refine_line_positions(raw_positions):
    """
    Take a list of raw line coordinates (e.g. xs_raw or ys_raw),
    and merge positions that are closer than the estimated line thickness.
    Returns a cleaned, sorted list of unique line positions.
    """
    if len(raw_positions) <= 1:
        return sorted(raw_positions)

    positions = np.array(sorted(raw_positions), dtype=np.float32)
    diffs = np.diff(positions)

    # If all diffs are zero-ish, just return the median
    if np.all(diffs == 0):
        return [int(np.median(positions))]

    # Sort diffs to separate "small" (thickness) vs "big" (cell gaps)
    sorted_diffs = np.sort(diffs)
    k = max(1, int(0.3 * len(sorted_diffs)))  # first 30% as "small"

    small_diffs = sorted_diffs[:k]
    if len(small_diffs) == 0:
        thickness_est = 0
    else:
        thickness_est = float(np.median(small_diffs))

    # Tolerance for grouping positions that belong to the same physical line
    # Some safety factor around the estimated thickness
    max_group_dist = max(2.0, 1.5 * thickness_est)

    merged = []
    current_group = [positions[0]]

    for p in positions[1:]:
        if abs(p - current_group[-1]) <= max_group_dist:
            current_group.append(p)
        else:
            merged.append(int(np.median(current_group)))
            current_group = [p]

    merged.append(int(np.median(current_group)))

    return sorted(merged)
# ------------------------------------------------------------------------------

def color_all_cells(reconstructed, xs_raw, ys_raw, n_cols, n_rows, grid_left, grid_top, orig ,plotting = False):
    debug_cells = np.zeros_like(reconstructed, dtype=np.uint8)

    rng = np.random.default_rng(0)  # fixed seed for reproducibility

    for j in range(n_rows):
        y1 = ys_raw[j]
        y2 = ys_raw[j + 1]

        for i in range(n_cols):
            x1 = xs_raw[i]
            x2 = xs_raw[i + 1]

            # random BGR color for this cell
            rand_color = rng.integers(0, 256, size=3, dtype=np.uint8)

            # map to grid-local coordinates
            new_y1 = y1 - grid_top
            new_y2 = y2 - grid_top
            new_x1 = x1 - grid_left
            new_x2 = x2 - grid_left

            debug_cells[new_y1:new_y2, new_x1:new_x2] = rand_color
    if plotting:
        plt.figure(figsize=(6, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        plt.title(f"Original + detected lines\n(rows={n_rows}, cols={n_cols})")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(debug_cells, cv2.COLOR_BGR2RGB))
        plt.title("Each detected cell = different color")
        plt.axis("off")
        plt.show()

    return debug_cells
