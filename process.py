''' Teleutaia prospatheia gia grid-extraction pipeline.
Kainourgio repository: https://github.com/nikolpapad/grid_extraction.git

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from utils import extend_line, classify_cell


# img_path =r"c:\Users\nikol\Downloads\page_102.png"
img_path =r"C:\Users\nikol\Downloads\page_5.png"
img = cv2.imread(img_path)

# Gray scale + light blur to keep edges 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur1 = cv2.GaussianBlur(gray, (5,5), 0)
gray_blur = cv2.GaussianBlur(gray_blur1, (5,5), 0)

otsu = cv2.threshold(
    gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]  # Binarize with Otsu

erosion = cv2.erode(otsu, np.ones((5, 5), np.uint8), iterations=1)
dilation = cv2.dilate(erosion, np.ones((3, 3), np.uint8), iterations=1)
binary_copy = cv2.bitwise_not(dilation)  # Invert: grid lines are white

orig = img.copy() # original grayscale or color image
height, width = binary_copy.shape

edges = cv2.Canny(binary_copy, 100, 150, apertureSize=3)
# Standard Hough Line 
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=250, minLineLength=20, maxLineGap=40)
print(f"Detected {0 if lines is None else len(lines)} lines")
blank = np.zeros((512,512,3), np.uint8)

if lines is None:
    raise RuntimeError("No lines detected. Tune parameters or preprocessing.")
# ----------- Separate lines into horizontal and vertical + cluster -------------
atol = 0.1  # angle tolerance in degrees
pixel_tol = 5

hor_lines = {}
vert_lines = {}

from collections import defaultdict
horizontal_groups = defaultdict(list)
vertical_groups = defaultdict(list)

for x1, y1, x2, y2 in lines[:, 0, :]:
    # Check if angle is straight enough
    dx, dy = (x2 - x1), (y2 - y1)
    angle = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
    near_horizontal = min(angle, 180.0 - angle) < atol
    near_vertical   = abs(angle - 90.0) < atol

    if not (near_horizontal or near_vertical):
        continue

    x_start, y_start, x_end, y_end = extend_line(height, width, x1, y1, x2, y2)

    if near_horizontal:
        key = int(round(y_start / pixel_tol))
        if key not in hor_lines:
            hor_lines[key] = (x_start, y_start, x_end, y_end)
            horizontal_groups[key].append(y_start)
    elif near_vertical:
        key = int(round(x_start / pixel_tol))
        if key not in vert_lines:
            vert_lines[key] = (x_start, y_start, x_end, y_end)
            vertical_groups[key].append(x_start)

    for (x_start, y_start, x_end, y_end) in list(hor_lines.values()) + list(vert_lines.values()):
        cv2.line(orig, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)

## Original image + detected lines overlay
# plt.figure(figsize=(10, 10))
# plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
# plt.title("Detected Grid Lines Overlay")

###################################################################################################

# # Get one unique position per line (median of grouped positions)
# ys_raw = sorted(int(np.median(vals)) for vals in horizontal_groups.values())
# xs_raw = sorted(int(np.median(vals)) for vals in vertical_groups.values())

# ys_raw = np.array(ys_raw, dtype=np.int32)
# xs_raw = np.array(xs_raw, dtype=np.int32)

# # ---- compute distances along each axis ----
# x_diffs = np.diff(xs_raw) if len(xs_raw) > 1 else np.array([])
# y_diffs = np.diff(ys_raw) if len(ys_raw) > 1 else np.array([])

# all_diffs = np.concatenate([x_diffs, y_diffs]) if len(x_diffs) + len(y_diffs) > 0 else np.array([])
# plot_diff = False

# if all_diffs.size > 0 and plot_diff == True:
#     # Sorted distances
#     sorted_diffs_x = np.sort(x_diffs)
#     sorted_diffs_y = np.sort(y_diffs)
#     plt.figure(figsize=(7,4))
#     # Plot X-axis distances (vertical grid spacing)
#     plt.subplot(1, 2, 1)
#     plt.plot(sorted_diffs_x, marker='o')
#     plt.title("Sorted distances on X axis (Vertical lines)")
#     plt.xlabel("Index")
#     plt.ylabel("Distance (pixels)")
#     plt.grid(True)

#     # Plot Y-axis distances (horizontal grid spacing)
#     plt.subplot(1, 2, 2)
#     plt.plot(sorted_diffs_y, marker='o')
#     plt.title("Sorted distances on Y axis (Horizontal lines)")
#     plt.xlabel("Index")
#     plt.ylabel("Distance (pixels)")
#     plt.grid(True)

#     threshold = np.median(all_diffs)  # simple separater; works because small<<big

#     small = all_diffs[all_diffs < threshold]
#     big   = all_diffs[all_diffs >= threshold]

#     plt.figure(figsize=(7,4))
#     plt.hist(small, bins='auto', alpha=0.6, label='Small distances (thickness)')
#     plt.hist(big, bins='auto', alpha=0.6, label='Large distances (grid spacing)')
#     plt.title("Separated distance clusters")
#     plt.xlabel("Distance")
#     plt.ylabel("Count")
#     plt.legend()
#     plt.show()
#     print("Small distances: ", small)
#     print("Large distances: ", big)
# else:
#     print("Not enough lines to compute distances.")



# #####################################################################
# # TODO ----> Check if the heuristic separation works well in practice
# # ---- Compoute average positions ----
# def estimate_thickness_and_gap(xs_raw):
#     diffs = np.diff(xs_raw)  # distances between consecutive detections

#     # Sort the diffs: early ones tend to be "small", later ones "big"
#     sorted_diffs = np.sort(diffs)

#     # Heuristic split: first 30% = small edges, last 70% = bigger gaps
#     k = max(1, int(0.3 * len(sorted_diffs)))  # at least 1 element

#     small_diffs = sorted_diffs[:k]
#     big_diffs   = sorted_diffs[k:]

#     thickness_est = np.median(small_diffs) if len(small_diffs) > 0 else None
#     square_gap_est = np.median(big_diffs)  if len(big_diffs) > 0 else None

#     print("Estimated thickness:", thickness_est)
#     print("Estimated grid gap (square side):", square_gap_est)
# #####################################################
    
# def recreate_grid_lines(xs_raw, ys_raw):
#     # Recreate grid lines on original image
#     x_diffs = np.diff(xs_raw)
#     y_diffs = np.diff(ys_raw)

#     # Use median spacing as the "true" spacing
#     x_step = np.median(x_diffs)
#     y_step = np.median(y_diffs)

#     # Optional: snap each line to the nearest multiple of the step
#     x0 = xs_raw[0]
#     y0 = ys_raw[0]

#     xs_snapped = [int(round(x0 + i * x_step)) for i in range(len(xs_raw))]
#     ys_snapped = [int(round(y0 + j * y_step)) for j in range(len(ys_raw))]
#     grid_img = np.ones_like(gray) * 255  # white background

#     for x in xs_snapped:
#         cv2.line(grid_img, (x, ys_snapped[0]), (x, ys_snapped[-1]), 0, 1)

#     for y in ys_snapped:
#         cv2.line(grid_img, (xs_snapped[0], y), (xs_snapped[-1], y), 0, 1)

#     plt.imshow(grid_img, cmap='inferno')
#     plt.title("Recreated grid")
#     plt.show()
#######################################################################################################

xs_raw = sorted(int(np.median(group)) for group in vertical_groups.values()) # unique x for each vertical line
ys_raw = sorted(int(np.median(group)) for group in horizontal_groups.values())
# recreate_grid_lines(xs_raw, ys_raw)


grid_left,grid_right = xs_raw[0], xs_raw[-1]
grid_top, grid_bottom = ys_raw[0], ys_raw[-1]

grid_width  = grid_right - grid_left
grid_height = grid_bottom - grid_top

print(f"Grid bounding box: x=[{grid_left}, {grid_right}], y=[{grid_top}, {grid_bottom}]")

n_cols = len(xs_raw) - 1 # for 3 vertical lines -> 2 columns
n_rows = len(ys_raw) - 1

# image where we'll reconstruct the pattern inside the grid only
reconstructed = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

# margin to avoid sampling the black grid lines
cell_margin = 5

# store per-cell colors if you also want the data structure
pattern_colors = []

for j in range(n_rows):
    row_colors = []
    y1 = ys_raw[j]
    y2 = ys_raw[j + 1]

    for i in range(n_cols):
        x1 = xs_raw[i]
        x2 = xs_raw[i + 1]

        # ----- Crop original cell (inside the grid) -----
        yy1 = max(y1 + cell_margin, 0)
        yy2 = min(y2 - cell_margin, height)
        xx1 = max(x1 + cell_margin, 0)
        xx2 = min(x2 - cell_margin, width)

        cell = img[yy1:yy2, xx1:xx2]

        if cell.size == 0:
            mean_color = np.array([255, 255, 255], dtype=np.uint8)
        else:
            mean_color = cell.reshape(-1, 3).mean(axis=0).astype(np.uint8)
            color, clabel = classify_cell(mean_color)
            if clabel == "other": 
                print(f"Cell ({i},{j}) mean color: {mean_color}, classified as {clabel}")
                
            
        row_colors.append(color)

        # ----- Paint that cell area in the reconstructed image -----
        new_y1 = y1 - grid_top
        new_y2 = y2 - grid_top
        new_x1 = x1 - grid_left
        new_x2 = x2 - grid_left

        reconstructed[new_y1:new_y2, new_x1:new_x2] = color

    pattern_colors.append(row_colors)

print(f"Grid cells: {n_rows} rows x {n_cols} cols")

print(f"Cell {cell.size}")


plt.figure(figsize=(4, 4))
plt.subplot(1, 2, 1),plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)),plt.title("Original grid area"),plt.axis("off")
plt.subplot(1, 2, 2),plt.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB)),plt.title("Reconstructed pattern"),plt.axis("off")
# plt.show()


