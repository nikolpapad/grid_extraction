''' Teleutaia prospatheia gia grid-extraction pipeline.
Kainourgio repository: https://github.com/nikolpapad/grid_extraction.git

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from utils import extend_line


img_path =r"c:\Users\nikol\Downloads\page_93.png"
img = cv2.imread(img_path)

# Gray scale + light blur to keep edges 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.GaussianBlur(gray, (5,5), 0)
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

if lines is not None:
    atol = 1.0  # angle tolerance in degrees
    pixel_tol = 5

    hor_lines = {}
    vert_lines = {}

    from collections import defaultdict
    horizontal_groups = defaultdict(list)
    vertical_groups = defaultdict(list)


    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = (x2 - x1), (y2 - y1)
        angle = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
        near_horizontal = min(angle, 180.0 - angle) < atol
        near_vertical   = abs(angle - 90.0) < atol

        if not (near_horizontal or near_vertical):
            continue

        x_start, y_start, x_end, y_end = extend_line(x1, y1, x2, y2, width, height)

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

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title("Detected Grid Lines Overlay")
    plt.show()

    ###################################################################################################

    xs_raw = sorted(v for group in vertical_groups.values() for v in group)
    ys_raw = sorted(v for group in horizontal_groups.values() for v in group)

    xs_raw = np.array(xs_raw, dtype=np.int32)
    ys_raw = np.array(ys_raw, dtype=np.int32)

    # ---- compute distances along each axis ----
    x_diffs = np.diff(xs_raw) if len(xs_raw) > 1 else np.array([])
    y_diffs = np.diff(ys_raw) if len(ys_raw) > 1 else np.array([])

    all_diffs = np.concatenate([x_diffs, y_diffs]) if len(x_diffs) + len(y_diffs) > 0 else np.array([])

    if all_diffs.size > 0:
        # Histogram
        plt.figure(figsize=(7,4))
        plt.hist(all_diffs, bins='auto', edgecolor='black')
        plt.title("Histogram of distances between detected lines")
        plt.xlabel("Distance (pixels)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        # Sorted distances
        sorted_diffs = np.sort(all_diffs)
        plt.figure(figsize=(7,4))
        plt.plot(sorted_diffs, marker='o')
        plt.title("Sorted distances")
        plt.xlabel("Index")
        plt.ylabel("Distance (pixels)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        threshold = np.median(all_diffs)  # simple separater; works because small<<big

        small = all_diffs[all_diffs < threshold]
        big   = all_diffs[all_diffs >= threshold]

        plt.figure(figsize=(7,4))
        plt.hist(small, bins='auto', alpha=0.6, label='Small distances (thickness)')
        plt.hist(big, bins='auto', alpha=0.6, label='Large distances (grid spacing)')
        plt.title("Separated distance clusters")
        plt.xlabel("Distance")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
    else:
        print("Not enough lines to compute distances.")
    print("Small distances: ", small)
    print("Large distances: ", big)

    
    #####################################################################

    # ---- Compoute average positions ----
    diffs = np.diff(xs_raw)  # distances between consecutive detections

    # Sort the diffs: early ones tend to be "small", later ones "big"
    sorted_diffs = np.sort(diffs)

    # Heuristic split: first 30% = small edges, last 70% = bigger gaps
    k = max(1, int(0.3 * len(sorted_diffs)))  # at least 1 element

    small_diffs = sorted_diffs[:k]
    big_diffs   = sorted_diffs[k:]

    thickness_est = np.median(small_diffs) if len(small_diffs) > 0 else None
    square_gap_est = np.median(big_diffs)  if len(big_diffs) > 0 else None

    print("Estimated thickness:", thickness_est)
    print("Estimated grid gap (square side):", square_gap_est)
