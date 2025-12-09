"""
Object-oriented implementation of the grid extraction pipeline.
"""
import cv2
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from matplotlib import pyplot as plt

from utils import (
    extend_line,
    classify_cell,
    refine_line_positions,
    color_all_cells,
    rle_labels,
    generate_instructions
)


class GridExtractor:
    def __init__(self, img_path, cell_margin=5, atol=0.1, pixel_tol=5):
        self.img_path = img_path
        self.cell_margin = cell_margin
        self.atol = atol
        self.pixel_tol = pixel_tol

        self.img = None
        self.gray = None
        self.binary = None
        self.edges = None
        self.lines = None

        # line coordinates
        self.xs_raw = []
        self.ys_raw = []

        # cell output
        self.pattern_colors = []
        self.pattern_labels = []
        self.reconstructed = None
        self.debug_cells = None

        # grid boundaries
        self.grid_left = None
        self.grid_right = None
        self.grid_top = None
        self.grid_bottom = None

    # ------------------------------
    # 1. Load and preprocess image
    # ------------------------------
    def load_image(self):
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            raise ValueError("Could not read image at " + self.img_path)

    def preprocess(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        blur = cv2.GaussianBlur(blur, (5, 5), 0)

        otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        erode = cv2.erode(otsu, np.ones((5, 5), np.uint8), iterations=1)
        dilate = cv2.dilate(erode, np.ones((3, 3), np.uint8), iterations=1)

        self.binary = cv2.bitwise_not(dilate)
        self.gray = gray
        self.edges = cv2.Canny(self.binary, 100, 150, apertureSize=3)

    # -------------------------------------
    # 2. Detect lines with Hough Transform
    # --------------------------------------
    def detect_lines(self):
        self.lines = cv2.HoughLinesP(
            self.edges, 1, np.pi / 180,
            threshold=250, minLineLength=20, maxLineGap=40
        )
        if self.lines is None:
            raise RuntimeError("No lines detected")

    # ----------------------------------------------------------
    # 3. Separate horizontal/vertical lines and extract grid
    # ---------------------------------------------------------
    def extract_grid_lines(self):
        height, width = self.binary.shape
        horizontal_groups = defaultdict(list)
        vertical_groups = defaultdict(list)

        for (x1, y1, x2, y2) in tqdm(self.lines[:, 0, :], desc="Processing lines"):
            dx, dy = x2 - x1, y2 - y1
            angle = (np.degrees(np.arctan2(dy, dx)) + 180) % 180

            near_horizontal = min(angle, 180 - angle) < self.atol
            near_vertical = abs(angle - 90) < self.atol
            if not (near_horizontal or near_vertical):
                continue

            x_s, y_s, x_e, y_e = extend_line(height, width, x1, y1, x2, y2)

            if near_horizontal:
                key = int(round(y_s / self.pixel_tol))
                horizontal_groups[key].append(y_s)

            if near_vertical:
                key = int(round(x_s / self.pixel_tol))
                vertical_groups[key].append(x_s)

        self.xs_raw = refine_line_positions([int(np.median(v)) for v in vertical_groups.values()])
        self.ys_raw = refine_line_positions([int(np.median(v)) for v in horizontal_groups.values()])

        if not self.xs_raw or not self.ys_raw:
            raise RuntimeError("No valid grid lines extracted")

        self.grid_left, self.grid_right = self.xs_raw[0], self.xs_raw[-1]
        self.grid_top, self.grid_bottom = self.ys_raw[0], self.ys_raw[-1]

    # ----------------------------
    # 4. EXTRACT CELLS + COLORS
    # ----------------------------
    def extract_cells(self):
        img = self.img
        height, width = self.binary.shape

        n_cols = len(self.xs_raw) - 1
        n_rows = len(self.ys_raw) - 1

        grid_w = self.grid_right - self.grid_left
        grid_h = self.grid_bottom - self.grid_top

        self.reconstructed = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for j in tqdm(range(n_rows), desc="Extracting cells"):
            row_colors = []
            row_labels = []

            y1, y2 = self.ys_raw[j], self.ys_raw[j + 1]

            for i in range(n_cols):
                x1, x2 = self.xs_raw[i], self.xs_raw[i + 1]

                yy1 = max(y1 + self.cell_margin, 0)
                yy2 = min(y2 - self.cell_margin, height)
                xx1 = max(x1 + self.cell_margin, 0)
                xx2 = min(x2 - self.cell_margin, width)

                cell = img[yy1:yy2, xx1:xx2]

                if cell.size == 0:
                    color = np.array([255, 255, 255], dtype=np.uint8)
                    label = "white"
                else:
                    mean_color = cell.reshape(-1, 3).mean(axis=0).astype(float)
                    color, label = classify_cell(mean_color)

                row_colors.append(color)
                row_labels.append(label)

                new_y1 = y1 - self.grid_top
                new_y2 = y2 - self.grid_top
                new_x1 = x1 - self.grid_left
                new_x2 = x2 - self.grid_left

                self.reconstructed[new_y1:new_y2, new_x1:new_x2] = color

            self.pattern_colors.append(row_colors)
            self.pattern_labels.append(row_labels)

        self.debug_cells = color_all_cells(
            self.reconstructed.copy(),
            self.xs_raw, self.ys_raw, n_cols, n_rows,
            self.grid_left, self.grid_top, self.img, plotting=False
        )

    # ----------------------------
    # 5. BUILD CROCHET INSTRUCTIONS
    # ----------------------------
    def build_instructions(self, start_corner="bottom-left", alternate=False):
        # convert each row of labels → RLE
        rle_rows = [rle_labels(row) for row in self.pattern_labels]
        return generate_instructions(
            rle_rows, start_corner=start_corner, crochet=True
        )

    # ----------------------------
    # 6. VISUALIZE RESULTS
    # ----------------------------
    def show_results(self):
        plt.figure(figsize=(6, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(self.debug_cells, cv2.COLOR_BGR2RGB))
        plt.title("Cells (Debug)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(self.reconstructed, cv2.COLOR_BGR2RGB))
        plt.title("Reconstructed Pattern")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

