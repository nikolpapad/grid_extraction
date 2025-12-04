''' Teleutaia prospatheia gia grid-extraction pipeline.
Kainourgio repository: https://github.com/nikolpapad/grid_extraction.git

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider


img_path =r"c:\Users\nikol\Downloads\page_93.png"
img = cv2.imread(img_path)

# Gray scale + light blur to keep edges 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_blur = cv2.medianBlur(gray, 3)

#Edges detection
edges = cv2.Canny(gray_blur, 50, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Binarize with Otsu
otsu = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# plt.subplot(1,2,1), plt.imshow(gray_blur, cmap='gray'), plt.title("Blurred Gray")
# plt.subplot(1,2,2),plt.imshow(otsu, cmap='gray'), plt.title("Otsu Binarization")
# plt.show()

binary_copy = otsu.copy()
orig = img.copy() # original grayscale or color image
height, width = binary_copy.shape
maxL = int(max(height, width) )

edges = cv2.Canny(binary_copy, 50, 150, apertureSize=3)
# Standard Hough Line 
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=250, minLineLength=20, maxLineGap=30)
print(f"Detected {0 if lines is None else len(lines)} lines")

if lines is not None:
    atol = 1.0  # angle tolerance in degrees
    for x1, y1, x2, y2 in lines[:, 0, :]:
        dx, dy = (x2 - x1), (y2 - y1)
        angle = (np.degrees(np.arctan2(dy, dx)) + 180.0) % 180.0
        near_horizontal = min(angle, 180.0 - angle) < atol
        near_vertical   = abs(angle - 90.0) < atol
        # draw only near-horizontal/vertical lines
        if near_horizontal or near_vertical:
            cv2.line(orig, (x1, y1), (x2, y2), (255, 0, 0), 5)
        
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title("Detected Grid Lines Overlay")
    plt.show()