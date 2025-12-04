''' Teleutaia prospatheia gia grid-extraction pipeline.
Kainourgio repository: https://github.com/nikolpapad/grid_extraction.git

'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

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
height, width = binary_copy.shape


# Kernel width relative to image width (tune this!)
vertical_kernel_len = max(10, height // 40)  # adjust if needed

# Vertical kernel: tall and thin
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))

# Erode then dilate to keep only strong vertical segments
vertical_lines = cv2.erode(binary_copy, vertical_kernel, iterations=1)
vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)