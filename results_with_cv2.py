import cv2
import numpy as np
from split_image import split_image
import matplotlib.pyplot as plt
import sys

or_image = cv2.imread("pics/2.png")

gray = cv2.cvtColor(or_image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (3, 3), 0)

thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11, 2
)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = or_image.copy()

# Loop through contours and filter squares
square_sizes = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 4:  # You can adjust this value based on grid square size
        # Approximate contour to reduce noise
        approx = cv2.approxPolyDP(cnt, 0.015 * cv2.arcLength(cnt, True), True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            # Only keep near-square shapes
            if aspect_ratio == 1:
                square_sizes.append((w,h))
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

cv2.imshow("Filtered Square Contours", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Average size
if square_sizes:
    avg_w = sum(w for w, h in square_sizes) / len(square_sizes)
    avg_h = sum(h for w, h in square_sizes) / len(square_sizes)
    print(f"\nAverage square size: {avg_w:.2f} x {avg_h:.2f} pixels")
else:
    print("No squares detected.")
    sys.exit("No squares detecteeed.")
    
avg_w = int(avg_w)
avg_h = int(avg_h)

# Draw grid overlay:
grid_img = or_image.copy()
height, width = or_image.shape[:2]
annotated = or_image.copy()

ret, binary = cv2.threshold(
    blur,
    0,                   # ignored when using OTSU
    255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
)

classification = []
for y in range(0, height, avg_h):
    row = []
    for x in range(0, width, avg_w):
        tile = binary[y:y+avg_h, x:x+avg_w]
        # Count how manu pixels are white (255) or black (0)
        white_count = cv2.countNonZero(tile)
        total_pixels = avg_h * avg_w
        label = "black" if white_count > (total_pixels / 2) else "white"
        row.append(label)
        
        color = (255, 255, 255) if label == "white" else (0, 0, 0)
        cv2.rectangle(annotated, (x, y), (x + avg_w, y + avg_h), color, thickness=-1)
    classification.append(row)

annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(8, 8))
# plt.imshow(annotated_rgb)
# plt.title("Tiles Classified via Otsu + Majority‐Vote")
# plt.axis("off")
# plt.show()

# in_x = 1 if 
in_x = 2 if width > height else 1
in_y = 1 if width > height else 2
fig, ax = plt.subplots(in_x, in_y, figsize = (12, 8))
ax[0].imshow(output)
ax[0].set_title("Filtered Square Contours")
ax[0].axis("off")

ax[1].imshow(annotated_rgb)
ax[1].set_title("Annotated (RGB with Borders/Text)")
ax[1].axis("off")

plt.tight_layout()
plt.show()
