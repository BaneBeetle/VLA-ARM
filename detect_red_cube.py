import cv2
import numpy as np
from pathlib import Path


IMAGE_PATH = r"C:\Users\lolly\OneDrive\Desktop\Projects\VLA-ARM\images\test_scene.jpg"
OUT_PATH   = str(Path(IMAGE_PATH).with_name("test_scene_annotated.jpg"))

# Load image
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Could not load image at: {IMAGE_PATH}")

# Convert to HSV
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# Red wraps around hue=0, so use two ranges
# Tune if needed: S/V mins prevent picking up gray backgrounds
lower_red_1 = np.array([0,   100, 80], dtype=np.uint8)
upper_red_1 = np.array([10,  255, 255], dtype=np.uint8)
lower_red_2 = np.array([170, 100, 80], dtype=np.uint8)
upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)

mask1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
mask2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)
mask  = cv2.bitwise_or(mask1, mask2)

# Clean up noise
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Find contours
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

h, w = img_bgr.shape[:2]
annot = img_bgr.copy()
found = False
best = None
best_area = 0

for c in cnts:
    area = cv2.contourArea(c)
    if area < 300:  # ignore tiny specks; raise/lower if needed
        continue
    x, y, bw, bh = cv2.boundingRect(c)
    if area > best_area:
        best_area = area
        best = (x, y, bw, bh, c)
        found = True

if found:
    x, y, bw, bh, c = best
    # Draw bbox
    cv2.rectangle(annot, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
    # Centroid
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(annot, (cx, cy), 6, (255, 0, 255), -1)
    else:
        cx, cy = x + bw // 2, y + bh // 2
        cv2.circle(annot, (cx, cy), 6, (255, 0, 255), -1)

    # Overlay text
    cv2.putText(annot, f"red cube ~ area={int(best_area)}", (x, max(0, y-10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

    # Print stats
    print("Detected RED region:")
    print(f"  bbox: x={x}, y={y}, w={bw}, h={bh}")
    print(f"  centroid: ({cx}, {cy})")
    print(f"  area: {int(best_area)} pixels")
else:
    print("No red region found. Try relaxing S/V mins or lighting.")

# Save annotated output and show windows
cv2.imwrite(OUT_PATH, annot)
print(f"Annotated image saved to: {OUT_PATH}")

cv2.imshow("image", annot)
cv2.imshow("mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
