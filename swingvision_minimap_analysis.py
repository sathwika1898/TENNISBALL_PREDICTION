import cv2
import numpy as np
import pandas as pd
import os

# Create folders
os.makedirs("output", exist_ok=True)
os.makedirs("excel", exist_ok=True)

# Input image paths
images = {
    "shot_placement": "MiniMap/",
    "hitting_position": "MiniMap/"
}

# HSV color ranges
color_ranges = {
    "forehand": ([40, 40, 40], [80, 255, 255]),       # green
    "backhand": ([5, 100, 100], [20, 255, 255]),      # orange
    "volley": ([25, 150, 150], [35, 255, 255]),       # yellow
    "overhead": ([85, 100, 100], [100, 255, 255])     # blue
}

# Detect shot dots
def detect_shots(image_path, shot_type_label):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    detected = []

    for shot_type, (lower, upper) in color_ranges.items():
        lower_np = np.array(lower, dtype="uint8")
        upper_np = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower_np, upper_np)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            cx = x + w // 2
            cy = y + h // 2
            detected.append({
                "shot_type": shot_type,
                "x": cx,
                "y": cy,
                "source": shot_type_label
            })

    return detected

# Process both images
all_detections = []
for label, path in images.items():
    all_detections.extend(detect_shots(path, label))

df = pd.DataFrame(all_detections)

# Convert to real-world coordinates
image_sample = cv2.imread(images["shot_placement"])
image_height_px, image_width_px = image_sample.shape[:2]

court_width_m = 8.23
court_length_m = 23.77

def convert_to_meters(x_px, y_px):
    x_m = x_px * (court_width_m / image_width_px)
    y_m = y_px * (court_length_m / image_height_px)
    return round(x_m, 2), round(y_m, 2)

df["x_m"], df["y_m"] = zip(*df.apply(lambda row: convert_to_meters(row["x"], row["y"]), axis=1))

# Label player side
midline_y = image_height_px // 2
df["player"] = df["y"].apply(lambda y: "Djokovic (SM)" if y > midline_y else "Opponent (Op)")

# Save to Excel
df.to_excel("excel/djokovic_shot_data_labeled.xlsx", index=False)
