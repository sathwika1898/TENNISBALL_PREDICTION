import os
import cv2
import numpy as np
import pandas as pd

# Create output folders
os.makedirs("excel", exist_ok=True)

# Define HSV color ranges for different shot types
color_ranges = {
    "forehand": ([40, 40, 40], [80, 255, 255]),       # green
    "backhand": ([5, 100, 100], [20, 255, 255]),      # orange
    "volley": ([25, 150, 150], [35, 255, 255]),       # yellow
    "overhead": ([85, 100, 100], [100, 255, 255])     # blue
}

# Court dimensions (in meters)
court_width_m = 8.23
court_length_m = 23.77

# Directory with uploaded minimaps (jpeg files)
input_dir = "./minimaps"  # Replace with your folder path
image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpeg")])

# Initialize output list
all_data = []

# Loop through image pairs (placement and hitting position)
for idx in range(0, len(image_files), 2):
    rally_id = f"R{(idx // 2) + 1}"

    pair = image_files[idx:idx+2]
    if len(pair) < 2:
        continue

    for frame_id, image_name in enumerate(pair, start=1):
        source = "shot_placement" if "placement" in image_name else "hitting_position"
        image_path = os.path.join(input_dir, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        midline_y = h // 2

        for shot_type, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower, dtype="uint8"), np.array(upper, dtype="uint8"))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                x, y, bw, bh = cv2.boundingRect(c)
                cx = x + bw // 2
                cy = y + bh // 2
                x_m = round(cx * (court_width_m / w), 2)
                y_m = round(cy * (court_length_m / h), 2)
                player = "Djokovic (SM)" if cy > midline_y else "Opponent (Op)"

                all_data.append({
                    "rally_id": rally_id,
                    "frame_id": frame_id,
                    "player": player,
                    "shot_type": shot_type,
                    "source": source,
                    "x": cx,
                    "y": cy,
                    "x_m": x_m,
                    "y_m": y_m
                })

# Convert to DataFrame
df = pd.DataFrame(all_data)

# Save to Excel
df.to_excel("excel/time_series_shot_data.xlsx", index=False)
print("✅ Data saved to excel/time_series_shot_data.xlsx")
