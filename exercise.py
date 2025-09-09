import os
import cv2
import re
import numpy as np
import pandas as pd
from collections import defaultdict

# Create output folder
os.makedirs("excel", exist_ok=True)

# Define HSV color ranges for shot types
color_ranges = {
    "forehand": ([40, 40, 40], [80, 255, 255]),       # green
    "backhand": ([5, 100, 100], [20, 255, 255]),      # orange
    "volley": ([25, 150, 150], [35, 255, 255]),       # yellow
    "overhead": ([85, 100, 100], [100, 255, 255])     # blue
}

# Court dimensions
court_width_m = 8.23
court_length_m = 23.77

# Folders to process
folders = ["MiniMap", "Output"]

# Initialize data
all_data = []

# Helper function to extract rally number
def extract_rally_number(rally_id):
    match = re.search(r'\d+', rally_id)
    return int(match.group()) if match else 99999  # non-matching go to end

for folder in folders:
    input_dir = f"./{folder}"
    image_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpeg")])

    # Group files by rally ID (everything before first "_")
    rallies = defaultdict(list)
    for f in image_files:
        rally_id = f.split("_")[0]  # e.g., "R10" or "Rally10"
        rallies[rally_id].append(f)

    # Process each rally
    for rally_id, files in sorted(rallies.items(), key=lambda x: extract_rally_number(x[0])):
        rally_num = extract_rally_number(rally_id)
        if rally_num == 99999:
            print(f"⚠️ Warning: Rally ID not numeric -> {rally_id}")

        for frame_id, image_name in enumerate(files, start=1):
            source = "shot_placement" if "placement" in image_name else "hitting_position"
            image_path = os.path.join(input_dir, image_name)
            image = cv2.imread(image_path)

            if image is None:
                print(f"⚠️ Skipped unreadable image: {image_name}")
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

# Create final DataFrame
df = pd.DataFrame(all_data)

# Save to Excel
output_path = "excel/time_series_shot_data_all_new1.xlsx"
df.to_excel(output_path, index=False)

print(f"✅ Combined dataset saved to: {output_path}")
print(f"📊 Unique rallies processed: {df['rally_id'].nunique()}")
