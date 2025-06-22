import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the minimap screenshot
image_path = 'swingvision_minimap.png'  # Change to your screenshot path
image = cv2.imread(image_path)

# Convert image to HSV color space (for better color detection)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define color ranges in HSV (tweak if detection is off)
color_ranges = {
    'green': ([40, 40, 40], [80, 255, 255]),
    'orange': ([5, 100, 100], [20, 255, 255]),
    'blue': ([90, 50, 50], [130, 255, 255])
}

# Detect dots and store (x, y, color)
detected_points = []

for color, (lower, upper) in color_ranges.items():
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # Mask the image to only get specific color
    mask = cv2.inRange(hsv, lower, upper)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w // 2
        cy = y + h // 2
        detected_points.append({'color': color, 'x': cx, 'y': cy})

# Convert to DataFrame
df = pd.DataFrame(detected_points)

# Plot detected points on top of image
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 6))
plt.imshow(image_rgb)
for point in detected_points:
    plt.scatter(point['x'], point['y'], label=point['color'], s=50, edgecolors='black')
plt.title("Detected Shot Placements on SwingVision Minimap")
plt.axis('off')
plt.show()

# Optional: save coordinates to CSV
df.to_csv("swingvision_shot_coordinates.csv", index=False)
