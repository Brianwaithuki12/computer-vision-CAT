import cv2
import numpy as np

# Load image
image = cv2.imread('parking_lot.jpg')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours in the edged image
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through contours and approximate the polygons
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    
    # If the contour has 4 vertices, it could be a parking spot
    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Parking Lot Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()







