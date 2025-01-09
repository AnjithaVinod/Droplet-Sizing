import cv2
import numpy as np

# Load the image
image_path = r"C:\Users\anjit\Downloads\C++_ANJITHA\OELP\drop1.jpg"  # Adjust the path to your image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Set the pixel-to-cm conversion factor (estimated)
pixels_per_cm = 50  # Adjust this based on your image's scale

# Minimum diameter threshold in pixels (1 cm converted to pixels)
min_diameter_pixels = 1 * pixels_per_cm

# Convert the image to grayscale for edge detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply bilateral filtering to reduce noise while preserving edges
filtered_image = cv2.bilateralFilter(gray_image, 9, 75, 75)

# Use adaptive thresholding to separate the droplets from the background
thresholded_image = cv2.adaptiveThreshold(
    filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# Apply morphological operations to clean up the thresholded image
kernel = np.ones((3, 3), np.uint8)
morphed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel, iterations=2)

# Detect contours of the droplets
contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert the original image to BGR for drawing
output_image = image.copy()

# Initialize lists to store droplet areas and equivalent circular diameters
droplet_areas = []
equivalent_diameters = []

# Loop over all contours found
for contour in contours:
    # Calculate the contour area
    area = cv2.contourArea(contour)
    
    # Calculate the equivalent circular diameter based on area
    equivalent_diameter = np.sqrt(4 * area / np.pi)

    # Filter out droplets smaller than the minimum diameter in pixels
    if equivalent_diameter > min_diameter_pixels:
        droplet_areas.append(area)
        equivalent_diameters.append(equivalent_diameter)

        # Draw a bounding rectangle around the droplet
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(output_image, f"Area: {int(area)} px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Thresholded Image", thresholded_image)
cv2.imshow("Droplets with Size", output_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate average droplet area and equivalent diameter
if len(droplet_areas) > 0:
    avg_droplet_area = sum(droplet_areas) / len(droplet_areas)
    avg_equivalent_diameter = sum(equivalent_diameters) / len(equivalent_diameters)
else:
    avg_droplet_area = 0
    avg_equivalent_diameter = 0

# Display results in the console
print(f"Detected {len(droplet_areas)} droplets")
print(f"Average droplet area (in pixels): {avg_droplet_area}")
print(f"Average equivalent diameter (in pixels): {avg_equivalent_diameter}")