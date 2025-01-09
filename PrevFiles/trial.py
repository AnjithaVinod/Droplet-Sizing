import cv2
import numpy as np

# Load the image in grayscale
image_path = r"C:\Users\anjit\Downloads\C++_ANJITHA\OELP\drop37.jpg"  # Adjust the path to your image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Set the pixel-to-cm conversion factor (estimated)
pixels_per_cm = 50  # Adjust this based on your image's scale

# Minimum diameter threshold in pixels (1 cm converted to pixels)
min_diameter_pixels = 1 * pixels_per_cm

# Apply Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

# Use Otsu's thresholding to separate the droplets from the background
_, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Apply morphological operations to clean up the thresholded image
kernel = np.ones((3, 3), np.uint8)
morphed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)

# Detect contours of the droplets
contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert the grayscale image to BGR for drawing
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

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
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Thicker line for solid rectangles
        cv2.putText(output_image, f"Area: {int(area)} px", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Initialize zoom-related variables
scale_factor = 1.0  # Initial zoom level
max_scale = 5.0  # Maximum zoom level
min_scale = 0.5  # Minimum zoom level
scroll_step = 0.1  # Zoom step for each scroll

# Callback function to handle mouse scroll events
def zoom_image(event, x, y, flags, param):
    global scale_factor, output_image

    # Handle the mouse scroll event
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # Scroll up (zoom in)
            scale_factor = min(max_scale, scale_factor + scroll_step)
        elif flags < 0:  # Scroll down (zoom out)
            scale_factor = max(min_scale, scale_factor - scroll_step)

    # Rescale the image based on the new scale factor
    height, width = output_image.shape[:2]
    zoomed_image = cv2.resize(output_image, (int(width * scale_factor), int(height * scale_factor)))
    
    # Show the zoomed image
    cv2.imshow("Droplets with Size", zoomed_image)

# Display the initial image
cv2.imshow("Droplets with Size", output_image)

# Set the mouse callback function for zooming
cv2.setMouseCallback("Droplets with Size", zoom_image)

# Wait for a key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate average droplet area and the equivalent diameter
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
