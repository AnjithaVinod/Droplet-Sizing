import cv2
import numpy as np

def unsharp_mask(img, blur_size=(5, 5), img_weight=1.5, gaussian_weight=-0.5):
    """Applies unsharp masking to enhance image details."""
    gaussian = cv2.GaussianBlur(img, blur_size, 0)
    return cv2.addWeighted(img, img_weight, gaussian, gaussian_weight, 0)

def clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the image."""
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe_obj.apply(img)

def get_sobel(img, size=3):
    """Calculates the Sobel gradient for the image."""
    sobelx64f = cv2.Sobel(img, cv2.CV_64F, 2, 0, ksize=size)
    abs_sobel64f = np.absolute(sobelx64f)
    return np.uint8(abs_sobel64f)

# Load the input image
image_path =  r"C:\Users\anjit\Downloads\C++_ANJITHA\OELP\drop18.jpg"  # Replace with your image path
img = cv2.imread(image_path)

# Check if the image was loaded successfully
if img is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Save color copy for visualizing
imgc = img.copy()

# Resize the image to simplify analysis
resize_factor = 1.5
img = cv2.resize(img, None, fx=1 / resize_factor, fy=1 / resize_factor)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Sobel operator to evaluate high frequencies
sobel = get_sobel(img_gray)

# Experimentally calculated function for determining clip limit
clip_limit = (-2.556) * np.sum(sobel) / (img_gray.shape[0] * img_gray.shape[1]) + 26.557

# Limit CLAHE if there's not enough or too much detail
clip_limit = max(0.1, min(clip_limit, 8.0))

# Apply CLAHE and unsharp masking to enhance high frequencies
img_gray = clahe(img_gray, clip_limit)
img_gray = unsharp_mask(img_gray)

# Apply Gaussian blur to smooth the image
blurred_img = cv2.GaussianBlur(img_gray, (5, 5), 0)

# Perform edge detection using Canny
canny = cv2.Canny(blurred_img, 100, 255)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize droplet count
droplet_count = 0

# Loop over each contour to calculate its area and draw it on the image
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    if area > 100:  # Only consider contours with area larger than a threshold (e.g., 100 pixels)
        droplet_count += 1
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(imgc, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(imgc, f"Droplet {droplet_count}: {int(area)} px", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Print the droplet number and pixel area in the terminal
        print(f"Droplet {droplet_count}: {int(area)} pixels")

# Print the total droplet count in the terminal
print(f"Total droplets detected: {droplet_count}")

# Show the result with contours and pixel counts labeled
cv2.imshow("Detected Droplets", imgc)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
