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
image_path = r"C:\Users\anjit\Downloads\C++_ANJITHA\OELP\drop18.jpg" # Replace with your image path
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Unable to load image at {image_path}")
    exit()

# Save color copy for visualizing
imgc = img.copy()

# Resize the image
resize_factor = 1.5
img = cv2.resize(img, None, fx=1 / resize_factor, fy=1 / resize_factor)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Use Sobel operator and determine CLAHE clip limit
sobel = get_sobel(img_gray)
clip_limit = (-2.556) * np.sum(sobel) / (img_gray.shape[0] * img_gray.shape[1]) + 26.557
clip_limit = max(0.1, min(clip_limit, 8.0))

# Apply CLAHE and unsharp mask
enhanced_img = clahe(img_gray, clip_limit)
enhanced_img = unsharp_mask(enhanced_img)

# Apply Gaussian blur and Canny edge detection
blurred_img = cv2.GaussianBlur(enhanced_img, (5, 5), 0)
canny = cv2.Canny(blurred_img, 100, 255)

# Find contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Annotate output image with droplet sizes
output_img = cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 50:  # Minimum threshold to filter noise
        # Draw contour
        cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 2)

        # Calculate centroid for placing the annotation
        moments = cv2.moments(contour)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.putText(output_img, f"{int(area)} px", (cx - 20, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

# Display the output image with the number of pixels for each droplet
cv2.imshow("Droplet Detection and Pixel Count", output_img)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
