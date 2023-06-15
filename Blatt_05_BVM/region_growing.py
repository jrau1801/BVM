import cv2
import numpy as np


def region_growing(event, x, y, flags, param):
    global seed_pixel

    # On leftclick
    if event == cv2.EVENT_LBUTTONDOWN:
        seed_pixel = (x, y)

        # Initialize image with zeros
        region_growing_result = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        # Set Background to blue
        region_growing_result[:, :] = (200, 0, 0)

        # Start region growing
        region_growing_result = grow_region(seed_pixel, region_growing_result)

        # Green marker
        cv2.circle(region_growing_result, seed_pixel, 3, (0, 255, 0), -1)

        cv2.imshow('Region Growing', region_growing_result)

    # On rightclick
    if event == cv2.EVENT_RBUTTONDOWN:
        cv2.imshow('Region Growing', image)


def grow_region(seed_px, region_growing_img):
    # Extract height and width from image
    height, width = image.shape[:2]

    # get x and y from Seed-pixel
    seed_value = float(image[seed_px[1], seed_px[0]])

    # Initialize queue
    queue = [seed_px]

    while len(queue) > 0:
        current_pixel = queue.pop(0)

        # Found region color is red
        if np.array_equal(region_growing_img[current_pixel[1], current_pixel[0]], [0, 0, 255]):
            continue

        current_value = float(image[current_pixel[1], current_pixel[0]])

        # If abs value lower or equal to threshold
        if abs(current_value - seed_value) <= threshold:

            # Set found region color to red
            region_growing_img[current_pixel[1], current_pixel[0]] = [0, 0, 255]

            # Iterate over neighboring pixels and add them to queue
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= current_pixel[1] + i < height and 0 <= current_pixel[0] + j < width:
                        queue.append((current_pixel[0] + j, current_pixel[1] + i))

    return region_growing_img


# Initialize Seed-pixel and threshold
seed_pixel = None
threshold = 37

# Load image
image = cv2.imread('p05_gummibaeren.png', cv2.IMREAD_GRAYSCALE)

# Create a named window and set the mouse callback function
cv2.namedWindow('Region Growing')
cv2.setMouseCallback('Region Growing', region_growing)

# Show image
cv2.imshow('Region Growing', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
