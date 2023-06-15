import random
import cv2
import numpy as np


def weighted_median_filter(image, weights):
    # Extract height and width for padding
    pad_height, pad_width = weights.shape[0] // 2, weights.shape[1] // 2

    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Create a new image
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            # Extract neighborhood of pixel (i, j)
            neighborhood = padded_image[i:i + weights.shape[0], j:j + weights.shape[1]]

            # Calculate weighted median of neighborhood
            weighted_values = np.repeat(neighborhood.flatten(), weights.flatten())
            weighted_median = np.median(weighted_values)
            filtered_image[i, j] = weighted_median

    return filtered_image


image = cv2.imread('p03_car.png', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)


# Create weight matrix with random ints
N = 1
weights = np.array([[random.randint(1, 5) for j in range(2 * N + 1)] for i in range(2 * N + 1)])
print(weights)

# Apply weighted median filter
filtered_image = weighted_median_filter(image, weights)

# Show images
cv2.imshow("Original", image)
cv2.imshow("Result", filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
