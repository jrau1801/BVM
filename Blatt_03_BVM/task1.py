import cv2
import numpy as np


def horizontal_filter_mask(N):
    # Check if N is even or uneven
    if N % 2 == 0:
        print("N darf nur ungerade sein")
        return
    # create a 2d Array with size 2*n+1, 2*n+1
    mask = np.zeros((2 * N + 1, 2 * N + 1))

    # Set value of N-th row to 1.0/2*n+1
    mask[N, :] = 1.0 / (2 * N + 1)

    return mask


def linear_filter(img, filter):
    # Compute the padding required to apply the filter to the image
    pad_height, pad_width = filter.shape[0] // 2, filter.shape[1] // 2

    # Pad the image with zeros
    padded_array = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Apply the filter mask to the image
    filtered_array = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            patch = padded_array[i:i + filter.shape[0], j:j + filter.shape[1]]
            filtered_value = np.sum(patch * filter)
            filtered_array[i, j] = filtered_value

    # Return the filtered image as a NumPy array
    return filtered_array


def fill_mask(img, mask):
    # Invert the mask to select the background
    background_mask = np.logical_not(mask)

    # Fill the background with black
    img[background_mask] = 0

    # Extract the car object from the original image
    car = img.copy()
    car[background_mask] = 0

    return car


def apply_masked_image(image, masked_image):
    # Extract height and width from image
    height, width = image.shape

    # Create a new image
    result = np.zeros_like(image)

    # Iterate through the image
    for x in range(height):
        for y in range(width):
            # if pixel is black take pixel from original image
            if masked_image[x, y] == 0:
                result[x, y] = image[x, y]

            # take pixel from masked image
            else:
                result[x, y] = masked_image[x, y]

    return result


# Size of mask
N = 11

# Load images
image = cv2.imread('p03_car.png', cv2.IMREAD_GRAYSCALE)
mask_image = cv2.imread('p03_maske.png', cv2.IMREAD_GRAYSCALE)

# Process images through filters
filter_mask = horizontal_filter_mask(N)
filtered_image = linear_filter(image, filter_mask)
filled_mask = fill_mask(image, mask_image)
result = apply_masked_image(filtered_image, filled_mask)

cv2.imshow('Filtered', filtered_image)
cv2.imshow("Car", filled_mask)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
