import cv2
import numpy as np


def gaussian_kernel(size, sigma):
    # Initialize new matrix with zeros
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            # Calculate x and y
            x, y = i - size // 2, j - size // 2
            # Add to kernel
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    # Normalize so sum is 1
    kernel /= np.sum(kernel)
    return kernel


def gaussian_filter(image, size, sigma):
    # Calculate kernel and apply with filter2D()
    kernel = gaussian_kernel(size, sigma)
    filtered = cv2.filter2D(image, -1, kernel)
    return filtered


def gradient(image):
    # Define Sobel filters for x and y directions
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Compute the gradient in x and y directions using filter2D()
    gradientx = cv2.filter2D(image, cv2.CV_64F, sobelx)
    gradienty = cv2.filter2D(image, cv2.CV_64F, sobely)

    # Compute the gradient magnitude and direction
    magnitude = np.sqrt(gradientx ** 2 + gradienty ** 2)
    direction = np.arctan2(gradienty, gradientx)

    return magnitude, direction


def non_maxima_suppression(magnitude, direction):
    # Initialize output image
    non = np.zeros_like(magnitude, dtype=np.float32)

    # Extract height and width from image
    height, width = magnitude.shape

    # Round 45 degrees
    dir_rounded = np.round(direction / (np.pi / 4)) % 4

    # Loop over image
    for i in range(height):
        for j in range(width):
            # Horizontal on 0 degrees
            if dir_rounded[i, j] == 0:
                non[i, j] = magnitude[i, j] if (magnitude[i, j] > magnitude[i, j - 1]) and \
                                               (magnitude[i, j] > magnitude[i, j + 1]) else 0
            # Diagonal on 45 degress
            elif dir_rounded[i, j] == 1:
                non[i, j] = magnitude[i, j] if (magnitude[i, j] > magnitude[i - 1, j - 1]) and \
                                               (magnitude[i, j] > magnitude[i + 1, j + 1]) else 0
            # Vertical on 90 degrees
            elif dir_rounded[i, j] == 2:
                non[i, j] = magnitude[i, j] if (magnitude[i, j] > magnitude[i - 1, j]) and \
                                               (magnitude[i, j] > magnitude[i + 1, j]) else 0
            # Diagonal on 135 degrees
            else:
                non[i, j] = magnitude[i, j] if (magnitude[i, j] > magnitude[i - 1, j + 1]) and \
                                               (magnitude[i, j] > magnitude[i + 1, j - 1]) else 0

    return non


def hysteresis_thresholding(image, low_threshold, high_threshold):
    # Initialize output image with zeros
    thresholded = np.zeros_like(image)

    # Extract height and width from image
    height, width = image.shape

    # Find pixels with gradient magnitude higher than the high threshold
    strong_pixels = image >= high_threshold

    # Find pixels with gradient magnitude between the low and high thresholds
    weak_pixels = (image >= low_threshold) & (image < high_threshold)

    # Set strong pixels to white in the output image
    thresholded[strong_pixels] = 255

    # Keep track of weak pixels that are connected to strong pixels
    connected_pixels = set()
    for i in range(height - 1):
        for j in range(width - 1):
            if weak_pixels[i, j]:
                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        if thresholded[i + ii, j + jj] == 255:
                            connected_pixels.add((i, j))
                            break
                    else:
                        continue
                    break

    # Set connected weak pixels to white in the output image
    for pixel in connected_pixels:
        thresholded[pixel] = 0

    return thresholded


# Load the image
image = cv2.imread('p04_arches.jpg', cv2.IMREAD_GRAYSCALE)

# Apply a Gaussian filter with sigma=3 and kernel size 7x7
size = 7
sigma = 3
filtered = gaussian_filter(image, size, sigma)

# Get gradient magnitude and direction from filtered image
magnitude, direction = gradient(filtered)

# Apply non-maxima-suppression with magnitude and direction values
non_maxima = non_maxima_suppression(magnitude, direction)

# Apply Hysterese on non-maxima image with low and high threshold
t_low = 20
t_high = 55
hyst = hysteresis_thresholding(non_maxima, t_low, t_high)

# Display the original and filtered images side by side
cv2.imshow('Filtered', filtered)
cv2.imshow('Magnitude', magnitude)
cv2.imshow('Direction', direction)
cv2.imshow('Non-Maxima', non_maxima)
cv2.imshow('Canny', hyst)

cv2.waitKey(0)
cv2.destroyAllWindows()
