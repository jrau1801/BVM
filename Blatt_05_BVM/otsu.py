import numpy as np
import cv2


def otsu_thresholding(img):

    max_gray_value = 256

    # Calculate the histogram of the image.
    histogram, _ = np.histogram(img, bins=max_gray_value, range=[0, max_gray_value])
    histogram = histogram.astype(float)
    histogram /= histogram.sum()  # Normiere das Histogramm

    max_variance = 0.0
    threshold = 0

    # Iterate through all threshold values from 0 to 255.
    for t in range(max_gray_value-1):
        # Calculate the probabilities of the two classes.
        p1 = histogram[:t].sum()
        p2 = histogram[t:].sum()

        # Calculate the mean intensities of the two classes.
        if p1 == 0 or p2 == 0:
            continue

        μ1 = np.dot(np.arange(t), histogram[:t]) / p1
        μ2 = np.dot(np.arange(t, max_gray_value), histogram[t:]) / p2

        # Calculate the variance between the classes.
        inter_variance = p1 * p2 * (μ1 - μ2) ** 2

        # Pass parameters to calculate equation
        variance_eq(p1, p2, μ1, μ2)

        # Update the maximum variance value and threshold.
        if inter_variance > max_variance:
            max_variance = inter_variance
            threshold = t

    # Apply the threshold to the image.
    result_img = (img < threshold).astype(np.uint8) * 255

    return result_img


def variance_eq(p1, p2, μ1, μ2):
    global listLeft, listRight

    # Overall mean value.
    μg = p1 * μ1 + p2 * μ2

    # Calculate left side and right side of equation
    left_eq = round(p1 * (μ1 - μg) ** 2 + p2 * (μ2 - μg) ** 2, 3)
    right_eq = round(p1 * p2 * (μ1 - μ2) ** 2, 3)

    # Add results to list
    listLeft.append(left_eq)
    listRight.append(right_eq)


# Initialize lists
listLeft = []
listRight = []

# Load image
image = cv2.imread("p05_gummibaeren.png")

# Start Otsu-method
otsu_image = otsu_thresholding(image)

# Compare both lists
if listLeft == listRight:
    print("TRUE")

# Show image
cv2.imshow('Otsu Image', otsu_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
