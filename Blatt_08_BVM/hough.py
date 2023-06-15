import cv2
import numpy as np
from math import pi, sin, cos


def detect_coins(image, radius, hough_thresh):
    rows, cols = image.shape

    # Initialize result matrix with same dimensions as image
    result_matrix = np.zeros(image.shape)

    # Iterate over the image
    for x in range(rows - radius):
        for y in range(cols - radius):

            if image[x][y] == 255:
                # Calculate coordinates
                for theta in range(0, 360):
                    point_b = y - radius * sin(theta * pi / 180)
                    point_a = x - radius * cos(theta * pi / 180)

                    result_matrix[int(point_a)][int(point_b)] += 1

    # Draw circles and center points
    count, circles = center_points(image, result_matrix, radius, hough_thresh)

    return count, circles


def center_points(image, res_mat, radius, hough_thresh):
    # Initialize circles and center points
    center = np.where(res_mat > hough_thresh)
    circles = np.zeros(image.shape)

    # Set coin-counter and proximity
    coin_counter = 0
    proximity = 5

    # Iterate through the center-points
    for i in range(center[0].size):
        # Check if there is a nearby center within the proximity threshold
        is_near = any(
            np.sum(
                circles[
                center[0][i]: int(center[0][i] - (radius / proximity) * cos(theta * pi / 180)),
                center[1][i]: int(center[1][i] - (radius / proximity) * sin(theta * pi / 180)),
                ]
            ) > 128
            for theta in range(0, 360)
        )

        # If nothing is in proximity
        if not is_near:
            # Increment coin counter
            coin_counter += 1
            # Draw the center
            cv2.circle(circles, (center[1][i], center[0][i]), 1, 255, 3)
            # Draw circle around center
            cv2.circle(circles, (center[1][i], center[0][i]), radius, 150, 0)

    return coin_counter, circles


def print_coins(cents, radii, colors, total):
    # Start coin detection and draw circles and center
    for i in range(len(radii)):
        coin_counter, cent_circles = detect_coins(canny_image, radii[i], threshold)
        color_image[cent_circles >= threshold] = colors[i]

        # Print type and amount of coins
        print(f"{coin_counter} x {cents[i]} Cent")
        total += cents[i] * coin_counter

    # Print total value
    print(f"Total: {total} Cent")


# Load image
image = cv2.imread("p08_muenzen.png", cv2.IMREAD_GRAYSCALE)

# Apply gaussian and canny filter
gaussian_image = cv2.GaussianBlur(image, (5, 5), 1)
canny_image = cv2.Canny(gaussian_image, 170, 200)

# Convert back to RGB
color_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Initialize Threshold
threshold = 130

# Initialize Cent-type, radii and colors
cents = [1, 2, 5]
radii = [23, 27, 32]
colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
total = 0

print_coins(cents, radii, colors, total)

cv2.imshow("hough", color_image)
cv2.waitKey()
cv2.destroyAllWindows()
