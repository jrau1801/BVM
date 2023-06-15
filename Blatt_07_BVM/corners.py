import cv2
import numpy as np


def corner_detection(img, alpha):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate derivatives using Sobel operator
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # Calculate entries of matrix S using a 5x5 box filter
    IxIx = cv2.boxFilter(dx * dx, cv2.CV_64F, (5, 5))
    IxIy = cv2.boxFilter(dx * dy, cv2.CV_64F, (5, 5))
    IyIy = cv2.boxFilter(dy * dy, cv2.CV_64F, (5, 5))

    # Calculate eigenvalues for each pixel
    det_S = IxIx * IyIy - IxIy ** 2
    trace_S = IxIx + IyIy
    lambda_1 = 0.5 * (trace_S + np.sqrt(trace_S ** 2 - 4 * det_S))
    lambda_2 = 0.5 * (trace_S - np.sqrt(trace_S ** 2 - 4 * det_S))

    # Find corner positions
    corners = np.argwhere(np.minimum(lambda_1, lambda_2) > alpha)

    # Draw circles at the corner positions
    result = np.copy(img)
    for corner in corners:
        cv2.circle(result, (corner[1], corner[0]), 1, (0, 0, 255), -1)

    return result


alpha = 1000

# Example usage
image = cv2.imread("p07_ecken.png")
corners = corner_detection(image, alpha)

# Display the result
cv2.imshow("Corner Detection", corners)
cv2.waitKey(0)
cv2.destroyAllWindows()
