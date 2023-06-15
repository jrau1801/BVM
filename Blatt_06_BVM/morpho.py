import cv2
import numpy as np


def dilate(image, str_element):
    # Get kernel dimensions
    k_rows, k_cols = str_element.shape
    k_center_row, k_center_col = k_rows // 2, k_cols // 2

    # Get image dimensions
    rows, cols = image.shape

    # Create a blank output image
    output = np.zeros_like(image)

    # Iterate over each pixel in the image
    for row in range(rows):
        for col in range(cols):
            # Check if the current pixel is non-zero (foreground)
            if image[row, col] != 0:
                # Iterate over each pixel in the kernel
                for k_row in range(k_rows):
                    for k_col in range(k_cols):
                        # Calculate the corresponding pixel position in the image
                        i_row = row + (k_row - k_center_row)
                        i_col = col + (k_col - k_center_col)

                        # Check if the corresponding position is within the image boundaries
                        if 0 <= i_row < rows and 0 <= i_col < cols:
                            # Set the corresponding pixel in the output image to non-zero
                            output[i_row, i_col] = 255

    return output


def erode(image, str_element):
    # Get kernel dimensions
    k_rows, k_cols = str_element.shape
    k_center_row, k_center_col = k_rows // 2, k_cols // 2

    # Get image dimensions
    rows, cols = image.shape

    # Create a blank output image
    output = np.zeros_like(image)

    # Iterate over each pixel in the image
    for row in range(rows):
        for col in range(cols):
            # Check if the current pixel is non-zero (foreground)
            if image[row, col] != 0:
                erode_pixel = True

                # Iterate over each pixel in the kernel
                for k_row in range(k_rows):
                    for k_col in range(k_cols):
                        # Calculate the corresponding pixel position in the image
                        i_row = row + (k_row - k_center_row)
                        i_col = col + (k_col - k_center_col)

                        # Check if the corresponding position is within the image boundaries
                        if 0 <= i_row < rows and 0 <= i_col < cols:
                            # Check if any of the kernel pixels and corresponding image pixels are zero
                            if str_element[k_row, k_col] != 0 and image[i_row, i_col] == 0:
                                erode_pixel = False
                                break

                    if not erode_pixel:
                        break

                # Set the corresponding pixel in the output image
                if erode_pixel:
                    output[row, col] = 255

    return output


def close(image, str_element):
    # Perform dilation followed by erosion
    dilated = dilate(image, str_element)
    closed_image = erode(dilated, str_element)
    return closed_image


image_gear = cv2.imread("p06_zahnrad.png", cv2.IMREAD_GRAYSCALE)

# Define the structural element
struct_element_size = 5
if struct_element_size % 2 == 0:
    print("Strukturelementgröße muss ungerade sein")
    exit(0)
struct_element = np.ones((struct_element_size, struct_element_size))

result_dilated = dilate(image_gear, struct_element)
result_eroded = erode(image_gear, struct_element)

# Remove disturbances using closing operation
result_gear = close(image_gear, struct_element)

cv2.imshow("Original", image_gear)
cv2.imshow("Dilated", result_dilated)
cv2.imshow("Eroded", result_eroded)
cv2.imshow("Closing", result_gear)

cv2.waitKey(0)
cv2.destroyAllWindows()
