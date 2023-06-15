import cv2
import numpy as np


def region_growing(region_image, seed_px):

    # Initialize image with zeros
    region_growing_result = np.zeros(region_image.shape, dtype=np.uint8)

    # Start region growing
    region_growing_result = grow_region(seed_px, region_growing_result)

    return region_growing_result


def grow_region(seed_px, region_growing_img):
    # Extract height and width from image
    height, width = closing.shape

    # get x and y from Seed-pixel
    seed_value = float(closing[seed_px[1], seed_px[0]])

    # Initialize queue
    queue = [seed_px]

    while len(queue) > 0:
        current_pixel = queue.pop(0)

        # Found region color is white (255)
        if region_growing_img[current_pixel[1], current_pixel[0]] == 255:
            continue

        current_value = float(closing[current_pixel[1], current_pixel[0]])

        # If abs value lower or equal to threshold
        if abs(current_value - seed_value) <= threshold:

            # Set found region color to white (255)
            region_growing_img[current_pixel[1], current_pixel[0]] = 255

            # Iterate over neighboring pixels and add them to queue
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if 0 <= current_pixel[1] + i < height and 0 <= current_pixel[0] + j < width:
                        queue.append((current_pixel[0] + j, current_pixel[1] + i))

    return region_growing_img


def one_px_skeleton(img):
    # Dilate the image to ensure connectivity of the street
    img = cv2.dilate(img, struct_element, iterations=5)

    # Extract image size
    img_size = np.size(img)

    # Initialize the image with zeros
    output = np.zeros(img.shape, np.uint8)

    # Threshold the image to binary (0 and 255)
    ret, img = cv2.threshold(img, 0, 255, 0)

    # Define the structuring element
    element = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]], dtype=np.uint8)

    while True:
        # Perform erosion on the image
        eroded = cv2.erode(img, element)

        # Perform dilation on the eroded image
        foo = cv2.dilate(eroded, element)

        # Subtract the dilated image from the original eroded image
        foo = cv2.subtract(img, foo)

        # Perform bitwise OR operation to update the skeleton image
        output = cv2.bitwise_or(output, foo)

        # Update the current image with the eroded image
        img = eroded.copy()

        # Count the number of zero pixels in the current image
        zeros = img_size - cv2.countNonZero(img)

        # If all pixels are zero, the skeletonization is complete
        if zeros == img_size:
            return output


def overlay_skeleton(orig_image, skel_mask):
    # Create a mask from the skeleton image
    skeleton_mask = np.zeros_like(orig_image)
    skeleton_mask[skel_mask > 0] = 255

    # Overlay the skeleton mask on the original image
    overlay = cv2.addWeighted(orig_image, 1, skeleton_mask, 1, 0)

    return overlay


threshold = 0

# Seed-Pixel for region growing, found out with mouse callback
# but deleted mouse callback for better handling
seed_pixel = (207, 168)

# Load image
image_original = cv2.imread('p06_strasse.jpg')
image_blur = cv2.GaussianBlur(image_original, (5, 5), 0)

# Define the structural element
struct_element_size = 5
if struct_element_size % 2 == 0:
    print("Strukturelementgröße muss ungerade sein")
    exit(0)
struct_element = np.ones((struct_element_size, struct_element_size))

# Canny for edge detection, define better edges with
# dilatation and closing, fill the street region
# skeletonize and overlay it
canny = cv2.Canny(image_blur, 100, 200)
dilated = cv2.dilate(canny, struct_element)
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, struct_element)
region = region_growing(closing, seed_pixel)
skeleton = one_px_skeleton(region)
overlayed_skeleton = overlay_skeleton(image_original, skeleton)

# Show images
cv2.imshow("Blur", image_blur)
cv2.imshow("Canny", canny)
cv2.imshow("Dilated", dilated)
cv2.imshow("Closing", closing)
cv2.imshow("Region", region)
cv2.imshow("Skeleton", skeleton)
cv2.imshow("Overlayed-Skeleton", overlayed_skeleton)

cv2.waitKey(0)
cv2.destroyAllWindows()
