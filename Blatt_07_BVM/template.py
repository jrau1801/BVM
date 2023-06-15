import cv2
import numpy as np


def template_matching_ssd(reference, template):

    # Extract width and height from reference and template
    ref_width, ref_height, _ = reference.shape
    temp_width, temp_height, _ = template.shape

    # Calculate rows and cols for result image
    match_rows = ref_width - temp_width + 1
    match_cols = ref_height - temp_height + 1

    # Initialize an image filled with zeros
    result = np.zeros((match_rows, match_cols))

    # Iterate over all possible positions of the template in the reference image.
    for x in range(match_rows):
        for y in range(match_cols):
            # Calculate the Sum of Squared Differences (SSD) for the current template at position (x, y).
            ssd = np.sum((reference[x:x + temp_width, y:y + temp_height] - template) ** 2)
            result[x, y] = ssd

    # Find the position with the minimum SSD value.
    min_val = np.min(result)
    min_loc = np.where(result == min_val)
    top_left = (min_loc[1][0], min_loc[0][0])
    bottom_right = (top_left[0] + temp_height, top_left[1] + temp_width)

    # Get top left and bottom right position to draw a rectangle
    result_image = cv2.rectangle(reference.copy(), top_left, bottom_right, (0, 0, 255), 2)

    return result_image


def template_matching_cor(reference, template):

    # Extract width and height from reference and template
    ref_width, ref_height, _ = reference.shape
    temp_width, temp_height, _ = template.shape

    # Calculate rows and cols for result image
    match_rows = ref_width - temp_width + 1
    match_cols = ref_height - temp_height + 1

    # Initialize an image filled with zeros
    result = np.zeros((match_rows, match_cols))

    # Iterate over all possible positions of the template in the reference image.
    for x in range(match_rows):
        for y in range(match_cols):
            # Extract the region in the reference image.
            region = reference[x:x + temp_width, y:y + temp_height, :]

            # Calculate the correlation coefficient for the current template at position (x, y).
            numerator = np.sum((region - np.mean(region)) * (template - np.mean(template)))
            denominator = np.sqrt(np.sum((region - np.mean(region)) ** 2) * np.sum((template - np.mean(template)) ** 2))

            # Calculate the correlation coefficient.
            corr_coeff = numerator / denominator

            # Store the correlation coefficient in the result map.
            result[x, y] = corr_coeff

    # Find the position with the maximum correlation coefficient
    max_val = np.max(result)
    max_loc = np.where(result == max_val)
    top_left = (max_loc[1][0], max_loc[0][0])
    bottom_right = (top_left[0] + temp_height, top_left[1] + temp_width)

    # Get top left and bottom right position to draw a rectangle
    result_image = reference.copy()
    result_image = cv2.rectangle(result_image, top_left, bottom_right, (0, 0, 255), 2)

    return result_image


def template_matching_brute_force(reference, template):

    # Extract width and height from reference and template
    ref_width, ref_height, _ = reference.shape
    temp_width, temp_height, _ = template.shape

    # Calculate rows and cols for result image
    match_rows = ref_width - temp_width + 1
    match_cols = ref_height - temp_height + 1

    # Initialize two images filled with zeros
    ssd_bf = np.zeros((match_rows, match_cols))
    cor_bf = np.zeros((match_rows, match_cols))

    # Calculation of similarities per pixel
    for x in range(match_rows):
        for y in range(match_cols):
            region = reference[x:x + temp_width, y:y + temp_height, :]

            # Sum of Squared Differences (SSD)
            ssd = np.sum((region - template) ** 2)
            ssd_bf[x, y] = ssd

            # Correlation Coefficient (COR)
            numerator = np.sum((region - np.mean(region)) * (template - np.mean(template)))
            denominator = np.sqrt(np.sum((region - np.mean(region)) ** 2) * np.sum((template - np.mean(template)) ** 2))
            correlation_coeff = numerator / denominator
            cor_bf[x, y] = correlation_coeff

    # Search for areas with the highest similarity.
    min_val_ssd = np.min(ssd_bf)
    max_val_cor = np.max(cor_bf)

    threshold_ssd = min_val_ssd + 0.1 * (max_val_cor - min_val_ssd)
    threshold_cor = max_val_cor - 0.1 * (max_val_cor - min_val_ssd)

    ssd_loc = np.where(ssd_bf <= threshold_ssd)
    cor_loc = np.where(cor_bf >= threshold_cor)

    # Visualization of the results.
    result_image_ssd = reference.copy()
    result_image_cor = reference.copy()

    # Get top left and bottom right position to draw a rectangle
    top_left = (ssd_loc[1][0], ssd_loc[0][0])
    bottom_right = (top_left[0] + temp_height, top_left[1] + temp_width)
    result_image_ssd = cv2.rectangle(result_image_ssd, top_left, bottom_right, (0, 0, 255), 2)

    top_left = (cor_loc[1][0], cor_loc[0][0])
    bottom_right = (top_left[0] + temp_height, top_left[1] + temp_width)
    result_image_cor = cv2.rectangle(result_image_cor, top_left, bottom_right, (0, 0, 255), 2)

    return result_image_ssd, result_image_cor


# Load images
reference_img = cv2.imread("p07_reference.png")
template_img = cv2.imread("p07_template.png")

# Save results
result_ssd = template_matching_ssd(reference_img, template_img)
result_cor = template_matching_cor(reference_img, template_img)
result_ssd_bf, result_cor_bf = template_matching_brute_force(reference_img, template_img)

# Show images
cv2.imshow("Result SSD", result_ssd)
cv2.imshow("Result COR", result_cor)
cv2.imshow("Result SSD Brute-Force", result_ssd_bf)
cv2.imshow("Result COR Brute-Force", result_cor_bf)

cv2.waitKey(0)
cv2.destroyAllWindows()
