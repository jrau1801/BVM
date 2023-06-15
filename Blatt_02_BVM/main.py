import cv2
import numpy as np


def fill_image_with_tiles(img1, img2, tile_size):

    original_image = img1
    tile_image = img2

    # Get the dimensions of the original image
    original_height, original_width, _ = original_image.shape

    # Calculate the number of tiles needed in horizontal and vertical directions
    num_horizontal_tiles = original_width // (tile_size + tile_size)
    num_vertical_tiles = original_height // (tile_size + tile_size)

    # Iterate over the tiles and paste random tile patches onto the filled image
    for y in range(num_vertical_tiles):
        for x in range(num_horizontal_tiles):
            # Choose a random tile patch from the tile image
            tile_patch_x = np.random.randint(0, tile_image.shape[1] - tile_size)
            tile_patch_y = np.random.randint(0, tile_image.shape[0] - tile_size)
            tile_patch = tile_image[tile_patch_y:tile_patch_y + tile_size, tile_patch_x:tile_patch_x + tile_size]

            # Calculate the coordinates of the tile in the filled image
            filled_x = x * (tile_size + tile_size)
            filled_y = y * (tile_size + tile_size)

            # Paste the tile patch onto the filled image
            img1[filled_y:filled_y + tile_size, filled_x:filled_x + tile_size] = tile_patch

    return img1


def recolor_image(image):

    # Create histogram from gray-scaled image
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    recolored_image = np.zeros_like(image)
    max_gray_value = len(histogram) - 1
    current_pixel_count = 0
    current_gray_value = 0

    # Iterate over the image and arrange pixels from black to white
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            while current_pixel_count >= histogram[current_gray_value] and current_gray_value < max_gray_value:

                current_gray_value += 1
                current_pixel_count = 0

            recolored_image[i, j] = current_gray_value
            current_pixel_count += 1

    return recolored_image


# Load images
minden = cv2.imread('p02_minden.jpg')
sonne = cv2.imread('p02_sonne.jpg')

# Make images gray
minden_gray = cv2.cvtColor(minden, cv2.COLOR_BGR2GRAY)
sonne_gray = cv2.cvtColor(sonne, cv2.COLOR_BGR2GRAY)

tile_nxn = 80

# Show images on screen
cv2.imshow('Tiles', fill_image_with_tiles(minden, sonne, tile_nxn))
cv2.imshow('Minden', recolor_image(minden_gray))
cv2.imshow('Sonne', recolor_image(sonne_gray))

cv2.waitKey(0)
cv2.destroyAllWindows()
