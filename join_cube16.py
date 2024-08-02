import cv2
import os
import numpy as np

def create_image_grid(image_folder, grid_size=(4, 4), image_size=(256, 256)):
    # List all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    if len(image_files) < np.prod(grid_size):
        raise ValueError("Not enough images in the folder to create a grid.")

    # Read the first 16 images
    images = [cv2.imread(os.path.join(image_folder, image_files[i])) for i in range(np.prod(grid_size))]

    # # Resize images to the specified size (if not already)
    # images = [cv2.resize(img, image_size) for img in images]

    # Create a blank image for the grid
    grid_image = np.zeros((image_size[0] * grid_size[0], image_size[1] * grid_size[1], 3), dtype=np.uint8)

    # Place images into the grid
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            index = i * grid_size[1] + j
            y_start = i * image_size[0]
            y_end = y_start + image_size[0]
            x_start = j * image_size[1]
            x_end = x_start + image_size[1]
            grid_image[y_start:y_end, x_start:x_end] = images[index]

    return grid_image

def save_grid_image(image_grid, output_path):
    cv2.imwrite(output_path, image_grid)

# Usage
image_folder = r'C:\Users\Aser\Downloads\archive\testB'
output_path = r'C:\Users\Aser\PycharmProjects\torch-yolo\Apples_Detection\grid_image_orange.png'

# Create the grid image
grid_image = create_image_grid(image_folder)

# Save the grid image
save_grid_image(grid_image, output_path)

# Optionally, display the grid image
cv2.imshow('Image Grid', grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
