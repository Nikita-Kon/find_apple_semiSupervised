import os
from ultralytics import YOLO
import cv2
import numpy as np

# read images paths into array
def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# normalize data from max array to min array
def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)

def picturesAmount(source_directory):
    image_paths = get_image_paths(source_directory)
    size = len(image_paths)
    return size
# make pseudo data
def makePseudoLabels(source_directory, dataset_directory):
    # load model
    model = YOLO(r"best.pt")
    # read pictures
    image_paths = get_image_paths(source_directory)
    for image_path in image_paths:
        image = cv2.imread(image_path)
        # predict
        result = model.predict(source=image)

        boxes = result[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            class_id = box.cls[0]
            class_name = model.names[int(class_id)]
            if confidence >= 0.6:
                data = [x1, y1, x2, y2, 0, 256]
                data = min_max_normalize(data)
                x1, y1, x2, y2 = data[:4]
                # Extract the filename without the directory path
                filename_with_extension = os.path.basename(image_path)
                # Replace the .jpg extension with .txt
                filename_txt = os.path.splitext(filename_with_extension)[0] + ".txt"
                dataset_labels_dir = dataset_directory + "\\labels\\" + filename_txt
                dataset_image_dir = dataset_directory + "\\images\\" + filename_with_extension

                if os.path.exists(image_path):
                    print(image_path, dataset_image_dir, dataset_labels_dir)
                    os.rename(image_path, dataset_image_dir)
                    with open(dataset_labels_dir, 'a') as file:
                        res_string = f"0 {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"
                        file.write(res_string + '\n')

# images_directory = r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\apples_learning\unlabeled_images\images"
# dataset_directory = "C:\\Users\\Aser\\PycharmProjects\\torch-yolo\\find_apple_semiSupervised\\apples_learning\\trainLabeled"
# makePseudoLabels(images_directory, dataset_directory)

