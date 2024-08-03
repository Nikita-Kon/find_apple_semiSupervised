import shutil
import os
from ultralytics import YOLO
import cv2
import numpy as np

def get_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)

# load model
model = YOLO(r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\runs\train\exp\weights\best.pt")
# read pictures
source_directory = r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\apples_learning\trainUnlabeled\images"
image_paths = get_image_paths(source_directory)

for i in image_paths:
    image = cv2.imread(i)
    # predict
    result = model.predict(source=image)

    boxes = result[0].boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        class_id = box.cls[0]
        class_name = model.names[int(class_id)]
        if confidence > 0.7:
            data = [x1, y1, x2, y2, 0, 256]
            data = min_max_normalize(data)
            x1, y1, x2, y2 = data[:4]
            # Extract the filename without the directory path
            filename_with_extension = os.path.basename(i)
            # Replace the .jpg extension with .txt
            filename_txt = os.path.splitext(filename_with_extension)[0] + ".txt"
            path = r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\apples_learning\trainUnlabeled\labels" + '\\' + filename_txt
            print(path)
            with open(path, 'a') as file:
                res_string = f"0 {x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}"
                file.write(res_string + '\n')