# Train no pretrained yolo model to recognise apples using gpu for this "device=0'
import time
import  make_pseudo_labels as mk
from ultralytics import YOLO
import os
import shutil

class datasetInfo:
    def __init__(self):
        self.trainAmount = 0
        self.startTime = 0
        self.endTime = 0
    def printDatasetInfo(self):
        time_hours = (self.endTime - self.startTime) / 3600
        print(f"Amount of trainings-{self.trainAmount:.2f}, Time for trainings-{time_hours:.2f}")
data_info = datasetInfo

# to change best.pt directory into project root
train_score = 0
def changeModelDir():
    global  data_info
    data_info.trainAmount += 1
    global train_score
    train_score += 1
    if train_score == 1:
        trained_model_path = "runs/detect/train/weights/best.pt"
    else:
        trained_model_path = "runs/detect/train" + str(train_score) + "/weights/best.pt"
    destination = "best.pt"
    if os.path.exists(destination):
        os.remove(destination)
    if os.path.exists(trained_model_path):
        shutil.copy2(trained_model_path, destination)

def trainOnLabeledData():
    global data_info
    data_info.startTime = time.time()
    if os.path.exists(r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\runs\detect"):
        shutil.rmtree(r"C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\runs\detect")
    model = YOLO("yolov8n.yaml")
    model.train(data="data_labeled.yaml", epochs=60, imgsz=256, device=0, pretrained=False)
    changeModelDir()

def trainOnMixedData():
    # Load the previously trained YOLO model
    model = YOLO("best.pt")
    #train with the new dataset
    model.train(data="data_labeled.yaml", epochs=60, imgsz=256, device=0 )
    # change model directory into project root
    changeModelDir()
    # update end time
    global data_info
    data_info.endTime = time.time()

