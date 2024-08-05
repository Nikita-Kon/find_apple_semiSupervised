# Train no pretrained yolo model to recognise apples using gpu for this "device=0'
from ultralytics import YOLO
import os

def trainOnLabeledData():
    if __name__ == "__main__":
        model = YOLO("yolov8n.yaml")
        model.train(data="data_labeled.yaml", epochs=70, imgsz=256, device=0, project="runs/train",  name="exp",  pretrained=False)
# to change best.pt directory into project root
def changeModelDir():
    trained_model_path = "runs/detect/train/weights/best.pt"
    destination = "best.pt"
    if os.path.exists(destination):
        os.remove(destination)
    if os.path.exists(trained_model_path):
        os.rename(trained_model_path, destination)

def trainOnMixedData():
    if __name__ == "__main__":
        # Load the previously trained YOLO model
        model = YOLO("best.pt")
        #train with the new dataset
        model.train(data="data_labeled.yaml", epochs=70, imgsz=256, device=0 )
        # change model directory into project root
        changeModelDir()