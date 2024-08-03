# Train no pretrained yolo model to recognise apples using gpu for this "device=0'
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")

    model.train(data="data_labeled.yaml", epochs=70, imgsz=256, device=0, project="runs/train",  name="exp",  pretrained=False)