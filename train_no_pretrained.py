# Train no pretrained yolo model to recognise apples using gpu for this "device=0'
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n.yaml")

    model.train(data="data.yaml", epochs=100, imgsz=640, device=0, project="runs/train",  name="exp",  pretrained=False)
