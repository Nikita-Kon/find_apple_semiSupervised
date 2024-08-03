from ultralytics import YOLO
import os

def changeModelDir():
    trained_model_path = "runs/detect/train/weights/best.pt"
    destination = "best.pt"
    if os.path.exists(destination):
        os.remove(destination)
    if os.path.exists(trained_model_path):
        os.rename(trained_model_path, destination)

if __name__ == "__main__":

    # Load the previously trained YOLO model
    model = YOLO("best.pt")

    # Continue training with the new dataset
    model.train(data="data_unlabeled.yaml", epochs=70, imgsz=256, device=0)

    changeModelDir()