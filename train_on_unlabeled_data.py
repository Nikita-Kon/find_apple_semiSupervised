from ultralytics import YOLO
import os
if __name__ == "__main__":

    # Load the previously trained YOLO model
    model = YOLO("runs/train/exp/weights/best.pt")

    # Continue training with the new dataset
    model.train(data="data_unlabeled.yaml", epochs=10, imgsz=256, device=0)

    temp_model_path = "runs/detect/train/weights/best.pt"
    final_model_path = "best.pt"
    if os.path.exists(temp_model_path):
        os.rename(temp_model_path, final_model_path)