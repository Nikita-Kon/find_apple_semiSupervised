import cv2
from ultralytics import YOLO
import numpy as np
def min_max_normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)

model = YOLO(r"best.pt")
frame = cv2.imread(r"C:\Users\Aser\Desktop\grid_image.png")

result = model.predict(source=frame)

boxes = result[0].boxes
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    confidence = box.conf[0]
    class_id = box.cls[0]
    class_name = model.names[int(class_id)]
        # Draw the bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    label = f"{model.names[int(class_id)]}: {confidence:.2f}"
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    data = [x1, y1, x2, y2, 0, 256]
    data = min_max_normalize(data)
    x1, y1, x2, y2 = data[:4]
    print('confidence - ', confidence, "cord - ", x1, y1, x2, y2)
# Display the resulting frame
cv2.imshow('Webcam Feed with YOLO Detection', frame)
# save photo if you need
# cv2.imwrite(r'C:\Users\Aser\PycharmProjects\torch-yolo\find_apple_semiSupervised\img_grid.jpg', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()


