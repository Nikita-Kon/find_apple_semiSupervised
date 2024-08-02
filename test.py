import cv2
from ultralytics import YOLO

model = YOLO(r"C:\Users\Aser\PycharmProjects\torch-yolo\Apples_Detection\runs\train\exp3\weights\best.pt")
frame = cv2.imread(r"C:\Users\Aser\PycharmProjects\torch-yolo\Apples_Detection\grid_image_orange.png")

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

# Display the resulting frame
cv2.imshow('Webcam Feed with YOLO Detection', frame)
# save photo if you need
# cv2.imwrite(r'C:\Users\Aser\PycharmProjects\torch-yolo\Apples_Detection\find_grid_image_orange.png', frame)

cv2.waitKey(0)
cv2.destroyAllWindows()


