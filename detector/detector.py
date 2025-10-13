import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

def detect_once():
    cap = cv2.VideoCapture(0)  
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("failed to grab frame")
        return None

    results = model(frame, verbose=False)
    detections = []

    for r in results[0].boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        label = model.names[int(r.cls[0])]
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2

        detections.append({
            "object": label,
            "x_center": x_center,
            "y_center": y_center
        })

    return detections