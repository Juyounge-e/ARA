# -*- coding: utf-8 -*-
import cv2, torch

print("모델 로드 중...")
model = torch.hub.load('/home/huro/Desktop/ara/yolov5', 'yolov5s', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print("모델 준비 완료 ")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results.render()[0]

    cv2.imshow("YOLOv5", annotated)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()