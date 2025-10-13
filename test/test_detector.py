# -*- coding: utf-8 -*-
import cv2
import torch
import os
from datetime import datetime

# 모델 로드 (로컬 yolov5 디렉토리에서 yolov5s.pt 불러오기)
model = torch.hub.load('/home/huro/Desktop/ARA/yolov5', 'yolov5s', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 저장 폴더 생성
save_dir = "detections"
os.makedirs(save_dir, exist_ok=True)

# USB 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    # YOLO 추론
    results = model(frame)

    # 결과 시각화 (바운딩박스 포함된 프레임)
    annotated = results.render()[0]

    # 탐지된 객체 정보 출력 + 저장
    if len(results.xyxy[0]) > 0:  # 탐지된 객체가 있을 때만 저장
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"detect_{now}.jpg")
        cv2.imwrite(save_path, annotated)
        print(f"탐지 결과 저장됨: {save_path}")

        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            print("탐지됨 → 클래스: {}, 좌표: ({}, {}), 신뢰도: {:.2f}".format(
                model.names[int(cls)], cx, cy, conf
            ))

    cv2.imshow("YOLOv5 Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
        break

cap.release()
cv2.destroyAllWindows()