# -*- coding: utf-8 -*-
import cv2
import torch
import datetime
import os

# 저장 폴더 생성
os.makedirs("detections", exist_ok=True)

# 모델 로드
model = torch.hub.load('/home/huro/Desktop/ara/yolov5', 'yolov5s', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# USB 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

# 저장할 동영상 파일 설정 (날짜 기반 이름)
fourcc = cv2.VideoWriter_fourcc(*'XVID')   # avi, 속도 안정적
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
frame_w, frame_h = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(f"detections/video_{now}.avi", fourcc, 20.0, (frame_w, frame_h))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    # YOLO 추론
    results = model(frame)
    annotated = results.render()[0]  # 시각화된 프레임

    # 동영상 저장
    out.write(annotated)

    # 50프레임마다 스냅샷 저장
    if frame_count % 50 == 0:
        cv2.imwrite(f"detections/snapshot_{now}_{frame_count}.jpg", annotated)

    # 탐지된 객체 출력 (10프레임마다만 출력 → 속도 최적화)
    if frame_count % 10 == 0:
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls = det.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            print("탐지됨 → 클래스: {}, 좌표: ({}, {}), 신뢰도: {:.2f}".format(
                model.names[int(cls)], cx, cy, conf
            ))

    # 결과 화면 표시
    cv2.imshow("YOLOv5 Detection (Video)", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()