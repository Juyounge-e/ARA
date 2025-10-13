# -*- coding: utf-8 -*-
import cv2
import torch
import datetime

# 모델 로드
model = torch.hub.load('/home/huro/Desktop/ara/yolov5', 'yolov5s', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# USB 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" 카메라 열기 실패")
    exit()

# 저장할 동영상 파일 설정 (날짜 기반 이름)
fourcc = cv2.VideoWriter_fourcc(*'XVID')   # or 'mp4v'
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out = cv2.VideoWriter(f"detections/video_{now}.avi", fourcc, 20.0, 
                      (int(cap.get(3)), int(cap.get(4))))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print(" 프레임 읽기 실패")
        break

    # YOLO 추론
    results = model(frame)
    annotated = results.render()[0]

    # 프레임 저장 (동영상으로 기록)
    out.write(annotated)

    # 50프레임마다 스냅샷 이미지 저장
    if frame_count % 50 == 0:
        cv2.imwrite(f"detections/snapshot_{now}_{frame_count}.jpg", annotated)

    # 탐지된 객체 출력
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        print("탐지됨 → 클래스: {}, 좌표: ({}, {}), 신뢰도: {:.2f}".format(
            model.names[int(cls)], cx, cy, conf
        ))

    # 결과 화면 표시
    cv2.imshow("YOLOv5 Detection (Video)", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키
        break

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()