# -*- coding: utf-8 -*-
import cv2
import torch
import datetime

# ===============================
# 1. GStreamer 파이프라인
# ===============================
def gstreamer_pipeline(
    capture_width=1280, capture_height=720,
    display_width=1280, display_height=720,
    framerate=30, flip_method=0
):
    return (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=(int){}, height=(int){}, framerate=(fraction){}/1 ! "
        "videoconvert ! "
        "videoscale ! "
        "video/x-raw, width=(int){}, height=(int){} ! "
        "appsink"
        .format(capture_width, capture_height, framerate, display_width, display_height)
    )

# ===============================
# 2. 모델 로드
# ===============================
print("YOLOv5 모델 불러오는 중...")
model = torch.hub.load('/home/huro/Desktop/ara/yolov5', 'yolov5s', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print("모델 준비 완료")

# ===============================
# 3. 카메라 열기
# ===============================
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

# 동영상 저장 준비
fourcc = cv2.VideoWriter_fourcc(*'XVID')
now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out = cv2.VideoWriter(f"detections/video_{now}.avi", fourcc, 20.0,
                      (1280, 720))

frame_count = 0

# ===============================
# 4. 실시간 추론 루프
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    # YOLO 추론
    results = model(frame)
    annotated = results.render()[0]

    # 동영상 저장
    out.write(annotated)

    # 50프레임마다 스냅샷 저장
    if frame_count % 50 == 0:
        cv2.imwrite(f"detections/snapshot_{now}_{frame_count}.jpg", annotated)

    # 탐지된 객체 출력
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        print("탐지됨 → 클래스: {}, 좌표: ({}, {}), 신뢰도: {:.2f}".format(
            model.names[int(cls)], cx, cy, conf
        ))

    # 결과 화면 출력
    cv2.imshow("YOLOv5 Detection (Video)", annotated)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

    frame_count += 1

# ===============================
# 5. 종료 처리
# ===============================
cap.release()
out.release()
cv2.destroyAllWindows()