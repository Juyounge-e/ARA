# -*- coding: utf-8 -*-
import cv2
import torch
import os
from pathlib import Path
import datetime

# ===============================
# 1. 모델 로드 (YOLOv5)
# ===============================
# 로컬 yolov5 디렉토리와 모델 파일 경로
YOLO_PATH = "/home/huro/Desktop/ara/yolov5"   # YOLOv5 코드 디렉토리
MODEL_NAME = "yolov5s"                        # yolov5s, yolov5m, yolov5l, yolov5x 등
model = torch.hub.load(YOLO_PATH, MODEL_NAME, source='local')

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# ===============================
# 2. 이미지 경로
# ===============================
IMAGE_PATH = "test/realTest1.jpg"       # 분석할 사진
SAVE_DIR = "food_result_img" "
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# 3. 이미지 불러오기
# ===============================
if not os.path.exists(IMAGE_PATH):
    print("이미지 경로가 잘못되었습니다:", IMAGE_PATH)
    exit()

img = cv2.imread(IMAGE_PATH)

# ===============================
# 4. 추론
# ===============================
results = model(img)

# 바운딩 박스 포함된 이미지
annotated = results.render()[0]

# ===============================
# 5. 탐지 정보 출력
# ===============================
for det in results.xyxy[0]:  
    x1, y1, x2, y2, conf, cls = det.tolist()
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    print(f"탐지됨 → 클래스: {model.names[int(cls)]}, 좌표: ({cx}, {cy}), 신뢰도: {conf:.2f}")

# ===============================
# 6. 결과 저장
# ===============================
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(SAVE_DIR, f"detect_{timestamp}.jpg")
cv2.imwrite(save_path, annotated)

print(f" 탐지 결과 저장 완료: {save_path}")

# 결과 화면도 표시
cv2.imshow("YOLOv5 Image Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()