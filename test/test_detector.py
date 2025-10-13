import cv2
import torch
import sys
from pathlib import Path

# yolov5 폴더를 Python path에 추가
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] / "yolov5"  # ~/Desktop/ARA/yolov5
sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import non_max_suppression

# 모델 로드
device = select_device('')
model = DetectMultiBackend(str(ROOT / 'yolov5s.pt'), device=device)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    # YOLO 추론
    results = model(frame)
    annotated = results.render()[0]

    # 첫 번째 탐지된 물체 정보 가져오기
    if len(results.xyxy[0]) > 0:
        x1, y1, x2, y2, conf, cls = results.xyxy[0][0].tolist()
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        print(f"탐지됨 → 클래스: {model.names[int(cls)]}, 좌표: ({cx}, {cy}), 신뢰도: {conf:.2f}")

    cv2.imshow("YOLO Detection", annotated)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()