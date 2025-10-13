import cv2
import torch

# YOLO 로드 (로컬 yolov5 clone 경로 지정)
model = torch.hub.load('/home/huro/yolov5', 'yolov5s', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

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