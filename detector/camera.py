import cv2
import torch

# 모델 로드 (YOLOv5s, 로컬에서 불러오기)
model = torch.hub.load('/home/huro/yolov5', 'yolov5s', source='local')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# USB 카메라 열기 (보통 /dev/video0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 추론
    results = model(frame)

    # 결과 이미지 (Bounding Box 포함)
    annotated_frame = results.render()[0]

    # 화면에 출력
    cv2.imshow("YOLOv5 Detection", annotated_frame)

    # ESC 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()