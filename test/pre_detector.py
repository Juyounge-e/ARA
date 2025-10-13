# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ===============================
# 1. 모델 불러오기
# ===============================
MODEL_PATH = "best-fp16.tflite"  # 여기에 tflite 파일 경로 넣기
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 크기 가져오기
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# ===============================
# 2. 카메라 열기
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

# 클래스 이름 (필요시 수정)
class_names = ["person", "cup", "bottle", "plate", "food"]  # 예시

# ===============================
# 3. 실시간 추론 루프
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    # --- 전처리 ---
    img = cv2.resize(frame, (input_width, input_height))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # --- 추론 ---
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # --- 후처리 (YOLO 형식 가정) ---
    # output_data: [N, 6] → [x1, y1, x2, y2, conf, class]
    for det in output_data[0]:
        x1, y1, x2, y2, conf, cls = det
        if conf < 0.3:  # confidence threshold
            continue

        # 좌표 정수형 변환 (원본 크기에 맞게 비율 조정)
        h, w, _ = frame.shape
        x1 = int(x1 / input_width * w)
        y1 = int(y1 / input_height * h)
        x2 = int(x2 / input_width * w)
        y2 = int(y2 / input_height * h)

        # 중앙 좌표
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 바운딩박스 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "{} {:.2f}".format(class_names[int(cls)] if int(cls) < len(class_names) else str(int(cls)), conf)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 중앙점 찍기
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        print(f"탐지됨 → 클래스: {class_names[int(cls)] if int(cls) < len(class_names) else str(int(cls))}, 좌표: ({cx}, {cy}), 신뢰도: {conf:.2f}")

    # --- 결과 출력 ---
    cv2.imshow("TFLite Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()