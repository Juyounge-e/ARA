# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ===============================
# 1. 모델 불러오기
# ===============================
MODEL_PATH = "best-fp16.tflite"  # tflite 모델 파일 경로
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 입력 크기
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# ===============================
# 2. 클래스 이름 정의
# ===============================
class_names = [
    "마늘", "감자", "달걀", "양파", "닭고기", 
    "돼지고기", "대파", "소고기", "김치", "햄", "콩나물"
]

# ===============================
# 3. 카메라 열기
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

# ===============================
# 4. 실시간 추론 루프
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

    # --- 후처리 ---
    for det in output_data[0]:
        values = det.tolist()
        if len(values) < 6:
            continue

        x1, y1, x2, y2 = values[:4]
        conf = values[4]
        cls = int(values[5])

        if conf < 0.3:  # confidence threshold
            continue

        h, w, _ = frame.shape

        # 좌표 변환 (정규화 여부 자동 처리)
        if 0 <= x1 <= 1 and 0 <= x2 <= 1:
            x1 = int(x1 * w)
            x2 = int(x2 * w)
            y1 = int(y1 * h)
            y2 = int(y2 * h)
        else:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # 중앙 좌표
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 클래스 이름 가져오기
        cls_name = class_names[cls] if cls < len(class_names) else str(cls)

        # 바운딩 박스
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "{} {:.2f}".format(cls_name, conf)
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        print(f"탐지됨 → 클래스: {cls_name}, 좌표: ({cx}, {cy}), 신뢰도: {conf:.2f}")

    # --- 결과 출력 ---
    cv2.imshow("TFLite Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 종료
        break

cap.release()
cv2.destroyAllWindows()