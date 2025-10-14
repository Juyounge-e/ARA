# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os

# ===============================
# 1. 모델 & 클래스 불러오기
# ===============================
MODEL_PATH = "best-fp16.tflite"
IMAGE_PATH = "test/food.jpg"   
SAVE_PATH = "detections/result.jpg"   

# 클래스 이름 (11개)
class_names = [
    "마늘", "감자", "달걀", "양파", "닭고기",
    "돼지고기", "대파", "소고기", "김치", "햄", "콩나물"
]

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height, input_width = input_details[0]['shape'][1:3]

# ===============================
# 2. 이미지 불러오기
# ===============================
if not os.path.exists(IMAGE_PATH):
    print(" 이미지 경로가 잘못되었습니다:", IMAGE_PATH)
    exit()

frame = cv2.imread(IMAGE_PATH)
h, w, _ = frame.shape

# 전처리
img = cv2.resize(frame, (input_width, input_height))
img = np.expand_dims(img, axis=0).astype(np.float32)

# ===============================
# 3. 추론
# ===============================
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

# ===============================
# 4. 탐지 결과 처리
# ===============================
for det in output_data:
    if len(det) < 6:
        continue

    x1, y1, x2, y2 = det[:4]
    probs = det[5:]
    cls = int(np.argmax(probs))
    conf = probs[cls]

    if conf < 0.5:  # confidence threshold
        continue

    # 좌표 변환
    if 0 <= x1 <= 1 and 0 <= x2 <= 1:
        x1, x2 = int(x1 * w), int(x2 * w)
        y1, y2 = int(y1 * h), int(y2 * h)
    else:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # 중앙 좌표
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # 클래스 이름
    cls_name = class_names[cls]

    # 바운딩 박스
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

    print(f"탐지됨 → {cls_name}, 좌표: ({cx}, {cy}), 신뢰도: {conf:.2f}")

# ===============================
# 5. 결과 저장
# ===============================
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
cv2.imwrite(SAVE_PATH, frame)
print(f" 탐지 결과 저장 완료: {SAVE_PATH}")

# 화면도 표시 (옵션)
cv2.imshow("Detection Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()