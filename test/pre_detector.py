# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# ===============================
# 1. 모델 불러오기
# ===============================
MODEL_PATH = "best-fp16.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height, input_width = input_details[0]['shape'][1:3]

# ===============================
# 2. 클래스 이름 정의
# ===============================
class_names = [
    "마늘", "감자", "달걀", "양파", "닭고기",
    "돼지고기", "대파", "소고기", "김치", "햄", "콩나물"
]

# ===============================
# 3. NMS 함수
# ===============================
def nms(boxes, scores, iou_threshold=0.4):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        cur = idxs[0]
        keep.append(cur)
        if len(idxs) == 1:
            break

        cur_box = boxes[cur]
        rest_boxes = boxes[idxs[1:]]

        # IoU 계산
        x1 = np.maximum(cur_box[0], rest_boxes[:, 0])
        y1 = np.maximum(cur_box[1], rest_boxes[:, 1])
        x2 = np.minimum(cur_box[2], rest_boxes[:, 2])
        y2 = np.minimum(cur_box[3], rest_boxes[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_cur = (cur_box[2] - cur_box[0]) * (cur_box[3] - cur_box[1])
        area_rest = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        iou = inter / (area_cur + area_rest - inter + 1e-6)

        # IoU가 작은 것만 남김
        idxs = idxs[1:][iou < iou_threshold]
    return keep

# ===============================
# 4. 카메라 열기
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

frame_count = 0

# ===============================
# 5. 실시간 추론 루프
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 전처리 ---
    img = cv2.resize(frame, (input_width, input_height))
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # --- 추론 ---
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    boxes, scores, classes = [], [], []

    for det in output_data:
        if len(det) < 6:
            continue

        x1, y1, x2, y2 = det[:4]
        probs = det[5:]
        cls = int(np.argmax(probs))
        conf = probs[cls]

        if conf < 0.5:  # confidence threshold
            continue

        h, w, _ = frame.shape
        if 0 <= x1 <= 1 and 0 <= x2 <= 1:
            x1, x2 = int(x1 * w), int(x2 * w)
            y1, y2 = int(y1 * h), int(y2 * h)
        else:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        boxes.append([x1, y1, x2, y2])
        scores.append(conf)
        classes.append(cls)

    # --- NMS 적용 후 시각화 ---
    if boxes:
        boxes = np.array(boxes)
        scores = np.array(scores)
        classes = np.array(classes)

        keep = nms(boxes, scores, iou_threshold=0.4)

        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            conf, cls = scores[i], classes[i]
            cls_name = class_names[cls]

            # 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.circle(frame, ((x1+x2)//2, (y1+y2)//2), 4, (0, 0, 255), -1)

            # 매 10프레임마다만 터미널 출력
            if frame_count % 10 == 0:
                print(f"탐지됨 → {cls_name}, 좌표: ({(x1+x2)//2}, {(y1+y2)//2}), 신뢰도: {conf:.2f}")

    # --- 결과 화면 표시 ---
    cv2.imshow("TFLite Detection (NMS)", frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
        break

cap.release()
cv2.destroyAllWindows()