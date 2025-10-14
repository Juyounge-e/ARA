import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,       # 얼굴 하나만 탐지
    refine_landmarks=True, # 눈/입 세밀하게 탐지
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB 변환
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 추론
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # 입 부분 좌표 (Mediapipe FaceMesh 기준 61~88번이 입)
                for idx in range(61, 89):
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

                # 입 중앙 좌표 계산 (예: 78번이 입 중앙 근처)
                mouth_x = int(face_landmarks.landmark[78].x * w)
                mouth_y = int(face_landmarks.landmark[78].y * h)
                cv2.circle(frame, (mouth_x, mouth_y), 4, (0, 255, 0), -1)
                print(f"입 중앙 좌표: ({mouth_x}, {mouth_y})")

        cv2.imshow("Mediapipe FaceMesh", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()