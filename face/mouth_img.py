import cv2
import mediapipe as mp
import os
from datetime import datetime

# Mediapipe 초기화
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

save_dir = "face/result"
os.makedirs(save_dir, exist_ok=True)

# 분석할 이미지 경로
IMAGE_PATH = "face/closed.png" 

# 이미지 불러오기
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(" 이미지 로드 실패")
    exit()

# RGB 변환
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 얼굴 랜드마크 추출
with mp_face.FaceMesh(
    static_image_mode=True,   #
    max_num_faces=1,
    min_detection_confidence=0.5
) as face_mesh:

    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        print("얼굴을 찾지 못했습니다.")
    else:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape

            # 입 좌표 (MediaPipe FaeMesh에서 입 인덱스: 61~80, 308 )
    
            mouth_indices = [
                61, 291,  # 입 양쪽 끝
                13, 14,   # 윗입술 중앙, 아랫입술 중앙
                78, 308,  # 입 좌/우 위쪽
                81, 311   # 입 좌/우 아래쪽
            ]

            for idx in mouth_indices:
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                print(f"입 좌표 {idx}: ({x}, {y})")

            
                cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

        # 결과 이미지 
        cv2.imshow("Face with Mouth Landmarks", image)
        

        # 결과 이미지 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"mouth_landmarks_{timestamp}.jpg")
        cv2.imwrite(save_path, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
