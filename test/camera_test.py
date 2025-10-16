import cv2
import time
import os
import datetime

# 저장 디렉토리
save_dir = "detections"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("카메라 준비 중. 5초 후에 사진 촬영")
time.sleep(5) 

ret, frame = cap.read()
if ret:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"capture_{timestamp}.jpg")
    cv2.imwrite(save_path, frame)
    print(f"사진 저장 완료: {save_path}")
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(7000) 
else:
    print("프레임을 읽을 수 없습니다.")

cap.release()
cv2.destroyAllWindows()