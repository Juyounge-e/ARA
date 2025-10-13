import json
import socket
from server import wait_for_signal
from detector import detect_once

CONTROL_IP = "192.168.0.50"   # 제어팀 보드/PC IP
CONTROL_PORT = 5005           # 제어팀 수신 포트

def send_coordinates(data):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    message = json.dumps(data).encode("utf-8")
    sock.sendto(message, (CONTROL_IP, CONTROL_PORT))
    print(f"좌표 전송: {message}")

if __name__ == "__main__":
    while True:
        # 1. 제어팀 신호 대기
        if wait_for_signal():
            # 2. YOLO 한 번 실행
            detections = detect_once()
            if detections:
                # 3. 좌표 JSON으로 전송
                send_coordinates(detections)