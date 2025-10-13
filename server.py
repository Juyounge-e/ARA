import socket

def wait_for_signal(host="0.0.0.0", port=4000):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((host, port))
    print(f"waiting for signal (port {port})")

    while True:
        data, addr = sock.recvfrom(1024)
        message = data.decode("utf-8")
        if message.strip().lower() == "start":
            print("taking initial signal")
            return True