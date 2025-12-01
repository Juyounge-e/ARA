import numpy as np
import sounddevice as sd
import whisper

SAMPLE_RATE = 16000

FOOD_KEYWORDS = {
    "RICE": ["밥", "공깃밥", "흰쌀밥"],
    "SOUP": ["국", "국물", "된장국", "미역국"],
    "SIDE": ["반찬", "김치", "나물"],
}
STOP_KEYWORDS = ["멈춰", "그만", "스톱", "stop"]
HOME_KEYWORDS = ["처음으로", "원위치", "처음 위치", "기본 위치"]


def parse_command(text: str): 
    """
    - STOP/HOME/MOVE 키워드는 dict로 리턴
    - 그 외 문장은 텍스트 그대로(str) 리턴
    """
    text = text.strip()
    if not text:
        return ""  # 빈 문자열 그대로

    # 1) 정지 명령
    if any(kw in text for kw in STOP_KEYWORDS):
        return {"type": "STOP"}

    # 2) 홈/원위치 명령
    if any(kw in text for kw in HOME_KEYWORDS):
        return {"type": "HOME"}

    # 3) 밥/국/반찬 위치 이동
    for target, keywords in FOOD_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return {
                "type": "MOVE",
                "target": target,   # "RICE" / "SOUP" / "SIDE"
                "raw_text": text,   # 디버깅용
            }

    # 4) 명령이 아니면 그냥 문장 그대로 리턴
    return text


def record_audio(seconds: int = 3):
    print(f"{seconds}초 동안 녹음합니다. 말하세요!")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("녹음 완료, 인식 중...\n")
    return audio.squeeze()


def main():
    print("모델 로딩 중...")
    model = whisper.load_model("base")
    print("모델 로딩 완료!\n")

    while True:
        try:
            audio = record_audio(seconds=3)

            result = model.transcribe(audio, language="ko", fp16=False)
            text = result.get("text", "").strip()
            print("[STT 결과]", text)

            parsed = parse_command(text)

            # dict면 → 명령, str이면 → 일반 텍스트
            if isinstance(parsed, dict):
                print("→ 명령으로 인식됨:", parsed, "\n")
                # TODO: 여기서 제어팀으로 전송
                # send_to_controller(parsed)
            else:
                print("→ 일반 텍스트로 취급:", text, "\n")
                # 원하면 여기서도 그대로 넘길 수 있음
                # send_to_controller({"type": "RAW", "text": parsed})

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break


if __name__ == "__main__":
    main()