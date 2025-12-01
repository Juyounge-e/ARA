import numpy as np
import sounddevice as sd
import whisper

SAMPLE_RATE = 16000  # Whisper 권장 샘플레이트

def record_audio(seconds: int = 5):
    print(f"{seconds}초 동안 녹음합니다. 말하세요!")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()  # 녹음 끝날 때까지 대기
    print("녹음 완료, 인식 중...\n")
    return audio.squeeze()  # (N, 1) → (N,)

def main():
    # tiny / base / small / medium / large 중 선택
    # 처음에는 가벼운 "base"나 "small" 추천
    print("모델 로딩 중...(처음 한 번은 조금 걸릴 수 있음)")
    model = whisper.load_model("base")  # or "small"
    print("모델 로딩 완료!\n")

    while True:
        try:
            sec = 5  # 5초씩 잘라서 인식
            audio = record_audio(seconds=sec)

            # numpy 배열 직접 전달, M1 CPU라 fp16=False
            result = model.transcribe(
                audio,
                language="ko",
                fp16=False
            )

            text = result.get("text", "").strip()
            if text:
                print(">>> 인식 결과:", text, "\n")
            else:
                print(">>> (아무 말도 인식하지 못했어요)\n")

        except KeyboardInterrupt:
            print("\n종료합니다.")
            break

if __name__ == "__main__":
    main()