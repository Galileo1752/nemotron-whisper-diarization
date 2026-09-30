from audio_utils import prepare_audio
from diarizer import NemotronDiarizer
from transcriber import WhisperTranscriber
from align import merge


def format_time(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def run(input_audio: str, lang: str = "zh"):
    # 1) 前處理
    audio = prepare_audio(input_audio)

    # 2) 講者分離（誰、何時）
    print("▶ 執行 Nemotron 3 Diarization ...")
    diarizer = NemotronDiarizer()
    diar_segments = diarizer.diarize(audio)

    # 3) 語音轉文字（說了什麼 + 時間戳）
    print("▶ 執行 WhisperS2T 轉錄 ...")
    transcriber = WhisperTranscriber()
    words = transcriber.transcribe(audio, lang=lang)

    # 4) 合併對齊
    print("▶ 合併對齊 ...")
    conversation = merge(words, diar_segments)

    # 5) 輸出講者標註逐字稿
    print("\n===== 講者標註逐字稿 =====")
    for turn in conversation:
        ts = f"[{format_time(turn['start'])}-{format_time(turn['end'])}]"
        print(f"{ts} {turn['speaker']}: {turn['text']}")

    return conversation


if __name__ == "__main__":
    run("meeting.mp3", lang="zh")
