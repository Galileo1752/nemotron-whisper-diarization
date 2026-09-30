import soundfile as sf
import librosa
import numpy as np


def prepare_audio(input_path: str, output_path: str = "prepared_16k_mono.wav") -> str:
    """將任意音訊轉為 16kHz、單聲道 wav（兩個模型的共同輸入格式）。"""
    y, sr = librosa.load(input_path, sr=16000, mono=True)
    sf.write(output_path, y, 16000, subtype="PCM_16")
    return output_path
