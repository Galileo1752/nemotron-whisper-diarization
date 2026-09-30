#!/usr/bin/env bash
# 建議使用 Python 3.10+ 與 NVIDIA GPU (CUDA)
set -e

pip install "nemo_toolkit[asr]" torch torchaudio
pip install whisper-s2t

# 音訊前處理
pip install soundfile librosa ffmpeg-python
# 系統需安裝 ffmpeg（Linux: apt install ffmpeg）
