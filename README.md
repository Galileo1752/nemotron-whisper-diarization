# nemotron-whisper-diarization

結合 **NVIDIA Nemotron 3 Diarization**（誰、何時說話）與 **WhisperS2T**（說了什麼 + 時間戳）的講者標註逐字稿管線。

## 流程概念

```
音訊 (16kHz mono wav)
    │
    ├─► [Nemotron 3 Diarization] ──► 語者時間段: (start, end, speaker_id)
    │
    └─► [WhisperS2T] ─────────────► 逐字文字 + 字詞時間戳: (start, end, word)
                                          │
                                          ▼
                          [合併對齊] 依時間戳把每個字詞指派給重疊最多的語者
                                          │
                                          ▼
                          講者標註逐字稿: "SPEAKER_00: ...", "SPEAKER_01: ..."
```

核心觀念：**Nemotron 負責「誰、何時」，WhisperS2T 負責「說了什麼＋精確時間」**，最後用時間重���度把字詞指派給語者（Nemotron 本身不含 forced alignment，字詞級對齊由 ASR 提供）。

## 安裝

```bash
bash install.sh
# 或
pip install -r requirements.txt
```

> 建議 Python 3.10+ 與 NVIDIA GPU (CUDA)。系統需安裝 ffmpeg。

## 使用

```bash
python main.py
```

或在程式中呼叫：

```python
from main import run
run("meeting.mp3", lang="zh")
```

### 輸出範例

```
===== 講者標註逐字稿 =====
[00:00-00:04] speaker_0: 大家好 今天我們來討論第三季的營運報告
[00:04-00:09] speaker_1: 好的 我先分享一下業績數字
[00:09-00:13] speaker_0: 沒問題 請開始
```

## 檔案結構

| 檔案 | 說明 |
|---|---|
| `audio_utils.py` | 音訊前處理，統一為 16kHz 單聲道 |
| `diarizer.py` | Nemotron 3 Diarization 封裝（誰、何時） |
| `transcriber.py` | WhisperS2T 封裝（說了什麼 + 時間戳） |
| `align.py` | 合併對齊，把字詞指派給語者 |
| `main.py` | 主管線，串接整條流程 |

## 注意事項

- **API 名稱**：Nemotron 3 / WhisperS2T 的載入函式請以官方最新文件為準，只需改 `diarizer.py`、`transcriber.py`。
- **重疊語音**：Nemotron 支援最多 8 人同時重疊；`assign_speaker` 取重疊最多的語者，必要時可改為多標籤。
- **字詞時間戳**：對齊品質高度依賴 Whisper word timestamps，務必開啟。
- **取樣率**：兩模型皆吃 16kHz mono，前處理務必統一，否則時間戳會對不齊。
- **串流即時**：本範例為離線批次流程；即時版需改為 chunk 串流架構。

## 參考來源

- [nvidia/Nemotron-3-Diarization · Hugging Face](https://huggingface.co/nvidia/Nemotron-3-Diarization)
- [NVIDIA NeMo 文件](https://docs.nvidia.com/deeplearning/nemo/)
- [WhisperS2T GitHub](https://github.com/shashikg/WhisperS2T)

## 授權

MIT
