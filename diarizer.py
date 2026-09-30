from typing import List, Tuple

# 統一輸出格式: (start_sec, end_sec, speaker_label)
DiarSegment = Tuple[float, float, str]


class NemotronDiarizer:
    def __init__(self, model_name: str = "nvidia/Nemotron-3-Diarization", device: str = "cuda"):
        # === 依官方 NeMo API 載入 Nemotron 3 Diarization ===
        # 範例（實際名稱以官方文件為準）:
        #   from nemo.collections.asr.models import SortformerEncLabelModel
        #   self.model = SortformerEncLabelModel.from_pretrained(model_name).to(device).eval()
        from nemo.collections.asr.models import SortformerEncLabelModel  # 名稱請以官方為準
        self.model = SortformerEncLabelModel.from_pretrained(model_name)
        self.model = self.model.to(device).eval()

    def diarize(self, audio_path: str) -> List[DiarSegment]:
        """回傳語者時間段清單: [(start, end, 'speaker_0'), ...]"""
        # 多數 NeMo diarization 模型提供 diarize() 並輸出 RTTM 或 segment 物件。
        # 這裡假設回傳可轉為 (start, end, label) 的結果。
        predictions = self.model.diarize(audio=[audio_path])  # 介面以官方為準

        segments: List[DiarSegment] = []
        # predictions 的結構可能是每檔一組 "start end speaker" 字串；以下為通用解析
        for item in predictions[0]:
            # 例如 item = "0.50 3.20 speaker_0"
            start, end, spk = item.split()
            segments.append((float(start), float(end), spk))

        return sorted(segments, key=lambda x: x[0])


def parse_rttm(rttm_path: str) -> List[DiarSegment]:
    """備用: 若模型輸出的是 RTTM 檔，用此函式解析。"""
    segments: List[DiarSegment] = []
    with open(rttm_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if not p or p[0] != "SPEAKER":
                continue
            start = float(p[3])
            dur = float(p[4])
            spk = p[7]
            segments.append((start, start + dur, spk))
    return sorted(segments, key=lambda x: x[0])
