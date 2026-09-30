from typing import List, Dict, Tuple

DiarSegment = Tuple[float, float, str]


def _overlap(a_start, a_end, b_start, b_end) -> float:
    """回傳兩區間重疊秒數。"""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(word: Dict, diar_segments: List[DiarSegment]) -> str:
    """把一個字詞指派給『時間重疊最多』的語者段。"""
    best_spk, best_ov = "UNKNOWN", 0.0
    for start, end, spk in diar_segments:
        ov = _overlap(word["start"], word["end"], start, end)
        if ov > best_ov:
            best_ov, best_spk = ov, spk
    return best_spk


def merge(words: List[Dict], diar_segments: List[DiarSegment]) -> List[Dict]:
    """把字詞逐一標上語者，再合併連續同一語者為一段對話。"""
    tagged = []
    for w in words:
        tagged.append({**w, "speaker": assign_speaker(w, diar_segments)})

    # 合併連續相同語者
    merged: List[Dict] = []
    for w in tagged:
        if merged and merged[-1]["speaker"] == w["speaker"]:
            merged[-1]["end"] = w["end"]
            merged[-1]["text"] += (" " if w["word"].strip().isascii() else "") + w["word"]
        else:
            merged.append({
                "speaker": w["speaker"],
                "start": w["start"],
                "end": w["end"],
                "text": w["word"],
            })
    return merged
