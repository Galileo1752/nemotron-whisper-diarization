from typing import List, Dict


class WhisperTranscriber:
    def __init__(self, model_size: str = "large-v3", device: str = "cuda",
                 compute_type: str = "int8_float16"):
        import whisper_s2t
        # WhisperS2T 的載入 API 依版本略有不同，常見為 load_model
        self.model = whisper_s2t.load_model(
            model_identifier=model_size,
            backend="CTranslate2",
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio_path: str, lang: str = "zh") -> List[Dict]:
        """回傳每個字詞: [{'start': float, 'end': float, 'word': str}, ...]"""
        out = self.model.transcribe_with_vad(
            [audio_path],
            lang_codes=[lang],
            tasks=["transcribe"],
            batch_size=16,
        )

        words: List[Dict] = []
        # out[0] 為第一個檔案的段落清單
        for segment in out[0]:
            # 若有 word-level 時間戳
            for w in segment.get("word_timestamps", []):
                words.append({
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                    "word": w["word"],
                })
            # 若該版本沒有字詞級，退回段落級
            if not segment.get("word_timestamps"):
                words.append({
                    "start": float(segment["start"]),
                    "end": float(segment["end"]),
                    "word": segment["text"],
                })
        return words
