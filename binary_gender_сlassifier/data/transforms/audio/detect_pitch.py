from typing import Any, Dict
import torchaudio


class DetectPitch:
    def __init__(self, win_length: int = 100, freq_low: int = 50, freq_high: int = 700):
        self.__freq_low = freq_low
        self.__freq_high = freq_high
        self.__win_length = win_length

    def __call__(self, data: Dict[str, Any]):
        audio, sr = data["audio"], data["sr"]
        pitch = torchaudio.functional.detect_pitch_frequency(
            audio,
            sr,
            freq_high=self.__freq_high,
            freq_low=self.__freq_low,
            win_length=self.__win_length,
        )
        data["audio"] = pitch.unsqueeze(0)
        return data
