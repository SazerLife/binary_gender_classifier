from typing import Any, Dict, Optional

import torchaudio.transforms as T
from torch import Tensor


class MelSpectrogram:
    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        win_length: Optional[int],
        hop_length: Optional[int],
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ):
        self.__transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )

    def __call__(self, data: Dict[str, Any]):
        audio: Tensor = data["audio"]
        melspec = self.__transform(audio)
        data["audio"] = melspec
        return data
