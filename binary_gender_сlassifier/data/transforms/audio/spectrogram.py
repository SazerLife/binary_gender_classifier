from typing import Any, Dict, Optional

import torchaudio.transforms as T
from torch import Tensor


class Spectrogram:
    def __init__(
        self,
        n_fft: int,
        win_length: Optional[int],
        hop_length: Optional[int],
        pad: int = 0,
        power: Optional[float] = 2.0,
        normalized: bool = False,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: bool = True,
    ):
        self.__transform = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            pad=pad,
            power=power,
            normalized=normalized,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
        )

    def __call__(self, data: Dict[str, Any]):
        audio: Tensor = data["audio"]
        spec = self.__transform(audio)
        data["audio"] = spec
        return data
