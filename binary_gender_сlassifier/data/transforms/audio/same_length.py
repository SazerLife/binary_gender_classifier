from typing import Any, Dict

import torch.nn.functional as F
from torch import Tensor


class SameLength:
    def __init__(self, length: int, padding_value: float, dim: int = 0):
        self.__length = length
        self.__value = padding_value
        self.__dim = dim

    def __call__(self, data: Dict[str, Any]):
        audio: Tensor = data["audio"]
        if audio.shape[self.__dim] < self.__length:
            pad = [0 for _ in range(len(audio.shape) * 2)]
            left_idx = -2 * self.__dim - 2
            right_idx = -2 * self.__dim - 1
            pad[left_idx] = 0
            pad[right_idx] = self.__length - audio.shape[self.__dim]
            audio = F.pad(audio, pad, "constant", self.__value)
        else:
            audio = audio.narrow(self.__dim, 0, self.__length)

        data["audio"] = audio
        return data
