from typing import Tuple, Dict, List
import torch
from torch import Tensor


class DefaultCollator:
    def __init__(self):
        pass

    def __call__(self, batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        audio_batch = [sample["audio"] for sample in batch]
        labels_batch = [sample["label"] for sample in batch]
        return {"audio": audio_batch, "labels": labels_batch}

    def __to_device(self, value: Tensor, device: str) -> Tensor:
        value = value.contiguous()
        value = value.to(device, non_blocking=False)
        return value

    def x_y_split(
        self, batch: Dict[str, Tensor]
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        audio_batch = list()
        for audio in batch["audio"]:
            audio_batch.append(audio.unsqueeze(0))
        x = {"audio": torch.cat(audio_batch, dim=0)}
        y = {"labels": torch.as_tensor(batch["labels"])}
        return (x, y)

    def x_to_device(self, x: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
        x["audio"] = self.__to_device(x["audio"], device)
        return x

    def y_to_device(self, y: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
        y["labels"] = self.__to_device(y["labels"], device)
        return y
