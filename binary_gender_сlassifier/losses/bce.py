from typing import Dict

import torch
from torch import Tensor


class BCELoss:
    def __init__(self):
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def __call__(
        self,
        output: Dict[str, Tensor],
        target: Dict[str, Tensor],
    ) -> Tensor:
        output = output["labels"].view(-1, 1)
        target = target["labels"].view(-1, 1).to(torch.float32)
        loss = self.loss_fn(output, target)
        return loss
