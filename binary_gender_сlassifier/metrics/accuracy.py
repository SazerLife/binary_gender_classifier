import math
from typing import Iterable, List, Tuple, Dict

import torch
from torch import Tensor
from torchmetrics.functional import accuracy


class Accuracy:
    def __init__(self):
        pass

    def __call__(
        self,
        batched_outputs: List[Dict[str, Tensor]],
        batched_targets: List[Dict[str, Tensor]],
    ):
        outputs = torch.cat([batch["labels"] for batch in batched_outputs])
        targets = torch.cat([batch["labels"] for batch in batched_targets])
        acc = accuracy(outputs, targets)
        return acc
