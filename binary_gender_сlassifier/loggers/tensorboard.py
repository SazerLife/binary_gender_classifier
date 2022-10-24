from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from torch import Tensor, nn
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    def __init__(self, folder_to_save: str):
        self.__folder_to_save = Path(folder_to_save)
        self.__folder_to_save.mkdir(parents=True, exist_ok=True)
        self.__writer = SummaryWriter(self.__folder_to_save)

    def logging(
        self,
        training_objects: Tuple[nn.Module, nn.Module, int, int],
        train_data,
        val_data,
        train_metrics_values: Dict[str, Any],
        val_metrics_values: Dict[str, Any],
    ) -> None:
        _, _, _, iteration = training_objects
        _, _, train_loss_values = train_data
        _, _, val_loss_values = val_data

        self.__writer.add_scalar(
            "Loss/train", sum(train_loss_values) / len(train_loss_values), iteration
        )
        self.__writer.add_scalar(
            "Loss/val", sum(val_loss_values) / len(val_loss_values), iteration
        )
        for metric_name, train_metric, val_metric in zip(
            train_metrics_values.keys(),
            train_metrics_values.values(),
            val_metrics_values.values(),
        ):
            self.__writer.add_scalar(f"{metric_name}/train", train_metric, iteration)
            self.__writer.add_scalar(f"{metric_name}/val", val_metric, iteration)
